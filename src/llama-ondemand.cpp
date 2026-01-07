#include "llama-ondemand.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if LLAMA_ONDEMAND_SUPPORTED

#    include <errno.h>
#    include <fcntl.h>
#    include <linux/userfaultfd.h>
#    include <poll.h>
#    include <sys/ioctl.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <time.h>
#    include <unistd.h>

// Wrapper for userfaultfd syscall (may not be in glibc headers)
#    ifndef __NR_userfaultfd
#        if defined(__x86_64__)
#            define __NR_userfaultfd 323
#        elif defined(__i386__)
#            define __NR_userfaultfd 374
#        elif defined(__aarch64__)
#            define __NR_userfaultfd 282
#        elif defined(__arm__)
#            define __NR_userfaultfd 388
#        else
#            error "userfaultfd syscall number not defined for this architecture"
#        endif
#    endif

llama_ondemand_params llama_ondemand_default_params() {
    return llama_ondemand_params{
        /* .max_resident_bytes  = */ 0,  // unlimited
        /* .page_size           = */ 0,  // use system default
        /* .enable_lru_eviction = */ false,
        /* .prefetch_embeddings = */ true,
        /* .prefetch_layers     = */ 0,
    };
}

struct llama_ondemand_loader::impl {
    llama_file *          file;
    size_t                total_size;
    llama_ondemand_params params;

    void *            base_addr = nullptr;
    size_t            page_size = 0;
    int               uffd      = -1;
    std::thread       handler_thread;
    std::atomic<bool> running{ false };
    std::atomic<bool> started{ false };

    // LRU tracking
    mutable std::mutex         lru_mutex;
    std::map<size_t, uint64_t> page_access_time;  // page_offset -> access_time
    std::atomic<size_t>        resident_bytes{ 0 };
    std::atomic<size_t>        fault_count{ 0 };
    std::atomic<size_t>        eviction_count{ 0 };

    // Page buffer for loading
    std::vector<uint8_t> page_buffer;

    impl(llama_file * file, size_t total_size, const llama_ondemand_params & params) :
        file(file),
        total_size(total_size),
        params(params) {
        // Determine page size
        page_size = params.page_size;
        if (page_size == 0) {
            page_size = (size_t) sysconf(_SC_PAGESIZE);
        }

        // Validate page size is a power of 2
        if (page_size == 0 || (page_size & (page_size - 1)) != 0) {
            throw std::runtime_error("page_size must be a power of 2");
        }

        // Allocate page buffer for potential manual loading
        page_buffer.resize(page_size);

        int fd = file->file_id();

        // Use file-backed mmap WITHOUT MAP_POPULATE - this provides natural
        // on-demand page loading via kernel page faults (no userfaultfd needed)
        base_addr = mmap(nullptr, total_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base_addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed for on-demand allocation: %s", strerror(errno)));
        }

        // Tell the kernel we'll access randomly (disable readahead)
        // This is key for on-demand behavior - we only load pages when accessed
        if (posix_madvise(base_addr, total_size, POSIX_MADV_RANDOM) != 0) {
            LLAMA_LOG_WARN("%s: posix_madvise POSIX_MADV_RANDOM failed: %s\n", __func__, strerror(errno));
        }

        // If LRU eviction is enabled, we'll use userfaultfd to track pages
        // Otherwise we skip userfaultfd entirely and rely on kernel mmap magic
        if (params.enable_lru_eviction && params.max_resident_bytes > 0) {
            // Create userfaultfd for tracking (optional - for LRU eviction only)
            uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
            if (uffd < 0) {
                LLAMA_LOG_WARN("%s: userfaultfd creation failed (LRU eviction disabled): %s\n", __func__,
                               strerror(errno));
                // Continue without userfaultfd - on-demand still works via mmap
            } else {
                // Configure userfaultfd API
                struct uffdio_api api = {};
                api.api               = UFFD_API;
                api.features          = 0;
                if (ioctl(uffd, UFFDIO_API, &api) < 0) {
                    LLAMA_LOG_WARN("%s: userfaultfd API setup failed: %s\n", __func__, strerror(errno));
                    close(uffd);
                    uffd = -1;
                }
            }
        }

        LLAMA_LOG_INFO("%s: on-demand loader initialized, size=%.2f MiB, page_size=%zu, lru=%s\n", __func__,
                       total_size / 1024.0 / 1024.0, page_size, (uffd >= 0) ? "enabled" : "disabled");
    }

    ~impl() {
        stop();

        if (uffd >= 0) {
            // Unregister the region first
            struct uffdio_register unreg = {};
            unreg.range.start            = (uintptr_t) base_addr;
            unreg.range.len              = total_size;
            ioctl(uffd, UFFDIO_UNREGISTER, &unreg);

            close(uffd);
        }

        if (base_addr && base_addr != MAP_FAILED) {
            munmap(base_addr, total_size);
        }

        LLAMA_LOG_INFO("%s: on-demand loader destroyed, faults=%zu, evictions=%zu\n", __func__, fault_count.load(),
                       eviction_count.load());
    }

    bool start() {
        if (started.load()) {
            return true;
        }

        // With file-backed mmap, on-demand loading works automatically
        // via kernel page faults. The handler thread is only needed
        // if we're using userfaultfd for LRU tracking.
        if (uffd >= 0) {
            running.store(true);
            handler_thread = std::thread(&impl::handler_loop, this);
            LLAMA_LOG_DEBUG("%s: fault handler thread started\n", __func__);
        }

        started.store(true);
        return true;
    }

    void stop() {
        if (!started.load()) {
            return;
        }

        running.store(false);

        // Wake up the handler thread by writing to uffd
        // This is a workaround since poll() may be blocking
        if (handler_thread.joinable()) {
            handler_thread.join();
        }

        started.store(false);
        LLAMA_LOG_DEBUG("%s: fault handler thread stopped\n", __func__);
    }

    void handler_loop() {
        LLAMA_LOG_DEBUG("%s: handler loop starting\n", __func__);

        while (running.load()) {
            struct pollfd pfd = {};
            pfd.fd            = uffd;
            pfd.events        = POLLIN;

            int ret = poll(&pfd, 1, 100);  // 100ms timeout for clean shutdown

            if (ret < 0) {
                if (errno == EINTR) {
                    continue;
                }
                LLAMA_LOG_ERROR("%s: poll failed: %s\n", __func__, strerror(errno));
                break;
            }

            if (ret == 0) {
                continue;  // timeout
            }

            if (!(pfd.revents & POLLIN)) {
                continue;
            }

            // Read the fault message
            struct uffd_msg msg   = {};
            ssize_t         bytes = read(uffd, &msg, sizeof(msg));
            if (bytes != sizeof(msg)) {
                if (errno == EAGAIN) {
                    continue;
                }
                LLAMA_LOG_ERROR("%s: read failed: %s\n", __func__, strerror(errno));
                continue;
            }

            if (msg.event != UFFD_EVENT_PAGEFAULT) {
                LLAMA_LOG_WARN("%s: unexpected event: %d\n", __func__, msg.event);
                continue;
            }

            handle_fault(msg.arg.pagefault.address);
        }

        LLAMA_LOG_DEBUG("%s: handler loop exiting\n", __func__);
    }

    void handle_fault(uintptr_t fault_addr) {
        // Calculate page-aligned offset
        size_t offset      = fault_addr - (uintptr_t) base_addr;
        size_t page_offset = offset & ~(page_size - 1);

        if (page_offset >= total_size) {
            LLAMA_LOG_ERROR("%s: fault address out of bounds: %p\n", __func__, (void *) fault_addr);
            return;
        }

        fault_count.fetch_add(1, std::memory_order_relaxed);

        // Check memory pressure and evict if needed
        if (params.enable_lru_eviction && params.max_resident_bytes > 0) {
            while (resident_bytes.load(std::memory_order_relaxed) + page_size > params.max_resident_bytes) {
                if (!evict_lru_page()) {
                    break;  // No pages to evict
                }
            }
        }

        // Calculate how much to read (may be less than page_size at end of file)
        size_t read_size = std::min(page_size, total_size - page_offset);

        // Read data from file
        file->seek(page_offset, SEEK_SET);
        file->read_raw(page_buffer.data(), read_size);

        // Zero the rest of the page if partial
        if (read_size < page_size) {
            memset(page_buffer.data() + read_size, 0, page_size - read_size);
        }

        // Copy to the faulting address
        struct uffdio_copy copy = {};
        copy.dst                = (uintptr_t) base_addr + page_offset;
        copy.src                = (uintptr_t) page_buffer.data();
        copy.len                = page_size;
        copy.mode               = 0;

        if (ioctl(uffd, UFFDIO_COPY, &copy) < 0) {
            // EEXIST means the page was already resolved (race condition)
            if (errno != EEXIST) {
                LLAMA_LOG_ERROR("%s: UFFDIO_COPY failed: %s\n", __func__, strerror(errno));
            }
            return;
        }

        // Track for LRU
        {
            std::lock_guard<std::mutex> lock(lru_mutex);
            page_access_time[page_offset] = get_time_ns();
        }
        resident_bytes.fetch_add(page_size, std::memory_order_relaxed);
    }

    bool evict_lru_page() {
        size_t page_offset;

        {
            std::lock_guard<std::mutex> lock(lru_mutex);

            if (page_access_time.empty()) {
                return false;
            }

            // Find the oldest page
            auto oldest = page_access_time.begin();
            for (auto it = page_access_time.begin(); it != page_access_time.end(); ++it) {
                if (it->second < oldest->second) {
                    oldest = it;
                }
            }

            page_offset = oldest->first;
            page_access_time.erase(oldest);
        }

        // Tell the kernel to discard the page
        void * page_addr = (uint8_t *) base_addr + page_offset;
        if (madvise(page_addr, page_size, MADV_DONTNEED) != 0) {
            LLAMA_LOG_WARN("%s: madvise MADV_DONTNEED failed: %s\n", __func__, strerror(errno));
        }

        resident_bytes.fetch_sub(page_size, std::memory_order_relaxed);
        eviction_count.fetch_add(1, std::memory_order_relaxed);

        return true;
    }

    void prefetch_range(size_t offset, size_t len) {
        // Align to page boundaries
        size_t start_page = offset & ~(page_size - 1);
        size_t end_page   = (offset + len + page_size - 1) & ~(page_size - 1);
        end_page          = std::min(end_page, total_size);

        for (size_t page_off = start_page; page_off < end_page; page_off += page_size) {
            // Touch the page to trigger loading
            volatile char * p = (volatile char *) base_addr + page_off;
            (void) *p;
        }
    }

    void evict_range(size_t offset, size_t len) {
        // Align to page boundaries
        size_t start_page = offset & ~(page_size - 1);
        size_t end_page   = (offset + len + page_size - 1) & ~(page_size - 1);
        end_page          = std::min(end_page, total_size);

        for (size_t page_off = start_page; page_off < end_page; page_off += page_size) {
            void * page_addr = (uint8_t *) base_addr + page_off;
            madvise(page_addr, page_size, MADV_DONTNEED);

            std::lock_guard<std::mutex> lock(lru_mutex);
            auto                        it = page_access_time.find(page_off);
            if (it != page_access_time.end()) {
                page_access_time.erase(it);
                resident_bytes.fetch_sub(page_size, std::memory_order_relaxed);
                eviction_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    static uint64_t get_time_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
    }
};

// Public API implementations
llama_ondemand_loader::llama_ondemand_loader(llama_file *                  file,
                                             size_t                        total_size,
                                             const llama_ondemand_params & params) :
    pimpl(std::make_unique<impl>(file, total_size, params)) {}

llama_ondemand_loader::~llama_ondemand_loader() = default;

void * llama_ondemand_loader::get_base_addr() const {
    return pimpl->base_addr;
}

size_t llama_ondemand_loader::get_size() const {
    return pimpl->total_size;
}

bool llama_ondemand_loader::start() {
    return pimpl->start();
}

void llama_ondemand_loader::stop() {
    pimpl->stop();
}

bool llama_ondemand_loader::is_running() const {
    return pimpl->running.load();
}

void llama_ondemand_loader::prefetch_range(size_t offset, size_t len) {
    pimpl->prefetch_range(offset, len);
}

void llama_ondemand_loader::evict_range(size_t offset, size_t len) {
    pimpl->evict_range(offset, len);
}

size_t llama_ondemand_loader::get_resident_bytes() const {
    return pimpl->resident_bytes.load(std::memory_order_relaxed);
}

size_t llama_ondemand_loader::get_fault_count() const {
    return pimpl->fault_count.load(std::memory_order_relaxed);
}

size_t llama_ondemand_loader::get_eviction_count() const {
    return pimpl->eviction_count.load(std::memory_order_relaxed);
}

bool llama_ondemand_loader::is_supported() {
    return true;
}

#else   // !LLAMA_ONDEMAND_SUPPORTED

// Stub implementation for non-Linux platforms

llama_ondemand_params llama_ondemand_default_params() {
    return llama_ondemand_params{ 0, 0, false, false, 0 };
}

struct llama_ondemand_loader::impl {
    impl(llama_file *, size_t, const llama_ondemand_params &) {
        throw std::runtime_error("on-demand loading is not supported on this platform");
    }
};

llama_ondemand_loader::llama_ondemand_loader(llama_file *                  file,
                                             size_t                        total_size,
                                             const llama_ondemand_params & params) :
    pimpl(std::make_unique<impl>(file, total_size, params)) {}

llama_ondemand_loader::~llama_ondemand_loader() = default;

void * llama_ondemand_loader::get_base_addr() const {
    return nullptr;
}

size_t llama_ondemand_loader::get_size() const {
    return 0;
}

bool llama_ondemand_loader::start() {
    return false;
}

void llama_ondemand_loader::stop() {}

bool llama_ondemand_loader::is_running() const {
    return false;
}

void llama_ondemand_loader::prefetch_range(size_t, size_t) {}

void llama_ondemand_loader::evict_range(size_t, size_t) {}

size_t llama_ondemand_loader::get_resident_bytes() const {
    return 0;
}

size_t llama_ondemand_loader::get_fault_count() const {
    return 0;
}

size_t llama_ondemand_loader::get_eviction_count() const {
    return 0;
}

bool llama_ondemand_loader::is_supported() {
    return false;
}

#endif  // LLAMA_ONDEMAND_SUPPORTED
