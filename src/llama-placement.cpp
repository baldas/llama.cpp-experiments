#include "llama-placement.h"

#include "ggml.h"
#include "llama-model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Log macros usually provided by llama-impl.h or similar, but we'll use stderr for now to be safe and independent
#define LP_LOG_INFO(fmt, ...)  fprintf(stderr, "[placement] " fmt "\n", ##__VA_ARGS__)
#define LP_LOG_ERROR(fmt, ...) fprintf(stderr, "[placement] ERROR: " fmt "\n", ##__VA_ARGS__)

// --------------------------------------------------------------------------
// Configuration Parsing
// --------------------------------------------------------------------------

enum class TensorPlacement { DRAM, OPTANE };

struct LayerPlacement {
    TensorPlacement attn;
    TensorPlacement ffn;
};

struct LayerConfig {
    TensorPlacement               embed_placement  = TensorPlacement::DRAM;
    TensorPlacement               output_placement = TensorPlacement::DRAM;
    std::map<int, LayerPlacement> layer_placements;
};

static TensorPlacement parse_placement_str(const std::string & s) {
    if (s == "optane" || s == "pmem") {
        return TensorPlacement::OPTANE;
    }
    return TensorPlacement::DRAM;
}

static bool parse_placement_config(const char * config_str, LayerConfig & config, int n_layers) {
    // Example: "embed:dram,0-7:dram,8-31.attn:dram,8-31.ffn:optane,output:dram"
    std::string s(config_str);
    std::replace(s.begin(), s.end(), ',', ' ');
    std::stringstream ss(s);
    std::string       segment;

    // Default all layers to DRAM first
    for (int i = 0; i < n_layers; ++i) {
        config.layer_placements[i] = { TensorPlacement::DRAM, TensorPlacement::DRAM };
    }

    while (ss >> segment) {
        size_t colon = segment.find(':');
        if (colon == std::string::npos) {
            LP_LOG_ERROR("Invalid config segment: %s", segment.c_str());
            return false;
        }
        std::string     key = segment.substr(0, colon);
        std::string     val = segment.substr(colon + 1);
        TensorPlacement p   = parse_placement_str(val);

        if (key == "embed") {
            config.embed_placement = p;
        } else if (key == "output") {
            config.output_placement = p;
        } else {
            // Check for suffixes .attn and .ffn
            bool set_attn = true;
            bool set_ffn  = true;

            const std::string suffix_attn = ".attn";
            const std::string suffix_ffn  = ".ffn";

            if (key.length() > suffix_attn.length() &&
                key.compare(key.length() - suffix_attn.length(), suffix_attn.length(), suffix_attn) == 0) {
                set_ffn = false;
                key     = key.substr(0, key.length() - suffix_attn.length());
            } else if (key.length() > suffix_ffn.length() &&
                       key.compare(key.length() - suffix_ffn.length(), suffix_ffn.length(), suffix_ffn) == 0) {
                set_attn = false;
                key      = key.substr(0, key.length() - suffix_ffn.length());
            }

            // Check for range like "0-7" or single "5"
            size_t dash  = key.find('-');
            int    start = 0, end = 0;
            if (dash != std::string::npos) {
                try {
                    start = std::stoi(key.substr(0, dash));
                    end   = std::stoi(key.substr(dash + 1));
                } catch (...) {
                    LP_LOG_ERROR("Invalid range: %s", key.c_str());
                    return false;
                }
            } else {
                try {
                    start = std::stoi(key);
                    end   = start;
                } catch (...) {
                    LP_LOG_ERROR("Invalid layer index: %s", key.c_str());
                    return false;
                }
            }

            for (int i = start; i <= end; ++i) {
                if (i >= 0 && i < n_layers) {
                    if (set_attn) {
                        config.layer_placements[i].attn = p;
                    }
                    if (set_ffn) {
                        config.layer_placements[i].ffn = p;
                    }
                }
            }
        }
    }
    return true;
}

// --------------------------------------------------------------------------
// Persistent Memory Storage (DAX)
// --------------------------------------------------------------------------

class PmemWeightStore {
  public:
    PmemWeightStore(const char * path) : path_(path) {}

    // Delete copy/move to prevent accidental unmapping
    PmemWeightStore(const PmemWeightStore &)             = delete;
    PmemWeightStore & operator=(const PmemWeightStore &) = delete;

    ~PmemWeightStore() {
        if (should_unmap_) {
            if (addr_ && addr_ != MAP_FAILED) {
                munmap(addr_, size_);
            }
            if (fd_ >= 0) {
                close(fd_);
            }
        }
    }

    // Call this if you want the mapping to persist after object destruction (e.g. process duration)
    void release() { should_unmap_ = false; }

    bool init(size_t required_size) {
        fd_ = open(path_.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) {
            LP_LOG_ERROR("Failed to open pmem file: %s", path_.c_str());
            return false;
        }

        // Pre-allocate space
        if (ftruncate(fd_, required_size) != 0) {
            LP_LOG_ERROR("Failed to resize pmem file to %zu bytes", required_size);
            return false;
        }

// Map with DAX flags
#ifndef MAP_SYNC
#    define MAP_SYNC 0x080000
#endif
#ifndef MAP_SHARED_VALIDATE
#    define MAP_SHARED_VALIDATE 0x03
#endif

        //int flags = MAP_SHARED_VALIDATE | MAP_SYNC;   // we don't need
        // persistency
        int flags = MAP_SHARED;
        addr_     = mmap(NULL, required_size, PROT_READ | PROT_WRITE, flags, fd_, 0);
        if (addr_ == MAP_FAILED) {
          perror("mmap failed");
          return false;
        }

        /*
        if (addr_ == MAP_FAILED) {
            // Fallback for non-DAX filesystems (dev/testing)
            LP_LOG_ERROR("mmap with MAP_SYNC failed. Is this a DAX filesystem? Falling back to standard MAP_SHARED.");
            addr_ = mmap(NULL, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (addr_ == MAP_FAILED) {
                perror("mmap failed");
                return false;
            }
        }
        */

        size_           = required_size;
        current_offset_ = 0;
        base_ptr_       = static_cast<uint8_t *>(addr_);

        return true;
    }

    void * allocate(size_t bytes) {
        // align allocation to 64 bytes
        size_t padding = (64 - (current_offset_ % 64)) % 64;
        if (current_offset_ + padding + bytes > size_) {
            LP_LOG_ERROR("PmemStore out of memory! Requested %zu, available %zu", bytes, size_ - current_offset_);
            return nullptr;
        }
        current_offset_ += padding;
        void * ptr = base_ptr_ + current_offset_;
        current_offset_ += bytes;
        return ptr;
    }

  private:
    std::string path_;
    int         fd_             = -1;
    void *      addr_           = MAP_FAILED;
    uint8_t *   base_ptr_       = nullptr;
    size_t      size_           = 0;
    size_t      current_offset_ = 0;
    bool        should_unmap_   = true;
};

// --------------------------------------------------------------------------
// GGUF tensor offset lookup
// --------------------------------------------------------------------------

struct GgufContextWrapper {
    gguf_context * ctx = nullptr;

    GgufContextWrapper(const char * path) {
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ nullptr,
        };
        ctx = gguf_init_from_file(path, params);
    }

    ~GgufContextWrapper() {
        if (ctx) {
            gguf_free(ctx);
        }
    }

    uint64_t get_tensor_offset(const char * name) {
        int idx = gguf_find_tensor(ctx, name);
        if (idx < 0) {
            return -1;
        }
        return gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);
    }
};

// --------------------------------------------------------------------------
// Main Placement Logic
// --------------------------------------------------------------------------

int llama_apply_layer_placement(struct llama_model * model,
                                const char *         gguf_path,
                                const char *         pmem_path,
                                const char *         config_str) {
    if (!model || !gguf_path || !pmem_path || !config_str) {
        return -1;
    }

    LP_LOG_INFO("Applying layer placement: %s", config_str);

    // 1. Parse config
    LayerConfig config;
    if (!parse_placement_config(config_str, config, model->layers.size())) {
        return -1;
    }

    // 2. Identify tensors to move and calculate required pmem size
    // 5. Place Tensors
    size_t bytes_dram_attn    = 0;
    size_t bytes_dram_ffn     = 0;
    size_t bytes_dram_other   = 0;
    size_t bytes_optane_attn  = 0;
    size_t bytes_optane_ffn   = 0;
    size_t bytes_optane_other = 0;

    // Helper to categorize tensors (very rough, could be improved by tracking usage in loop 2)
    // Actually, we can't easily map back from tensor* to category unless we tracked it.
    // Let's just track total DRAM vs Optane for now, and maybe rough category totals if possible.
    // Since we just want total summary for now as requested by user ("DRAM vs Optane breakdown for attention and FFN separately"),
    // we should track it during the "check_tensor" phase or just track it here by knowing what we are processing.
    // BUT calculate loop 2 already pushed to "tensors_to_process" without category info.
    // Re-factor: Let's store category in tensors_to_process or just run the placement loop directly on model tensors again?
    // Running directly is risky if we have duplicates (shared tensors).
    // Better: Add a "category" enum to tensors_to_process.

    // Let's Refactor Loop 2 and 5 to be one pass or carry metadata.
    // Actually, the simplest way is to just do the placement inside the loop over layers/tensors directly,
    // instead of building a list.
    // BUT we need to calculate total size first to init the PmemWeightStore.
    // So we DO need two passes.

    enum TensorCategory { CAT_EMBED, CAT_ATTN, CAT_FFN, CAT_OUTPUT, CAT_OTHER };

    std::vector<std::tuple<ggml_tensor *, TensorPlacement, TensorCategory>> placement_list;
    size_t                                                                  total_pmem_needed = 0;

    auto add_tensor = [&](ggml_tensor * t, TensorPlacement p, TensorCategory c) {
        if (!t) {
            return;
        }
        if (p == TensorPlacement::OPTANE) {
            size_t nbytes = ggml_nbytes(t);
            total_pmem_needed += nbytes + 64;
        }
        placement_list.push_back({ t, p, c });
    };

    // Embeddings
    add_tensor(model->tok_embd, config.embed_placement, CAT_EMBED);
    add_tensor(model->type_embd, config.embed_placement, CAT_EMBED);
    add_tensor(model->pos_embd, config.embed_placement, CAT_EMBED);
    add_tensor(model->tok_norm, config.embed_placement, CAT_EMBED);
    add_tensor(model->tok_norm_b, config.embed_placement, CAT_EMBED);

    // Layers
    for (int i = 0; i < (int) model->layers.size(); ++i) {
        LayerPlacement p = config.layer_placements[i];
        auto &         l = model->layers[i];

        // Attn
        add_tensor(l.attn_norm, p.attn, CAT_ATTN);
        add_tensor(l.attn_norm_b, p.attn, CAT_ATTN);
        add_tensor(l.attn_norm_2, p.attn, CAT_ATTN);
        add_tensor(l.attn_norm_2_b, p.attn, CAT_ATTN);
        add_tensor(l.attn_q_norm, p.attn, CAT_ATTN);
        add_tensor(l.attn_q_norm_b, p.attn, CAT_ATTN);
        add_tensor(l.attn_k_norm, p.attn, CAT_ATTN);
        add_tensor(l.attn_k_norm_b, p.attn, CAT_ATTN);
        add_tensor(l.attn_out_norm, p.attn, CAT_ATTN);
        add_tensor(l.attn_out_norm_b, p.attn, CAT_ATTN);

        add_tensor(l.wq, p.attn, CAT_ATTN);
        add_tensor(l.wk, p.attn, CAT_ATTN);
        add_tensor(l.wv, p.attn, CAT_ATTN);
        add_tensor(l.wo, p.attn, CAT_ATTN);
        add_tensor(l.wqkv, p.attn, CAT_ATTN);

        add_tensor(l.bq, p.attn, CAT_ATTN);
        add_tensor(l.bk, p.attn, CAT_ATTN);
        add_tensor(l.bv, p.attn, CAT_ATTN);
        add_tensor(l.bo, p.attn, CAT_ATTN);
        add_tensor(l.bqkv, p.attn, CAT_ATTN);

        // FFN
        add_tensor(l.ffn_norm, p.ffn, CAT_FFN);
        add_tensor(l.ffn_norm_b, p.ffn, CAT_FFN);
        add_tensor(l.ffn_post_norm, p.ffn, CAT_FFN);

        add_tensor(l.ffn_gate, p.ffn, CAT_FFN);  // w1
        add_tensor(l.ffn_down, p.ffn, CAT_FFN);  // w2
        add_tensor(l.ffn_up, p.ffn, CAT_FFN);    // w3

        add_tensor(l.ffn_gate_b, p.ffn, CAT_FFN);
        add_tensor(l.ffn_down_b, p.ffn, CAT_FFN);
        add_tensor(l.ffn_up_b, p.ffn, CAT_FFN);
    }

    // Output
    add_tensor(model->output_norm, config.output_placement, CAT_OUTPUT);
    add_tensor(model->output_norm_b, config.output_placement, CAT_OUTPUT);
    add_tensor(model->output, config.output_placement, CAT_OUTPUT);
    add_tensor(model->output_b, config.output_placement, CAT_OUTPUT);

    // 3. Initialize Pmem Store if needed
    PmemWeightStore * pmem = nullptr;
    if (total_pmem_needed > 0) {
        pmem = new PmemWeightStore(pmem_path);
        // Add 1MB slack
        if (!pmem->init(total_pmem_needed + 1024 * 1024)) {
            delete pmem;
            return -1;
        }
    }

    // 4. Open GGUF file for reading data
    int gguf_fd = open(gguf_path, O_RDONLY);
    if (gguf_fd < 0) {
        LP_LOG_ERROR("Failed to open GGUF file: %s", gguf_path);
        if (pmem) {
            delete pmem;
        }
        return -1;
    }

    // Helper to find offsets
    GgufContextWrapper gguf_meta(gguf_path);

    for (auto & item : placement_list) {
        ggml_tensor *   t      = std::get<0>(item);
        TensorPlacement p      = std::get<1>(item);
        TensorCategory  c      = std::get<2>(item);
        size_t          nbytes = ggml_nbytes(t);

        size_t * tracker_dram   = &bytes_dram_other;
        size_t * tracker_optane = &bytes_optane_other;

        if (c == CAT_ATTN) {
            tracker_dram   = &bytes_dram_attn;
            tracker_optane = &bytes_optane_attn;
        } else if (c == CAT_FFN) {
            tracker_dram   = &bytes_dram_ffn;
            tracker_optane = &bytes_optane_ffn;
        }

        if (p == TensorPlacement::DRAM) {
            *tracker_dram += nbytes;
        } else {
            // OPTANE
            // 1. Allocate in pmem
            void * pmem_ptr = pmem->allocate(nbytes);
            if (!pmem_ptr) {
                close(gguf_fd);
                delete pmem;  // Unmaps logic
                return -1;
            }

            // 2. Read data from GGUF file to pmem
            uint64_t file_offset = gguf_meta.get_tensor_offset(t->name);
            if (file_offset == (uint64_t) -1) {
                LP_LOG_ERROR("Failed to find tensor offset for %s", t->name);
                continue;
            }

            ssize_t read_bytes = pread(gguf_fd, pmem_ptr, nbytes, file_offset);
            if (read_bytes != (ssize_t) nbytes) {
                LP_LOG_ERROR("Failed to read tensor data for %s", t->name);
                close(gguf_fd);
                delete pmem;
                return -1;
            }

            // 3. Advise OS we don't need the original page cache for this range
            uintptr_t old_addr         = (uintptr_t) t->data;
            size_t    page_size        = sysconf(_SC_PAGESIZE);
            uintptr_t old_addr_aligned = old_addr & ~(page_size - 1);
            size_t    len_aligned      = (old_addr + nbytes - old_addr_aligned + page_size - 1) & ~(page_size - 1);

            madvise((void *) old_addr_aligned, len_aligned, MADV_DONTNEED);

            // 4. Redirect pointer
            t->data = pmem_ptr;
            *tracker_optane += nbytes;
        }
    }

    close(gguf_fd);

    auto to_mib = [](size_t b) {
        return b / (1024.0 * 1024.0);
    };

    LP_LOG_INFO("Placement Complete.");
    LP_LOG_INFO("  DRAM Total:   %.2f MiB", to_mib(bytes_dram_attn + bytes_dram_ffn + bytes_dram_other));
    LP_LOG_INFO("    Attention:  %.2f MiB", to_mib(bytes_dram_attn));
    LP_LOG_INFO("    FFN:        %.2f MiB", to_mib(bytes_dram_ffn));
    LP_LOG_INFO("    Other:      %.2f MiB", to_mib(bytes_dram_other));
    LP_LOG_INFO("  Optane Total: %.2f MiB", to_mib(bytes_optane_attn + bytes_optane_ffn + bytes_optane_other));
    LP_LOG_INFO("    Attention:  %.2f MiB", to_mib(bytes_optane_attn));
    LP_LOG_INFO("    FFN:        %.2f MiB", to_mib(bytes_optane_ffn));
    LP_LOG_INFO("    Other:      %.2f MiB", to_mib(bytes_optane_other));

    if (pmem) {
        // Important: Release ownership so it doesn't unmap on destruction of the pointer or program exit
        // We want it to leak for the duration of the process.
        pmem->release();
        // We can safely delete the object wrapper now if it's just a file descriptor holder,
        // but we need to keep the mapping.
        // Wait, if I delete `pmem` (the pointer), the destructor calls `munmap` if strictly following C++.
        // `pmem->release()` sets `should_unmap_ = false`, so `delete pmem` will NOT unmap, but WILL close fd.
        // Closing fd is fine after mmap.
    }

    // We intentionally do not delete pmem to be extra safe against any destructor logic if we change it later,
    // but `delete pmem` with `release()` is cleaner.
    // Actually, let's just leak the PmemWeightStore* pointer too. It's one pointer per model load. Negligible.
    // (Or use a static vector to hold them if we wanted to be pedantic).

    return 0;
}
