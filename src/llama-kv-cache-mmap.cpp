#include "llama-kv-cache-mmap.h"

#include "llama-impl.h"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>

#    include <cerrno>
#endif

llama_kv_cache_mmap::llama_kv_cache_mmap(const std::string & path, size_t size) : mapped_size(size) {
    if (size == 0) {
        throw std::runtime_error("llama_kv_cache_mmap: size cannot be 0");
    }

#ifdef _WIN32
    // Windows implementation using CreateFileMapping

    // Open or create file
    file_handle = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                              OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (file_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("llama_kv_cache_mmap: failed to open file: " + path);
    }

    // Set file size
    LARGE_INTEGER file_size;
    file_size.QuadPart = static_cast<LONGLONG>(size);
    if (!SetFilePointerEx(file_handle, file_size, NULL, FILE_BEGIN) || !SetEndOfFile(file_handle)) {
        CloseHandle(file_handle);
        file_handle = nullptr;
        throw std::runtime_error("llama_kv_cache_mmap: failed to resize file");
    }

    // Create file mapping
    map_handle = CreateFileMappingA(file_handle, NULL, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                    static_cast<DWORD>(size & 0xFFFFFFFF), NULL);

    if (map_handle == NULL) {
        CloseHandle(file_handle);
        file_handle = nullptr;
        throw std::runtime_error("llama_kv_cache_mmap: failed to create file mapping");
    }

    // Map view
    ptr = MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (ptr == NULL) {
        CloseHandle(map_handle);
        CloseHandle(file_handle);
        map_handle  = nullptr;
        file_handle = nullptr;
        throw std::runtime_error("llama_kv_cache_mmap: failed to map view of file");
    }

    LLAMA_LOG_INFO("%s: mapped %zu bytes from '%s' (Windows)\n", __func__, size, path.c_str());

#else
    // POSIX implementation using mmap

    // Open or create file
    fd = open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd == -1) {
        throw std::runtime_error("llama_kv_cache_mmap: failed to open file: " + path + " (" + strerror(errno) + ")");
    }

    // Resize file to required size
    if (ftruncate(fd, static_cast<off_t>(size)) == -1) {
        close(fd);
        fd = -1;
        throw std::runtime_error("llama_kv_cache_mmap: failed to resize file (" + std::string(strerror(errno)) + ")");
    }

    // Map file into memory
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        fd  = -1;
        ptr = nullptr;
        throw std::runtime_error("llama_kv_cache_mmap: mmap failed (" + std::string(strerror(errno)) + ")");
    }

    LLAMA_LOG_INFO("%s: mapped %zu bytes from '%s' (POSIX)\n", __func__, size, path.c_str());
#endif

    // Zero-initialize the memory for consistent initial state
    memset(ptr, 0, size);
}

llama_kv_cache_mmap::~llama_kv_cache_mmap() {
#ifdef _WIN32
    if (ptr != nullptr) {
        UnmapViewOfFile(ptr);
        ptr = nullptr;
    }
    if (map_handle != nullptr) {
        CloseHandle(map_handle);
        map_handle = nullptr;
    }
    if (file_handle != nullptr) {
        CloseHandle(file_handle);
        file_handle = nullptr;
    }
#else
    if (ptr != nullptr && ptr != MAP_FAILED) {
        munmap(ptr, mapped_size);
        ptr = nullptr;
    }
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
#endif
    mapped_size = 0;
}

llama_kv_cache_mmap::llama_kv_cache_mmap(llama_kv_cache_mmap && other) noexcept :
    ptr(other.ptr),
    mapped_size(other.mapped_size)
#ifdef _WIN32
    ,
    file_handle(other.file_handle),
    map_handle(other.map_handle)
#else
    ,
    fd(other.fd)
#endif
{
    other.ptr         = nullptr;
    other.mapped_size = 0;
#ifdef _WIN32
    other.file_handle = nullptr;
    other.map_handle  = nullptr;
#else
    other.fd = -1;
#endif
}

llama_kv_cache_mmap & llama_kv_cache_mmap::operator=(llama_kv_cache_mmap && other) noexcept {
    if (this != &other) {
        // Clean up current resources
        this->~llama_kv_cache_mmap();

        // Move from other
        ptr         = other.ptr;
        mapped_size = other.mapped_size;
#ifdef _WIN32
        file_handle       = other.file_handle;
        map_handle        = other.map_handle;
        other.file_handle = nullptr;
        other.map_handle  = nullptr;
#else
        fd       = other.fd;
        other.fd = -1;
#endif
        other.ptr         = nullptr;
        other.mapped_size = 0;
    }
    return *this;
}
