#pragma once

// RAII wrapper for mmap'd KV cache memory
// Provides file-backed memory mapping for KV cache to enable:
// - Explicit memory management
// - Potential persistence across runs
// - OS-level paging for large contexts

#include <cstddef>
#include <string>

class llama_kv_cache_mmap {
public:
    // Create or open a file at 'path' and map 'size' bytes
    // If the file exists and is smaller, it will be extended
    // If the file exists and is larger, only 'size' bytes are mapped
    llama_kv_cache_mmap(const std::string & path, size_t size);
    ~llama_kv_cache_mmap();

    // Get pointer to mapped memory
    void * data() const { return ptr; }
    
    // Get size of mapped region
    size_t size() const { return mapped_size; }
    
    // Check if mapping is valid
    bool is_valid() const { return ptr != nullptr; }

    // Disable copy (mmap resources are not copyable)
    llama_kv_cache_mmap(const llama_kv_cache_mmap &) = delete;
    llama_kv_cache_mmap & operator=(const llama_kv_cache_mmap &) = delete;

    // Allow move
    llama_kv_cache_mmap(llama_kv_cache_mmap && other) noexcept;
    llama_kv_cache_mmap & operator=(llama_kv_cache_mmap && other) noexcept;

private:
    void * ptr = nullptr;
    size_t mapped_size = 0;
    
#ifdef _WIN32
    void * file_handle = nullptr;  // HANDLE
    void * map_handle = nullptr;   // HANDLE
#else
    int fd = -1;
#endif
};
