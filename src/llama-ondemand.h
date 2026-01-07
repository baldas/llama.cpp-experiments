#pragma once

// On-demand model parameter loading using Linux userfaultfd
// This enables loading model parameters from files on-the-fly during inference,
// allowing models larger than available RAM to be used (with performance tradeoffs)

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

// Forward declarations
struct llama_file;

#ifdef __linux__
#    define LLAMA_ONDEMAND_SUPPORTED 1
#else
#    define LLAMA_ONDEMAND_SUPPORTED 0
#endif

// Configuration for on-demand loading
struct llama_ondemand_params {
    size_t max_resident_bytes;   // Max memory to keep resident (0 = unlimited)
    size_t page_size;            // Page size for loading (0 = use system default)
    bool   enable_lru_eviction;  // Enable automatic eviction when memory is constrained
    bool   prefetch_embeddings;  // Prefetch embedding layer on init
    size_t prefetch_layers;      // Number of layers to prefetch ahead (0 = none)
};

// Default parameters for on-demand loading
llama_ondemand_params llama_ondemand_default_params();

// On-demand loading manager for a single file
class llama_ondemand_loader {
  public:
    llama_ondemand_loader(const llama_ondemand_loader &)             = delete;
    llama_ondemand_loader & operator=(const llama_ondemand_loader &) = delete;

    llama_ondemand_loader(llama_file * file, size_t total_size, const llama_ondemand_params & params);
    ~llama_ondemand_loader();

    // Get the base address of the virtual memory region
    // Access to this region will trigger on-demand loading via page faults
    void * get_base_addr() const;

    // Get the total size of the managed region
    size_t get_size() const;

    // Start the fault handling thread (must be called before any access)
    bool start();

    // Stop the fault handling thread
    void stop();

    // Check if the loader is currently running
    bool is_running() const;

    // Manually prefetch a range of bytes (async)
    void prefetch_range(size_t offset, size_t len);

    // Manually evict a range (mark for reload on next access)
    void evict_range(size_t offset, size_t len);

    // Get statistics
    size_t get_resident_bytes() const;
    size_t get_fault_count() const;
    size_t get_eviction_count() const;

    // Check if on-demand loading is supported on this platform
    static bool is_supported();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Collection of on-demand loaders (one per file in a split model)
using llama_ondemand_loaders = std::vector<std::unique_ptr<llama_ondemand_loader>>;
