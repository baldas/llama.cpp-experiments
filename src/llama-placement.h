#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Apply layer placement policy to a loaded model.
//
// After standard model loading (including mmap and CPU_REPACK), this function
// copies Optane-designated tensor data from their current location to the
// pmem mmap region, then redirects tensor->data pointers.  Because the
// copy source is the post-load data (not the raw GGUF file), the repacked
// layout is preserved correctly.
//
// model:      The loaded llama_model instance.
// pmem_path:  Path to the DAX-mounted file where Optane tensors will be stored/mapped.
// config_str: Configuration string specifying placement (e.g., "embed:dram,0-7:dram,8-31:optane,output:dram").
//
// Returns 0 on success, non-zero on failure.
GGML_API int llama_apply_layer_placement(struct llama_model * model, const char * pmem_path, const char * config_str);

#ifdef __cplusplus
}
#endif
