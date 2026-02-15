#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Apply layer placement policy to a loaded model.
//
// model:      The loaded llama_model instance.
// gguf_path:  Path to the original GGUF file (needed to read tensor data for Optane placement).
// pmem_path:  Path to the DAX-mounted file where Optane tensors will be stored/mapped.
// config_str: Configuration string specifying placement (e.g., "embed:dram,0-7:dram,8-31:optane,output:dram").
//
// Returns 0 on success, non-zero on failure.
GGML_API int llama_apply_layer_placement(struct llama_model * model,
                                         const char *         gguf_path,
                                         const char *         pmem_path,
                                         const char *         config_str);

#ifdef __cplusplus
}
#endif
