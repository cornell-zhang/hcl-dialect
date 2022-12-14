add_mlir_library(hcl_runtime_utils
    SHARED
    HCLRuntimeUtils.cpp
)
target_compile_definitions(hcl_runtime_utils PRIVATE hcl_runtime_utils_EXPORTS)