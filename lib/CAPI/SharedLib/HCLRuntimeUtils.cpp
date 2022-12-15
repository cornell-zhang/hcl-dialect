#include "hcl-c/SharedLib/HCLRuntimeUtils.h"
#include <iostream>

extern "C" void _mlir_ciface_loadMemrefI32(UnrankedMemRefType<int32_t> *m) {
  std::cout << "Hello from _mlir_ciface_loadMemrefI32" << std::endl;
}

// reference: https://github.com/llvm/llvm-project/blob/bd672e2fc03823e536866da6721b9f053cfd586b/mlir/lib/ExecutionEngine/CRunnerUtils.cpp#L59
extern "C" void loadMemrefI32(int64_t rank, void *ptr, char *str) {
    printf("%s", str);
    std::cout << "Hello from loadMemrefI32" << std::endl;
}