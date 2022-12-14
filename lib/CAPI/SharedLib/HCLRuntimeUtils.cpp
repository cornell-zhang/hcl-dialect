#include "hcl-c/SharedLib/HCLRuntimeUtils.h"
#include <iostream>

extern "C" void _mlir_ciface_loadMemrefI32(UnrankedMemRefType<int32_t> *m) {
  std::cout << "Hello from _mlir_ciface_loadMemrefI32" << std::endl;
}

extern "C" void loadMemrefI32(int64_t rank, void *ptr) {
    std::cout << "Hello from loadMemrefI32" << std::endl;
}