#include "hcl-c/SharedLib/HCLRuntimeUtils.h"
#include <iostream>

extern "C" void loadMemrefI32(int64_t rank, void *ptr) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  std::cout << "Hello from loadMemrefI32" << std::endl;
}