#ifndef HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H
#define HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H

#ifdef _WIN32
#ifndef HCL_RUNTIME_UTILS_EXPORT
#ifdef hcl_runtime_utils_EXPORTS
// We are building this library
#define HCL_RUNTIME_UTILS_EXPORT __declspec(dllexport)
#define HCL_RUNTIME_UTILS_DEFINE_FUNCTIONS
#else
// We are using this library
#define HCL_RUNTIME_UTILS_EXPORT __declspec(dllimport)
#endif // hcl_runtime_utils_EXPORTS
#endif // HCL_RUNTIME_UTILS_EXPORT
#else  // _WIN32
// Non-windows: use visibility attributes.
#define HCL_RUNTIME_UTILS_EXPORT __attribute__((visibility("default")))
#define HCL_RUNTIME_UTILS_DEFINE_FUNCTIONS
#endif // _WIN32

#include <string>

//===----------------------------------------------------------------------===//
// Codegen-compatible structure for UnrankedMemRef type.
//===----------------------------------------------------------------------===//
// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};


extern "C" HCL_RUNTIME_UTILS_EXPORT void
_mlir_ciface_loadMemrefI32(UnrankedMemRefType<int32_t> *m);

extern "C" HCL_RUNTIME_UTILS_EXPORT void
loadMemrefI32(int64_t rank, void *ptr);

#endif // HCLC_SHARED_LIB_HCL_RUNTIME_UTILS_H