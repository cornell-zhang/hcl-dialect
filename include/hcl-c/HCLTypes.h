//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_MLIR_C_HCLTYPES_H
#define HCL_MLIR_C_HCLTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool hclTypeIsALoopHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclLoopHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool hclTypeIsAStageHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclStageHandleTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_HCLTYPES_H