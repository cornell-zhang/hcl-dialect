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

MLIR_CAPI_EXPORTED bool hclMlirTypeIsALoopHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirLoopHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAStageHandle(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirStageHandleTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac);
MLIR_CAPI_EXPORTED unsigned hclMlirFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned hclMlirFixedTypeGetFrac(MlirType type);

MLIR_CAPI_EXPORTED bool hclMlirTypeIsAUFixedType(MlirType type);
MLIR_CAPI_EXPORTED MlirType hclMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac);
MLIR_CAPI_EXPORTED unsigned hclMlirUFixedTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED unsigned hclMlirUFixedTypeGetFrac(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_HCLTYPES_H