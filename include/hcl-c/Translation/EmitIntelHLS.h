//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_C_TRANSLATION_EMITINTELHLS_H
#define HCL_C_TRANSLATION_EMITINTELHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitIntelHls(
    MlirModule module, MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // HCL_C_TRANSLATION_EMITINTELHLS_H
