//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_MLIR_C_HCL_ATTRIBUTES__H
#define HCL_MLIR_C_HCL_ATTRIBUTES__H

#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IntegerSet.h"

#ifdef __cplusplus
extern "C" {
#endif

// MLIR_CAPI_EXPORTED bool mlirAttributeIsAIntegerSet(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirIntegerSetAttrGet(MlirIntegerSet set);

#ifdef __cplusplus
}
#endif

#endif // HCL_MLIR_C_HCL_ATTRIBUTES__H