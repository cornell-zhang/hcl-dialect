//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_C_DIALECT_HLSCPP_H
#define HCL_C_DIALECT_HLSCPP_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HCL, hcl);

#ifdef __cplusplus
}
#endif

#endif // HCL_C_DIALECT_HLSCPP_H
