//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/Dialects.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HCL, hcl, mlir::hcl::HeteroCLDialect)