//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/HCL.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

bool mlirTypeIsALoopHandle(MlirType type) {
  return unwrap(type).isa<hcl::LoopHandleType>();
}

MlirType mlirLoopHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::LoopHandleType::get(unwrap(ctx)));
}

bool mlirTypeIsAStageHandle(MlirType type) {
  return unwrap(type).isa<hcl::StageHandleType>();
}

MlirType mlirStageHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::StageHandleType::get(unwrap(ctx)));
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HCL, hcl, hcl::HeteroCLDialect)
