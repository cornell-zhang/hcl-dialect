//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Dialect/HeteroCLTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

bool hclTypeIsALoopHandle(MlirType type) {
  return unwrap(type).isa<hcl::LoopHandleType>();
}

MlirType hclLoopHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::LoopHandleType::get(unwrap(ctx)));
}

bool hclTypeIsAStageHandle(MlirType type) {
  return unwrap(type).isa<hcl::StageHandleType>();
}

MlirType hclStageHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::StageHandleType::get(unwrap(ctx)));
}