//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/HCLTypes.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

bool hclMlirTypeIsALoopHandle(MlirType type) {
  return unwrap(type).isa<hcl::LoopHandleType>();
}

MlirType hclMlirLoopHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::LoopHandleType::get(unwrap(ctx)));
}

bool hclMlirTypeIsAStageHandle(MlirType type) {
  return unwrap(type).isa<hcl::StageHandleType>();
}

MlirType hclMlirStageHandleTypeGet(MlirContext ctx) {
  return wrap(hcl::StageHandleType::get(unwrap(ctx)));
}

bool hclMlirTypeIsAFixedType(MlirType type) {
  return unwrap(type).isa<hcl::FixedType>();
}

MlirType hclMlirFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(hcl::FixedType::get(unwrap(ctx), width, frac));
}

bool hclMlirTypeIsAUFixedType(MlirType type) {
  return unwrap(type).isa<hcl::UFixedType>();
}

MlirType hclMlirUFixedTypeGet(MlirContext ctx, size_t width, size_t frac) {
  return wrap(hcl::UFixedType::get(unwrap(ctx), width, frac));
}