//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

bool applyFixedPointToInteger(ModuleOp &module) {
  return true;
}
} // namespace hcl
} // namespace mlir