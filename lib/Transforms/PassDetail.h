//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_MLIR_PASSDETAIL_H
#define HCL_MLIR_PASSDETAIL_H

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hcl {

#define GEN_PASS_CLASSES
#include "hcl/Transforms/Passes.h.inc"

} // namespace hcl
} // end namespace mlir

#endif // HCL_MLIR_PASSDETAIL_H
