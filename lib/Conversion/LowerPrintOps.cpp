//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// LowerPrintOps.cpp defines a pass to lower PrintOp and PrintMemRefOp to
// MLIR's utility printing functions or C printf functions. It also handles
// Fixed-point values/memref casting to float.
// We define our own memref printing and value printing operations to support 
// following cases:
// - Multiple values printed with format string.
// - Print memref. Note that memref printing doesn't support formating.
//===----------------------------------------------------------------------===//

#include "hcl/Conversion/Passes.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {
/// Pass entry point
bool applyLowerPrintOps(ModuleOp &module) { return true; }
} // namespace hcl
} // namespace mlir

namespace {
struct HCLLowerPrintOpsTransformation
    : public LowerPrintOpsBase<HCLLowerPrintOpsTransformation> {
  void runOnOperation() override {
    auto module = getOperation();
    // auto context = &getContext();
    if (applyLowerPrintOps(module)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintOpsPass() {
  return std::make_unique<HCLLowerPrintOpsTransformation>();
}
} // namespace hcl
} // namespace mlir