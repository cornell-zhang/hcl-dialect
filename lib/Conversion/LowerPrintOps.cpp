//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
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