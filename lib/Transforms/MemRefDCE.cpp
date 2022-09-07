//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MemRefDCE Pass
// This pass removes memrefs that are not loaded from.
// We only look at memrefs allocated in functions.
// Global memrefs and memrefs in function args are not removed.
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {
/// Pass entry point
bool applyMemRefDCE(ModuleOp &mod) { return true; }
} // namespace hcl
} // namespace mlir

namespace {
struct HCLMemRefDCETransformation
    : public MemRefDCEBase<HCLMemRefDCETransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMemRefDCE(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createMemRefDCEPass() {
  return std::make_unique<HCLMemRefDCETransformation>();
}
} // namespace hcl
} // namespace mlir