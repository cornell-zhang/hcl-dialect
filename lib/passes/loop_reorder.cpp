#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#include <iostream>

using namespace mlir;

namespace {

struct HCLLoopReorder : public PassWrapper<HCLLoopReorder, OperationPass<>> {
  void runOnOperation() override {
    Operation *op = getOperation();
    std::cout << "in HCLLoopReorder." << std::endl;
  }
  StringRef getArgument() const final { return "hcl-loop-reorder"; }
  StringRef getDescription() const {
      return "Loop Reorder in HeteroCL";
  }
};

} // namespace

namespace mlir {
namespace hcl{
// Register Loop Reorder Pass
void registerHCLLoopReorderPass() {
    PassRegistration<HCLLoopReorder>(createHCLLoopReorderPass());
}

// Create A Loop Reorder Pass
std::unique_ptr<mlir::Pass> createHCLLoopReorderPass() {
    return std::make_unique<HCLLoopReorder>();
}
} // namespace hcl
} // namespace mlir