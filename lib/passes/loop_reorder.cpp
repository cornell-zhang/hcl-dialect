#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include <iostream>

using namespace mlir;

namespace {

struct HCLLoopReorder : public PassWrapper<HCLLoopReorder, FunctionPass> {
  void runOnFunction() override {
    //Operation *op = getOperation();
    FuncOp f = getFunction();
    std::cout << "in an affine for loop" << std::endl;
    // Get loop bands
    std::vector<AffineForOp> bands;
    for (AffineForOp forOp : f.getOps<AffineForOp>()) {
      bands.push_back(forOp);
      // Print operation attributes
      if (!forOp->getAttrs().empty()) {
        std::cout << forOp->getAttrs().size() << " attributes:\n";
        for (NamedAttribute attr : forOp->getAttrs())
          printIndent() << " - '" << attr.first << "' : '" << attr.second
                        << "'\n";
      }
    }
  }
  StringRef getArgument() const final { return "hcl-loop-reorder"; }
  StringRef getDescription() const {
      return "Loop Reorder in HeteroCL";
  }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < 2; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
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