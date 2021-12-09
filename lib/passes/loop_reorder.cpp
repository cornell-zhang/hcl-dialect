#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct HCLLoopReorder : public PassWrapper<HCLLoopReorder, FunctionPass> {
  void runOnFunction() override; 
  void labelLoops();
  void reorderLoops();
  
  StringRef getArgument() const final { return "hcl-loop-reorder"; }
  StringRef getDescription() const final {
    return "Loop Reorder in HeteroCL";
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }

 private:
  /// Permutation specifying loop i is mapped to permList[i] in
  /// transformed nest (with i going from outermost to innermost).
  SmallVector<unsigned> permMap;
};

} // namespace

void HCLLoopReorder::runOnFunction()  {
  labelLoops();
  reorderLoops();
}

void HCLLoopReorder::labelLoops() {
  FuncOp f = getFunction();
  // Get loop bands
  std::vector<AffineForOp> bands;
  for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    bands.push_back(forOp);
    // Print operation attributes
    if (!forOp->getAttrs().empty()) {
      printIndent() << forOp->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : forOp->getAttrs())
        printIndent() << " - '" << attr.first << "' : '" << attr.second
                      << "'\n";
    }
  }
}

void HCLLoopReorder::reorderLoops() {
  permMap.push_back(0);
  permMap.push_back(2);
  permMap.push_back(1);
  SmallVector<AffineForOp, 2> forOps;
  getFunction().walk([&](AffineForOp forOp) { forOps.push_back(forOp); });

  for (auto forOp : forOps) {
    SmallVector<AffineForOp, 6> nest;
    // Get the maximal perfect nest.
    getPerfectlyNestedLoops(nest, forOp);
    // Permute if the nest's size is consistent with the specified
    // permutation.
    if (nest.size() >= 2 && nest.size() == permMap.size()) {
      permuteLoops(nest, permMap);
    }
  }
}



namespace mlir {
namespace hcl{
// Register Loop Reorder Pass
void registerHCLLoopReorderPass() {
    PassRegistration<HCLLoopReorder>();
}

// Create A Loop Reorder Pass
std::unique_ptr<mlir::Pass> createHCLLoopReorderPass() {
    return std::make_unique<HCLLoopReorder>();
}
} // namespace hcl
} // namespace mlir