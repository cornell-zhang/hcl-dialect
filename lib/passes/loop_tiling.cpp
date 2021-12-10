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

struct HCLLoopTiling : public PassWrapper<HCLLoopTiling, FunctionPass> {

  void runOnFunction() override;
  
  StringRef getArgument() const final { return "hcl-loop-split"; }
  StringRef getDescription() const final {
    return "Loop tiling in HeteroCL";
  }
};

} // namespace

// https://github.com/llvm/llvm-project/blob/release/13.x/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
void HCLLoopTiling::runOnFunction()  {
  FuncOp f = getFunction();
  SmallVector<AffineForOp, 6> forOps;
  for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    forOps.push_back(forOp);
    break;
  }
  // f.walk([&](AffineForOp forOp) { forOps.push_back(forOp); });
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(2);
  SmallVector<AffineForOp, 12> tiledNest;
  if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
    return signalPassFailure();
}


namespace mlir {
namespace hcl{
// Register Loop Tiling Pass
void registerHCLLoopTilingPass() {
  PassRegistration<HCLLoopTiling>();
}

// Create A Loop Tiling Pass
std::unique_ptr<mlir::Pass> createHCLLoopTilingPass() {
  return std::make_unique<HCLLoopTiling>();
}
} // namespace hcl
} // namespace mlir