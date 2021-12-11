#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
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

template <typename RangeT>
static auto findArg(RangeT &&range, StringRef name) {
  auto it = llvm::find_if(range, [=](auto &arg) { return arg.first.str() == name; });
  return it != range.end() ? &*it : nullptr;
}

// https://github.com/llvm/llvm-project/blob/release/13.x/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
void HCLLoopTiling::runOnFunction()  {
  FuncOp f = getFunction();
  // SmallVector<hcl::SplitOp, 6> splitOps;
  for (hcl::SplitOp splitOp : f.getOps<hcl::SplitOp>()) {
    unsigned int factor = splitOp.factor();
    const auto loop_name = splitOp.loop().getType().cast<hcl::LoopType>().getLoopName();
    SmallVector<AffineForOp, 6> forOps;
    // for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    f.walk([&](AffineForOp forOp) { // loop nest traversal
      const NamedAttribute* attr = findArg(forOp->getAttrs(), "loop_name");
      if (loop_name == attr->second.cast<StringAttr>().getValue()) {
        forOps.push_back(forOp);
        SmallVector<unsigned, 6> tileSizes;
        tileSizes.push_back(factor);
        SmallVector<AffineForOp, 6> tiledNest;
        if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
          return signalPassFailure();
      }
    });
  }
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