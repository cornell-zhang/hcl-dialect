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
  // TODO: Support multiple splits/tiles
  FuncOp f = getFunction();
  for (hcl::SplitOp splitOp : f.getOps<hcl::SplitOp>()) {
    unsigned int factor = splitOp.factor();
    const auto loop_name = splitOp.loop().getType().cast<hcl::LoopType>().getLoopName();
    f.walk([&](AffineForOp forOp) { // loop nest traversal
      const NamedAttribute* attr = findArg(forOp->getAttrs(), "loop_name");
      if (loop_name == attr->second.cast<StringAttr>().getValue()) {
        SmallVector<AffineForOp, 6> forOps;
        forOps.push_back(forOp);
        SmallVector<unsigned, 6> tileSizes;
        tileSizes.push_back(factor);
        SmallVector<AffineForOp, 6> tiledNest;
        if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
          return signalPassFailure();
      }
    });
  }
  for (hcl::TileOp tileOp : f.getOps<hcl::TileOp>()) {
    unsigned int x_factor = tileOp.x_factor();
    unsigned int y_factor = tileOp.y_factor();
    const StringRef x_loop = tileOp.x_loop().getType().cast<hcl::LoopType>().getLoopName();
    const StringRef y_loop = tileOp.y_loop().getType().cast<hcl::LoopType>().getLoopName();
    SmallVector<AffineForOp, 6> forOps;
    SmallVector<unsigned, 6> tileSizes;
    tileSizes.push_back(x_factor);
    tileSizes.push_back(y_factor);
    bool isFound = false;
    f.walk([&](AffineForOp forOp) { // loop nest traversal
      if (isFound)
        return;
      const NamedAttribute* attr = findArg(forOp->getAttrs(), "loop_name");
      const StringRef curr_loop_name = attr->second.cast<StringAttr>().getValue();
      if (x_loop == curr_loop_name) {
        // LoopUtils.cpp getPerfectlyNestedLoopsImpl
        forOps.clear();
        AffineForOp rootForOp = forOp;
        for (unsigned i = 0; i < 2; ++i) {
          if (i == 2) {
            const NamedAttribute* attr_y = findArg(rootForOp->getAttrs(), "loop_name");
            const StringRef curr_y_loop = attr_y->second.cast<StringAttr>().getValue();
            if (curr_y_loop != y_loop)
              break;
          }
          forOps.push_back(rootForOp);
          Block &body = rootForOp.region().front();
          if (body.begin() != std::prev(body.end(), 2))
            break;

          rootForOp = dyn_cast<AffineForOp>(&body.front());
          if (!rootForOp)
            break;
        }
        if (forOps.size() < 2) { // not this forOp
          return;
        }
        isFound = true;
      }
    });
    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
      return signalPassFailure();
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