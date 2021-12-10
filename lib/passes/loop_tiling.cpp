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
  // tile(ArrayRef<AffineForOp> forOps,
  //               ArrayRef<uint64_t> sizes,
  //               ArrayRef<AffineForOp> targets);
  FuncOp f = getFunction();
  // for (AffineForOp forOp : f.getOps<AffineForOp>()) { }
  ArrayRef<AffineForOp> forOps;
  f.walk([&](AffineForOp forOp) { forOps.push_back(forOp); });
  ArrayRef<uint64_t> size;
  size.push_back(2);
  size.push_back(2);
  size.push_back(2);
  tile(forOps, size, forOps);
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