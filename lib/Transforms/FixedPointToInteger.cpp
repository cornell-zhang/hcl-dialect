//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {


void lowerFixedAdd(FuncOp &f) {
  // get all fixed-point add ops
  SmallVector<Operation*, 10> FixedAddOps;
  f.walk([&](Operation* op) {
    if (auto add_op = dyn_cast<AddFixedOp>(op)) {
      FixedAddOps.push_back(op);
    }
  });

  for (Operation* op : FixedAddOps) {
    llvm::outs() << op->getName() << "\n";
  }

}

bool applyFixedPointToInteger(ModuleOp &mod) {


  for (FuncOp func : mod.getOps<FuncOp>()) {
    lowerFixedAdd(func);
  }

  return true;
}
} // namespace hcl
} // namespace mlir


namespace {

struct HCLFixedToIntegerTransformation 
    : public FixedToIntegerBase<HCLFixedToIntegerTransformation>{

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyFixedPointToInteger(mod))
      return signalPassFailure();  
  }
};

} // namespace

namespace mlir {
namespace hcl {

// Create A Fixed-Point to Integer Pass
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass() {
    return std::make_unique<HCLFixedToIntegerTransformation>();
}

} // namespace hcl
} // namespace mlir