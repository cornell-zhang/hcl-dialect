//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"


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
    // FixedAddOps are binary ops, they have two operands
    Value opr_l = op->getOperand(0);
    Value opr_r = op->getOperand(1);
    size_t lwidth, lfrac, rwidth, rfrac;
    // The operands are either fixed-point or unsigned fixed-point
    if (opr_l.getType().cast<FixedType>()) { // fixed
      FixedType ltype = opr_l.getType().cast<FixedType>();
      FixedType rtype = opr_r.getType().cast<FixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
    } else { // ufixed
      UFixedType ltype = opr_l.getType().cast<UFixedType>();
      UFixedType rtype = opr_r.getType().cast<UFixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
    }

    OpBuilder rewriter(op);
    auto loc = op->getLoc();
    rewriter.create<arith::AddIOp>(loc, opr_l, opr_r);

    // Check width and cast

    // Check frac and shift

    
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