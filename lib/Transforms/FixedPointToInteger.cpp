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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

// function calls also need to be handled

// We also need to replace block args
void updateFunctionSignature(FuncOp &funcOp) {
  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(funcOp.getArguments()))
    arg_types.push_back(argEn.value().getType());

  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;

  for (Type t : result_types) {
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      size_t width;
      if (FixedType ft = et.cast<FixedType>()) {
        width = ft.getWidth();
      } else {
        UFixedType uft = et.cast<UFixedType>();
        width = uft.getWidth();
      }

      Type newElementType = IntegerType::get(funcOp.getContext(), width);
      new_result_types.push_back(memrefType.clone(newElementType));
    //   llvm::outs() << memrefType.clone(newElementType) << "\n";
    }
  }

  for (Type t : arg_types) {
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      size_t width;
      if (FixedType ft = et.cast<FixedType>()) {
        width = ft.getWidth();
      } else {
        UFixedType uft = et.cast<UFixedType>();
        width = uft.getWidth();
      }

      Type newElementType = IntegerType::get(funcOp.getContext(), width);
      new_arg_types.push_back(memrefType.clone(newElementType));
    //   llvm::outs() << memrefType.clone(newElementType) << "\n";
    }
  }

  // Update FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.cast<MemRefType>()) {
        Type oldType = memrefType.getElementType();
        size_t width;
        if (FixedType ft = oldType.cast<FixedType>()) {
          width = ft.getWidth();
        } else {
          UFixedType uft = oldType.cast<UFixedType>();
          width = ft.getWidth();
        }
        Type newType = IntegerType::get(funcOp.getContext(), width);
        Type newMemRefType = memrefType.clone(newType);
        block.getArgument(i).setType(newMemRefType);
      }
    }
  }

  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);

  llvm::outs() << "function signature updated\n";
}

//
// void lowerAffineLoad(AffineLoadOp &op) {
//   llvm::outs() << "in lowerAffineLoad\n";
//   // llvm::outs() << op.getName() << "\n";
//   llvm::outs() << op.getResult().getType();
//   for (auto value : op.getOperands()) {
//     llvm::outs() << value;
//   }
// }
void lowerAffineLoad(FuncOp &f) {
  SmallVector<Operation *, 10> loads;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(op);
    }
  });

  for (auto op : loads) {
    // check operands
    for (auto value : op->getOperands()) {
      llvm::outs() << value << "\n";
    }
    // update results
    for (auto v : llvm::enumerate(op->getResults())) {
      Type t = v.value().getType();
      size_t width;
      if (FixedType ft = t.cast<FixedType>()) {
        width = ft.getWidth();
      } else {
        UFixedType uft = t.cast<UFixedType>();
        width = uft.getWidth();
      }
      Type newType = IntegerType::get(f.getContext(), width);
      op->getResult(v.index()).setType(newType);
    }
  }
}

// Lower a Fixed-point add op to AddIOp
void lowerAdd(Operation &addOp) {}

void lowerFixedAdd(FuncOp &f) {
  // get all fixed-point add ops
  SmallVector<Operation *, 10> FixedAddOps;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AddFixedOp>(op)) {
      FixedAddOps.push_back(op);
    }
  });

  for (Operation *op : FixedAddOps) {
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
    // lowerFixedAdd(func);
    updateFunctionSignature(func);
    lowerAffineLoad(func);
    // for (Operation &op : func.getOps()) {
    //   llvm::outs() << "opName: " << op.getName() << "\n";
    //   if (auto new_op = dyn_cast<AffineLoadOp>(op)) {
    //    lowerAffineLoad(new_op);
    //   }
    // }
  }

  return true;
}
} // namespace hcl
} // namespace mlir

namespace {

struct HCLFixedToIntegerTransformation
    : public FixedToIntegerBase<HCLFixedToIntegerTransformation> {

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