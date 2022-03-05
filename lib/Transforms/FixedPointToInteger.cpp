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

// TODO(Niansong): function calls also need to be handled

/* Update the function signature and
 * Because we need to interact with numpy, which only supports up
 * to 64-bit int/uint, so we update the input/output arguments
 * to 64-bit signless integer type. When the input memref
 */
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
      // If result memref element type is fixed
      // change it to i64 to be compatible with numpy
      if (et.isa<FixedType, UFixedType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
      } else {
        new_result_types.push_back(memrefType);
      }
    }
  }

  for (Type t : arg_types) {
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      // If argument memref element type is fixed
      // change it to i64 to be compatible with numpy
      if (et.isa<FixedType, UFixedType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_arg_types.push_back(memrefType.clone(newElementType));
      } else {
        new_arg_types.push_back(memrefType);
      }
    }
  }

  // Update FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.cast<MemRefType>()) {
        Type et = memrefType.getElementType();
        if (et.isa<FixedType, UFixedType>()) {
          size_t width = 64;
          Type newType = IntegerType::get(funcOp.getContext(), width);
          Type newMemRefType = memrefType.clone(newType);
          block.getArgument(i).setType(newMemRefType);
        }
      }
    }
  }

  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);
}

/* Update AffineLoad's result type
After we changed the function arguments, affine loads's argument
memref may change as well, which makes the affine load's result
type different from input memref's element type. This function
updates the result type of affine load operations
*/
void updateAffineLoad(FuncOp &f) {
  SmallVector<Operation *, 10> loads;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(op);
    }
  });

  for (auto op : loads) {
    for (auto v : llvm::enumerate(op->getResults())) {
      Type newType =
          op->getOperand(0).getType().cast<MemRefType>().getElementType();
      op->getResult(v.index()).setType(newType);
    }
  }
}

/* Update Return Op's arguement value to be i64 memref
 * Check ReturnOp's argument, if it is an AllocOp and
 * it's type is not i64 memref, update it to be i64 memeref
 */
void updateReturnOp(FuncOp &funcOp) {
  // Update FuncOp's return types
  SmallVector<Operation *, 4> returnOps;
  funcOp.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<ReturnOp>(op)) {
      returnOps.push_back(op);
    }
  });
  // If return op is not int64, we need to add a cast node
  for (auto op : returnOps) {
    MemRefType type = op->getOperand(0).getType().cast<MemRefType>();
    Type etype = type.getElementType();
    Type newType = type.clone(IntegerType::get(funcOp.getContext(), 64));
    if (etype != newType) {
      for (unsigned i = 0; i < op->getNumOperands(); i++) {
        Value arg = op->getOperand(i);
        if (auto allocOp = dyn_cast<memref::AllocOp>(arg.getDefiningOp())) {
          allocOp->getResult(0).setType(newType);
        }
      }
    }
  }
}

/* Add attributes to fixed-point operations
 * to preserve operands and result's fixed-type
 * information. After block arguments and
 * affine load operations are updated to integer
 * type, these information will not be directly
 * accessible through operands' types.
 */
void markFixedOperations(FuncOp &f) {
  SmallVector<Operation *, 10> fixedOps;
  f.walk([&](Operation *op) {
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, CmpFixedOp, MinFixedOp,
                  MaxFixedOp>(op)) {
      fixedOps.push_back(op);
    }
  });
  // set attribute to addOps
  for (auto op : fixedOps) {
    // FixedAddOps are binary ops, they have two operands
    Value opr_l = op->getOperand(0);
    Value opr_r = op->getOperand(1);
    Value res = op->getResult(0);
    size_t lwidth, lfrac, rwidth, rfrac, reswidth, resfrac;
    // The operands are either fixed-point or unsigned fixed-point
    if (opr_l.getType().cast<FixedType>()) { // fixed
      FixedType ltype = opr_l.getType().cast<FixedType>();
      FixedType rtype = opr_r.getType().cast<FixedType>();
      FixedType restype = res.getType().cast<FixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
      reswidth = restype.getWidth();
      resfrac = restype.getFrac();
    } else if (opr_l.getType().cast<UFixedType>()) { // ufixed
      UFixedType ltype = opr_l.getType().cast<UFixedType>();
      UFixedType rtype = opr_r.getType().cast<UFixedType>();
      UFixedType restype = res.getType().cast<UFixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
      reswidth = restype.getWidth();
      resfrac = restype.getFrac();
    }
    // add width, frac info as attributes
    OpBuilder builder(f.getContext());
    IntegerType targetType = builder.getIntegerType(32);
    op->setAttr("lwidth", builder.getIntegerAttr(targetType, lwidth));
    op->setAttr("lfrac", builder.getIntegerAttr(targetType, lfrac));
    op->setAttr("rwidth", builder.getIntegerAttr(targetType, rwidth));
    op->setAttr("rfrac", builder.getIntegerAttr(targetType, rfrac));
    op->setAttr("reswidth", builder.getIntegerAttr(targetType, reswidth));
    op->setAttr("resfrac", builder.getIntegerAttr(targetType, resfrac));
  }
}

// Fixed-point memref allocation op to integer memref
void updateAlloc(FuncOp &f) {
  SmallVector<Operation *, 10> allocOps;
  f.walk([&](Operation *op) {
    if (auto alloc_op = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(op);
    }
  });

  for (auto op : allocOps) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType().cast<MemRefType>();
    Type t = memRefType.getElementType();
    size_t width;
    if (FixedType ft = t.cast<FixedType>()) {
      width = ft.getWidth();
    } else if (UFixedType uft = t.cast<UFixedType>()) {
      width = uft.getWidth();
    } else {
      // Not a fixed-point alloc operation
      // Return without changing anything
      return;
    }
    Type newType = IntegerType::get(f.getContext(), width);
    Type newMemRefType = memRefType.clone(newType);
    op->getResult(0).setType(newMemRefType); // alloc has only one result
  }
}

void updateAffineStore(AffineStoreOp &op) {
  Type valueTyp = op->getOperand(0).getType();
  Type memRefEleTyp =
      op->getOperand(1).getType().cast<MemRefType>().getElementType();
  OpBuilder builder(op);
  if (valueTyp.cast<IntegerType>().getWidth() <
      memRefEleTyp.cast<IntegerType>().getWidth()) {
    // extend
    Value v = builder.create<arith::ExtSIOp>(op->getLoc(), memRefEleTyp,
                                             op->getOperand(0));
    op->setOperand(0, v);
  } else {
    // truncate
    Value v = builder.create<arith::TruncIOp>(op->getLoc(), memRefEleTyp,
                                              op->getOperand(0));
    op->setOperand(0, v);
  }
}

/* Cast integer to target bitwidth
 */
Value castIntegerWidth(MLIRContext *ctx, OpBuilder &builder, Location loc,
                       Value v, size_t target_width) {
  Value result;
  Type newType = IntegerType::get(ctx, target_width);
  if (v.getType().cast<IntegerType>().getWidth() < target_width) {
    // extend bits
    result = builder.create<arith::ExtSIOp>(loc, newType, v);
  } else {
    // truncate bits
    result = builder.create<arith::TruncIOp>(loc, newType, v);
  }
  return result;
}

// Lower AddFixedOp to AddIOp
void lowerFixedAdd(AddFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  arith::AddIOp newOp = rewriter.create<arith::AddIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower FixedSubOp to SubIOp
void lowerFixedSub(SubFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  arith::SubIOp newOp = rewriter.create<arith::SubIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower MulFixedop to MulIOp
void lowerFixedMul(MulFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  size_t frac =
      op->getAttr("lfrac").cast<IntegerAttr>().getValue().getSExtValue();

  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  arith::MulIOp newOp = rewriter.create<arith::MulIOp>(op->getLoc(), lhs, rhs);

  // lhs<width, frac> * rhs<width, frac> -> res<width, 2*frac>
  // Therefore, we need to right shift the result for frac bit
  // Right shift needs to consider signed/unsigned
  Type opTy = op->getOperand(0).getType();
  IntegerType intTy = IntegerType::get(op->getContext(), 32);
  auto fracAttr = rewriter.getIntegerAttr(intTy, frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);
  
  if (opTy.isa<FixedType>()) {
    // use signed right shift
    arith::ShRSIOp res =
        rewriter.create<arith::ShRSIOp>(op->getLoc(), newOp, fracCstOp);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned right shift
    arith::ShRUIOp res =
        rewriter.create<arith::ShRUIOp>(op->getLoc(), newOp, fracCstOp);
    op->replaceAllUsesWith(res);
  }
}

// Lower CmpFixedOp to CmpIOp
void lowerFixedCmp(CmpFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(2), width);

  arith::CmpIOp newOp = rewriter.create<arith::CmpIOp>(op->getLoc(), op->getOperand(0), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

void lowerFixedMin(MinFixedOp &op) {

}

void lowerFixedMax(MaxFixedOp &op) {

}

/// Visitors to recursively update all operations
void visitOperation(Operation &op);
void visitRegion(Region &region);
void visitBlock(Block &block);

void visitOperation(Operation &op) {
  if (auto new_op = dyn_cast<AddFixedOp>(op)) {
    lowerFixedAdd(new_op);
  } else if (auto new_op = dyn_cast<SubFixedOp>(op)) {
    lowerFixedSub(new_op);
  } else if (auto new_op = dyn_cast<MulFixedOp>(op)) {
    lowerFixedMul(new_op);
  } else if (auto new_op = dyn_cast<CmpFixedOp>(op)) {
    lowerFixedCmp(new_op);
  } else if (auto new_op = dyn_cast<MinFixedOp>(op)) {
    lowerFixedMin(new_op);
  } else if (auto new_op = dyn_cast<MaxFixedOp>(op)) {
    lowerFixedMax(new_op);
  } else if (auto new_op = dyn_cast<AffineStoreOp>(op)) {
    updateAffineStore(new_op);
  }

  for (auto &region : op.getRegions()) {
    visitRegion(region);
  }
}

void visitBlock(Block &block) {
  SmallVector<Operation *, 10> opToRemove;
  for (auto &op : block.getOperations()) {
    visitOperation(op);
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, CmpFixedOp, MinFixedOp, MaxFixedOp>(op)) {
      opToRemove.push_back(&op);
    }
  }

  // Remove fixed-point operations after the block
  // is visited. 
  std::reverse(opToRemove.begin(), opToRemove.end());
  for (Operation *op : opToRemove) {
    op->erase();
  }
}

void visitRegion(Region &region) {
  for (auto &block : region.getBlocks()) {
    visitBlock(block);
  }
}

bool applyFixedPointToInteger(ModuleOp &mod) {

  for (FuncOp func : mod.getOps<FuncOp>()) {
    // lowerFixedAdd(func);
    markFixedOperations(func);
    updateFunctionSignature(func);
    updateAffineLoad(func);
    updateAlloc(func);
    updateReturnOp(func);
    for (Operation &op : func.getOps()) {
      visitOperation(op);
    }
    llvm::outs() << func << "\n";
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