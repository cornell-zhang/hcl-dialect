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
      // If result memref element type is floating point
      // change it to i64 to be compatible with numpy
      if (et.isa<BfloatType>()) {
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
      // If argument memref element type is floating point
      // change it to i64 to be compatible with numpy
      if (et.isa<BfloatType>()) {
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
        if (et.isa<BfloatType>()) {
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
After we changed the function arguments, affine load's argument
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

/* Update Return Op's argument value to be i64 memref
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
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value arg = op->getOperand(i);
      MemRefType type = arg.getType().cast<MemRefType>();
      Type etype = type.getElementType();
      Type newType = type.clone(IntegerType::get(funcOp.getContext(), 64));
      if (etype != newType and etype.isa<BfloatType>()) {
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
void markBfloatOperations(FuncOp &f) {
  SmallVector<Operation *, 10> bfloatOps;
  f.walk([&](Operation *op) {
    if (llvm::isa<AddBfloatOp, SubBfloatOp, MulBfloatOp, CmpBfloatOp, MinBfloatOp,
                  MaxBfloatOp>(op)) {
      bfloatOps.push_back(op);
    }
  });
  // set attribute to addOps
  for (auto op : bfloatOps) {
    // FixedAddOps are binary ops, they have two operands
    Value opr_l = op->getOperand(0);
    Value opr_r = op->getOperand(1);
    Value res = op->getResult(0);
    size_t lexp, lfrac, rexp, rfrac, resexp, resfrac;
    // The operands are floating point
    if (opr_l.getType().cast<BfloatType>()) { // BfloatType
      BfloatType ltype = opr_l.getType().cast<BfloatType>();
      BfloatType rtype = opr_r.getType().cast<BfloatType>();
      BfloatType restype = res.getType().cast<BfloatType>();
      lexp = ltype.getExp();
      lfrac = ltype.getFrac();
      rexp = rtype.getExp();
      rfrac = rtype.getFrac();
      resexp = restype.getExp();
      resfrac = restype.getFrac();
    }
    // add exp, frac info as attributes
    OpBuilder builder(f.getContext());
    IntegerType targetType = builder.getIntegerType(32);
    op->setAttr("lexp", builder.getIntegerAttr(targetType, lexp));
    op->setAttr("lfrac", builder.getIntegerAttr(targetType, lfrac));
    op->setAttr("rexp", builder.getIntegerAttr(targetType, rexp));
    op->setAttr("rfrac", builder.getIntegerAttr(targetType, rfrac));
    op->setAttr("resexp", builder.getIntegerAttr(targetType, resexp));
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
    if (auto ft = t.dyn_cast_or_null<BfloatType>()) {
      width = ft.getExp();
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
  if (!valueTyp.isa<IntegerType>() || !memRefEleTyp.isa<IntegerType>()) {
    return;
  }
  if (valueTyp.cast<IntegerType>().getWidth() <
      memRefEleTyp.cast<IntegerType>().getWidth()) {
    // extend
    Value v = builder.create<arith::ExtSIOp>(op->getLoc(), memRefEleTyp,
                                             op->getOperand(0));
    op->setOperand(0, v);
  } else if (valueTyp.cast<IntegerType>().getWidth() >
             memRefEleTyp.cast<IntegerType>().getWidth()) {
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
void lowerBfloatAdd(AddBfloatOp &op) {
  size_t width =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  arith::AddIOp newOp = rewriter.create<arith::AddIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower FixedSubOp to SubIOp
void lowerBfloatSub(SubBfloatOp &op) {
  size_t width =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  arith::SubIOp newOp = rewriter.create<arith::SubIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower MulFixedop to MulIOp
void lowerBfloatMul(MulBfloatOp &op) {
  size_t lexp =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  size_t lfrac =
      op->getAttr("lfrac").cast<IntegerAttr>().getValue().getSExtValue();
      
  OpBuilder rewriter(op);
  
  Value lhs_exp = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), lexp);
  Value rhs_exp = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), lexp);
  Value lhs_frac = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), lfrac);
  Value rhs_frac = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), lfrac);
  Value lhs = (lhs_frac)*pow(10,lhs_exp);
  Value rhs = (rhs_frac)*pow(10,rhs_exp);

  arith::MulIOp newOp = rewriter.create<arith::MulIOp>(op->getLoc(), lhs, rhs);

  // lhs<width, frac> * rhs<width, frac> -> res<width, 2*frac>
  // Therefore, we need to right shift the result for frac bit
  // Right shift needs to consider signed/unsigned
  Type opTy = op->getOperand(0).getType();
  IntegerType intTy = IntegerType::get(op->getContext(), 32);
  auto fracAttr = rewriter.getIntegerAttr(intTy, lfrac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);

  if (opTy.isa<BfloatType>()) {
    // use signed right shift
    arith::ShRSIOp res =
        rewriter.create<arith::ShRSIOp>(op->getLoc(), newOp, fracCstOp);
    op->replaceAllUsesWith(res);
  }
}

// Lower CmpFixedOp to CmpIOp
void lowerBfloatCmp(CmpBfloatOp &op) {
  size_t width =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);

  auto prednum =
      op->getAttr("predicate").cast<IntegerAttr>().getValue().getSExtValue();
  auto loc = op->getLoc();
  arith::CmpIOp newOp;
  switch (prednum) {
  case 0:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
    break;
  case 1:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
    break;
  case 2:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
    break;
  case 3:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs, rhs);
    break;
  case 4:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
    break;
  case 5:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
    break;
  case 6:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lhs, rhs);
    break;
  case 7:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, lhs, rhs);
    break;
  case 8:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs, rhs);
    break;
  case 9:
    rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, lhs, rhs);
    break;
  default:
    llvm::errs() << "unknown predicate code in CmpFixedOp\n";
  }

  op->replaceAllUsesWith(newOp);
}

// Lower MinFixedOp to MinSIOp or MinUIOp
void lowerBfloatMin(MinBfloatOp &op) {
  size_t width =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(2), width);

  Type opTy = op->getOperand(0).getType();
  if (opTy.isa<BfloatType>()) {
    // use signed integer min
    auto res = rewriter.create<arith::MinSIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned integer min
    auto res = rewriter.create<arith::MinUIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  }
}

// Lower MaxFixedOp to MaxSIOp or MaxUIOp
void lowerBfloatMax(MaxBfloatOp &op) {
  size_t width =
      op->getAttr("lexp").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(2), width);

  Type opTy = op->getOperand(0).getType();
  if (opTy.isa<BfloatType>()) {
    // use signed integer max
    auto res = rewriter.create<arith::MaxSIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned integer max
    auto res = rewriter.create<arith::MaxUIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  }
}

/// Visitors to recursively update all operations
void visitOperation(Operation &op);
void visitRegion(Region &region);
void visitBlock(Block &block);

void visitOperation(Operation &op) {
  if (auto new_op = dyn_cast<AddBfloatOp>(op)) {
    lowerBfloatAdd(new_op);
  } else if (auto new_op = dyn_cast<SubBfloatOp>(op)) {
    lowerBfloatSub(new_op);
  } else if (auto new_op = dyn_cast<MulBfloatOp>(op)) {
    lowerBfloatMul(new_op);
  } else if (auto new_op = dyn_cast<CmpBfloatOp>(op)) {
    lowerBfloatCmp(new_op);
  } else if (auto new_op = dyn_cast<MinBfloatOp>(op)) {
    lowerBfloatMin(new_op);
  } else if (auto new_op = dyn_cast<MaxBfloatOp>(op)) {
    lowerBfloatMax(new_op);
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
    if (llvm::isa<AddBfloatOp, SubBfloatOp, MulBfloatOp, CmpBfloatOp, MinBfloatOp,
                  MaxBfloatOp>(op)) {
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

bool applyFloatingPointToInteger(ModuleOp &mod) {

  for (FuncOp func : mod.getOps<FuncOp>()) {
    // lowerBfloatAdd(func);
    markBfloatOperations(func);
    updateFunctionSignature(func);
    updateAffineLoad(func);
    updateReturnOp(func);
    updateAlloc(func);
    for (Operation &op : func.getOps()) {
      visitOperation(op);
    }
  }

  return true;
}



} // namespace hcl
} // namespace mlir

namespace {

struct HCLFloatingPointToIntegerTransformation
    : public FloatingPointToIntegerBase<HCLFloatingPointToIntegerTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyFloatingPointToInteger(mod))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace hcl {

// Create A Fixed-Point to Integer Pass
std::unique_ptr<OperationPass<ModuleOp>> createFloatingPointToIntegerPass() {
  return std::make_unique<HCLFloatingPointToIntegerTransformation>();
}

} // namespace hcl
} // namespace mlir