//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Support/Utils.h"
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
FunctionType updateFunctionSignature(FuncOp &funcOp) {
  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types;
  for (const auto &argEn : llvm::enumerate(funcOp.getArguments()))
    arg_types.push_back(argEn.value().getType());

  SmallVector<Type, 4> new_result_types;
  SmallVector<Type, 8> new_arg_types;

  // Set the extra type hint based on the input/output memref type
  std::string itypes = "";
  if (funcOp->hasAttr("itypes")) {
    itypes = funcOp->getAttr("itypes").cast<StringAttr>().getValue().str();
  }
  std::string otypes = "";
  if (funcOp->hasAttr("otypes")) {
    otypes = funcOp->getAttr("otypes").cast<StringAttr>().getValue().str();
  }

  for (auto v : llvm::enumerate(result_types)) {
    Type t = v.value();
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      // If result memref element type is fixed
      // change it to i64 to be compatible with numpy
      if (et.isa<FixedType, UFixedType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
        // update the otypes
        if (et.isa<FixedType>() and v.index() < otypes.length()) {
          otypes[v.index()] = 's';
        } else if (et.isa<UFixedType>() and v.index() < otypes.length()) {
          otypes[v.index()] = 'u';
        }
      } else {
        new_result_types.push_back(memrefType);
      }
    } else { // If result type is not memref, do not change it
      new_result_types.push_back(t);
    }
  }

  for (auto v : llvm::enumerate(arg_types)) {
    Type t = v.value();
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      // If argument memref element type is fixed
      // change it to i64 to be compatible with numpy
      if (et.isa<FixedType, UFixedType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_arg_types.push_back(memrefType.clone(newElementType));
        // update the itypes
        if (et.isa<FixedType>() and v.index() < itypes.length()) {
          itypes[v.index()] = 's';
        } else if (et.isa<UFixedType>() and v.index() < itypes.length()) {
          itypes[v.index()] = 'u';
        }
      } else {
        new_arg_types.push_back(memrefType);
      }
    } else { // If argument type is not memref, do not change it
      new_arg_types.push_back(t);
    }
  }

  funcOp->setAttr("itypes", StringAttr::get(funcOp.getContext(), itypes));
  funcOp->setAttr("otypes", StringAttr::get(funcOp.getContext(), otypes));

  // Update FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.dyn_cast<MemRefType>()) {
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
  // funcOp.setType(newFuncType);
  return newFuncType;
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
  // get the return type of the function
  FunctionType funcType = funcOp.getType();
  SmallVector<Type, 4> result_types = llvm::to_vector<4>(funcType.getResults());

  // If return op is not int64, we need to add a cast node
  for (auto op : returnOps) {
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value arg = op->getOperand(i);
      if (MemRefType type = arg.getType().dyn_cast<MemRefType>()) {
        Type etype = type.getElementType();
        Type newType = type.clone(IntegerType::get(funcOp.getContext(), 64));
        if (result_types[i]
                .cast<MemRefType>()
                .getElementType()
                .isa<FixedType, UFixedType>() &&
            etype != newType) {
          // if (etype != newType and etype.isa<FixedType, UFixedType>()) {
          OpBuilder builder(op);
          Location loc = op->getLoc();
          // Get signedness hint information
          std::string otypes = "";
          if (funcOp->hasAttr("otypes")) {
            otypes =
                funcOp->getAttr("otypes").cast<StringAttr>().getValue().str();
          }
          bool is_unsigned = false;
          if (i < otypes.length()) {
            is_unsigned = otypes[i] == 'u';
          }
          Value castedMemRef =
              castIntMemRef(builder, loc, arg, 64, is_unsigned, false);
          op->setOperand(i, castedMemRef);
        }
      }
    }
  }
}

/* Update hcl.print (PrintOp) operations.
 * Create a float64 memref to store the real value
 * of hcl.print's operand memref
 */
void lowerPrintOp(FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto new_op = dyn_cast<PrintOp>(op)) {
      // Only lower fixed-point prints
      MemRefType memRefType =
          new_op->getOperand(0).getType().cast<MemRefType>();
      Type elemType = memRefType.getElementType();
      if (elemType.isa<FixedType, UFixedType>())
        printOps.push_back(op);
    }
  });
  for (auto *printOp : printOps) {
    OpBuilder builder(printOp);
    Type F64 = builder.getF64Type();
    Location loc = printOp->getLoc();
    Value oldMemRef = printOp->getOperand(0);
    MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
    Type oldType = oldMemRefType.getElementType();
    MemRefType newMemRefType = oldMemRefType.clone(F64).cast<MemRefType>();
    Value newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
    SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
    SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
    buildAffineLoopNest(
        builder, loc, lbs, oldMemRefType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
          Value casted;
          size_t frac;
          if (oldType.isa<FixedType>()) {
            casted = nestedBuilder.create<arith::SIToFPOp>(loc, F64, v);
            frac = oldType.cast<FixedType>().getFrac();
          } else {
            casted = nestedBuilder.create<arith::UIToFPOp>(loc, F64, v);
            frac = oldType.cast<UFixedType>().getFrac();
          }
          Value const_frac = nestedBuilder.create<mlir::arith::ConstantOp>(
              loc, F64, nestedBuilder.getFloatAttr(F64, std::pow(2, frac)));
          Value realV = nestedBuilder.create<mlir::arith::DivFOp>(
              loc, F64, casted, const_frac);
          nestedBuilder.create<AffineStoreOp>(loc, realV, newMemRef, ivs);
        });
    printOp->setOperand(0, newMemRef);
  }
}

/* Add attributes to fixed-point operations
 * to preserve operands and result's fixed-type
 * information. After block arguments and
 * affine load operations are updated to integer
 * type, these information will not be directly
 * accessible through operands' types.
 */
void markFixedArithOps(FuncOp &f) {
  SmallVector<Operation *, 10> fixedOps;
  f.walk([&](Operation *op) {
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, DivFixedOp, CmpFixedOp,
                  MinFixedOp, MaxFixedOp>(op)) {
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
    if (opr_l.getType().isa<FixedType>()) { // fixed
      FixedType ltype = opr_l.getType().cast<FixedType>();
      FixedType rtype = opr_r.getType().cast<FixedType>();
      FixedType restype = res.getType().cast<FixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
      reswidth = restype.getWidth();
      resfrac = restype.getFrac();
    } else if (opr_l.getType().isa<UFixedType>()) { // ufixed
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
    // if op is MulFixedOp/DivFixedOp, double lwidth, rwidth, and reswidth
    if (llvm::isa<MulFixedOp>(op) || llvm::isa<DivFixedOp>(op)) {
      lwidth *= 2;
      rwidth *= 2;
      reswidth *= 2;
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
    if (opr_l.getType().isa<FixedType>()) {
      op->setAttr("sign", builder.getStringAttr("signed"));
    } else {
      op->setAttr("sign", builder.getStringAttr("unsigned"));
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
void markFixedCastOps(FuncOp &f) {
  // collect operations to mark
  SmallVector<Operation *, 10> fixedOps;
  f.walk([&](Operation *op) {
    if (llvm::isa<IntToFixedOp, FixedToIntOp, FloatToFixedOp, FixedToFloatOp,
                  FixedToFixedOp>(op)) {
      fixedOps.push_back(op);
    }
  });
  // They are unary ops, they have one operand
  for (auto op : fixedOps) {
    OpBuilder builder(f.getContext());
    Value opr = op->getOperand(0);
    Value res = op->getResult(0);
    // Mark operand's fixed-type information
    if (opr.getType().isa<FixedType>()) {
      FixedType srcType = opr.getType().cast<FixedType>();
      size_t width = srcType.getWidth();
      size_t frac = srcType.getFrac();
      IntegerType targetType = builder.getIntegerType(32);
      op->setAttr("src_width", builder.getIntegerAttr(targetType, width));
      op->setAttr("src_frac", builder.getIntegerAttr(targetType, frac));
      op->setAttr("src_sign", builder.getStringAttr("signed"));
    } else if (opr.getType().isa<UFixedType>()) {
      UFixedType srcType = opr.getType().cast<UFixedType>();
      size_t width = srcType.getWidth();
      size_t frac = srcType.getFrac();
      IntegerType targetType = builder.getIntegerType(32);
      op->setAttr("src_width", builder.getIntegerAttr(targetType, width));
      op->setAttr("src_frac", builder.getIntegerAttr(targetType, frac));
      op->setAttr("src_sign", builder.getStringAttr("unsigned"));
    }
    // Mark result's fixed-type information
    if (res.getType().isa<FixedType>()) {
      FixedType dstType = res.getType().cast<FixedType>();
      size_t width = dstType.getWidth();
      size_t frac = dstType.getFrac();
      IntegerType targetType = builder.getIntegerType(32);
      op->setAttr("dst_width", builder.getIntegerAttr(targetType, width));
      op->setAttr("dst_frac", builder.getIntegerAttr(targetType, frac));
      op->setAttr("dst_sign", builder.getStringAttr("signed"));
    } else if (res.getType().isa<UFixedType>()) {
      UFixedType dstType = res.getType().cast<UFixedType>();
      size_t width = dstType.getWidth();
      size_t frac = dstType.getFrac();
      IntegerType targetType = builder.getIntegerType(32);
      op->setAttr("dst_width", builder.getIntegerAttr(targetType, width));
      op->setAttr("dst_frac", builder.getIntegerAttr(targetType, frac));
      op->setAttr("dst_sign", builder.getStringAttr("unsigned"));
    }
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
  // llvm::outs() << "length of allocOps: " << allocOps.size() << "\n";

  for (auto op : allocOps) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType().cast<MemRefType>();
    Type t = memRefType.getElementType();
    size_t width;
    if (auto ft = t.dyn_cast_or_null<FixedType>()) {
      width = ft.getWidth();
    } else if (auto uft = t.dyn_cast_or_null<UFixedType>()) {
      width = uft.getWidth();
    } else {
      // Not a fixed-point alloc operation
      // Return without changing anything
      continue;
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
  } else if (v.getType().cast<IntegerType>().getWidth() > target_width) {
    // truncate bits
    result = builder.create<arith::TruncIOp>(loc, newType, v);
  } else {
    result = v;
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
  IntegerType intTy = IntegerType::get(op->getContext(), width);
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

// Lower FixedDivOp to DivSIOp/DivUIOp
void lowerFixedDiv(DivFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  size_t frac =
      op->getAttr("lfrac").cast<IntegerAttr>().getValue().getSExtValue();

  OpBuilder rewriter(op);
  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  // lhs<width, frac> / rhs<width, frac> -> res<width, 0>
  // Therefore, we need to left shift the lhs for frac bit
  // lhs<width, 2 * frac> / rhs<width, frac> -> res<width, frac>
  Type opTy = op->getOperand(0).getType();
  IntegerType intTy = IntegerType::get(op->getContext(), width);
  auto fracAttr = rewriter.getIntegerAttr(intTy, frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);
  arith::ShLIOp lhs_shifted =
      rewriter.create<arith::ShLIOp>(op->getLoc(), lhs, fracCstOp);
  if (opTy.isa<FixedType>()) { // fixed
    arith::DivSIOp res =
        rewriter.create<arith::DivSIOp>(op->getLoc(), lhs_shifted, rhs);
    op->replaceAllUsesWith(res);
  } else { // ufixed
    arith::DivUIOp res =
        rewriter.create<arith::DivUIOp>(op->getLoc(), lhs_shifted, rhs);
    op->replaceAllUsesWith(res);
  }
}

// Lower CmpFixedOp to CmpIOp
void lowerFixedCmp(CmpFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
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
void lowerFixedMin(MinFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(2), width);

  Type opTy = op->getOperand(0).getType();
  if (opTy.isa<FixedType>()) {
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
void lowerFixedMax(MaxFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(2), width);

  Type opTy = op->getOperand(0).getType();
  if (opTy.isa<FixedType>()) {
    // use signed integer max
    auto res = rewriter.create<arith::MaxSIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  } else {
    // use unsigned integer max
    auto res = rewriter.create<arith::MaxUIOp>(op->getLoc(), lhs, rhs);
    op->replaceAllUsesWith(res);
  }
}

// Build a memref.get_global operation that points to an I64 global memref
// The assumption is that all fixed-point encoding's global memrefs are of
// type I64.
void lowerGetGlobalFixedOp(GetGlobalFixedOp &op) {
  // TODO(Niansong): truncate the global memref to the width of the fixed-point
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  MemRefType oldType = op->getResult(0).getType().dyn_cast<MemRefType>();
  auto memRefType = oldType.clone(IntegerType::get(op.getContext(), 64));
  auto symbolName = op.global();
  auto res = rewriter.create<memref::GetGlobalOp>(loc, memRefType, symbolName);
  op->replaceAllUsesWith(res);
}

void lowerFixedToFloat(FixedToFloatOp &op) {
  OpBuilder rewriter(op);
  // size_t src_width =
  // op->getAttr("src_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t src_frac =
      op->getAttr("src_frac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign =
      op->getAttr("src_sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  auto dstTy = dst.getType().cast<FloatType>();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, dstTy, rewriter.getFloatAttr(dstTy, std::pow(2, src_frac)));
  if (isSigned) {
    auto res = rewriter.create<arith::SIToFPOp>(loc, dstTy, src);
    auto real = rewriter.create<arith::DivFOp>(loc, dstTy, res, frac);
    op->replaceAllUsesWith(real);
  } else {
    auto res = rewriter.create<arith::UIToFPOp>(loc, dstTy, src);
    auto real = rewriter.create<arith::DivFOp>(loc, dstTy, res, frac);
    op->replaceAllUsesWith(real);
  }
}

void lowerFloatToFixed(FloatToFixedOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  size_t dst_width =
      op->getAttr("dst_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t dst_frac =
      op->getAttr("dst_frac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign =
      op->getAttr("dst_sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  auto FType = src.getType().cast<FloatType>();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, FType, rewriter.getFloatAttr(FType, std::pow(2, dst_frac)));
  auto dstType = IntegerType::get(op.getContext(), dst_width);
  auto FEncoding = rewriter.create<arith::MulFOp>(loc, FType, src, frac);
  if (isSigned) {
    auto IEncoding = rewriter.create<arith::FPToSIOp>(loc, dstType, FEncoding);
    op->replaceAllUsesWith(IEncoding);
  } else {
    auto IEncoding = rewriter.create<arith::FPToUIOp>(loc, dstType, FEncoding);
    op->replaceAllUsesWith(IEncoding);
  }
}

void lowerFixedToInt(FixedToIntOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  auto dst = op.getResult();
  size_t src_frac =
      op->getAttr("src_frac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign =
      op->getAttr("src_sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  auto srcType = src.getType().cast<IntegerType>();
  auto dstType = dst.getType().cast<IntegerType>();
  size_t src_width = srcType.getWidth();
  size_t dst_width = dstType.getWidth();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, srcType, rewriter.getIntegerAttr(srcType, src_frac));
  if (isSigned) {
    auto rshifted = rewriter.create<arith::ShRSIOp>(loc, srcType, src, frac);
    if (dst_width > src_width) {
      auto res = rewriter.create<arith::ExtSIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else if (dst_width < src_width) {
      auto res = rewriter.create<arith::TruncIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else {
      op->replaceAllUsesWith(rshifted);
    }
  } else {
    auto rshifted = rewriter.create<arith::ShRUIOp>(loc, srcType, src, frac);
    if (dst_width > src_width) {
      auto res = rewriter.create<arith::ExtUIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else if (dst_width < src_width) {
      auto res = rewriter.create<arith::TruncIOp>(loc, dstType, rshifted);
      op->replaceAllUsesWith(res);
    } else {
      op->replaceAllUsesWith(rshifted);
    }
  }
}

void lowerIntToFixed(IntToFixedOp &op) {
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  size_t dst_width =
      op->getAttr("dst_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t dst_frac =
      op->getAttr("dst_frac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign =
      op->getAttr("dst_sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  auto srcType = src.getType().cast<IntegerType>();
  auto dstType = IntegerType::get(op.getContext(), dst_width);
  size_t src_width = srcType.getWidth();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, srcType, rewriter.getIntegerAttr(srcType, dst_frac));
  auto lshifted = rewriter.create<arith::ShLIOp>(loc, srcType, src, frac);
  if (dst_width > src_width) {
    if (isSigned) {
      auto res = rewriter.create<arith::ExtSIOp>(loc, dstType, lshifted);
      op->replaceAllUsesWith(res);
    } else {
      auto res = rewriter.create<arith::ExtUIOp>(loc, dstType, lshifted);
      op->replaceAllUsesWith(res);
    }
  } else if (dst_width < src_width) {
    auto res = rewriter.create<arith::TruncIOp>(loc, dstType, lshifted);
    op->replaceAllUsesWith(res);
  } else {
    op->replaceAllUsesWith(lshifted);
  }
}

// src and dst is guaranteed to be of different fixed types.
// src: src_width, src_frac
// dst: dst_width, dst_frac
// case 1: src_width > dst_width, src_frac > dst_frac
// case 2: src_width > dst_width, src_frac < dst_frac
// case 3: src_width < dst_width, src_frac > dst_frac
// case 4: src_width < dst_width, src_frac < dst_frac
// src_base * 2^(-src_frac) = dst_base * 2^(-dst_frac)
// ==> dst_base = src_base * 2^(dst_frac - src_frac)
void lowerFixedToFixed(FixedToFixedOp &op) {
  // Step 1: match bitwidth to max(src_width, dst_width)
  // Step 2: shift src_base to get dst_base
  //    - if dst_frac > src_frac, left shift (dst_frac - src_frac)
  //    - if dst_frac < src_frac, right shift (src_frac - dst_frac)
  // Step 3 (optional): truncate dst_base
  OpBuilder rewriter(op);
  auto loc = op.getLoc();
  auto src = op.getOperand();
  size_t src_width =
      op->getAttr("src_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t src_frac =
      op->getAttr("src_frac").cast<IntegerAttr>().getValue().getSExtValue();
  size_t dst_width =
      op->getAttr("dst_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t dst_frac =
      op->getAttr("dst_frac").cast<IntegerAttr>().getValue().getSExtValue();

  std::string src_sign =
      op->getAttr("src_sign").cast<StringAttr>().getValue().str();
  std::string dst_sign =
      op->getAttr("dst_sign").cast<StringAttr>().getValue().str();
  bool isSignedSrc = src_sign == "signed";
  // bool isSignedDst = dst_sign == "signed";

  auto srcType = src.getType().cast<IntegerType>();
  if (srcType.getWidth() != src_width) {
    llvm::errs() << "src_width != srcType.getWidth()\n";
  }
  auto dstType = IntegerType::get(op.getContext(), dst_width);

  // Step1: match bitwidth to max(src_width, dst_width)
  bool truncate_dst = false;
  Value matched_src;
  if (dst_width >= src_width) {
    // if (dst_width >= src_width), no need to truncate dst_base at step3
    truncate_dst = false;
    // extend src_base to dst_width
    if (isSignedSrc) {
      matched_src = rewriter.create<arith::ExtSIOp>(loc, dstType, src);
    } else {
      matched_src = rewriter.create<arith::ExtUIOp>(loc, dstType, src);
    }
  } else {
    // if (dst_width < src_width), truncate dst_base at step3
    truncate_dst = true;
    matched_src = src;
  }

  // Step2: shift src_base to get dst_base
  Value shifted_src;
  if (dst_frac > src_frac) {
    // if (dst_frac > src_frac), left shift (dst_frac - src_frac)
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, srcType, rewriter.getIntegerAttr(srcType, dst_frac - src_frac));
    shifted_src =
        rewriter.create<arith::ShLIOp>(loc, srcType, matched_src, frac);
  } else if (dst_frac < src_frac) {
    // if (dst_frac < src_frac), right shift (src_frac - dst_frac)
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, srcType, rewriter.getIntegerAttr(srcType, src_frac - dst_frac));
    if (isSignedSrc) {
      shifted_src =
          rewriter.create<arith::ShRSIOp>(loc, srcType, matched_src, frac);
    } else {
      shifted_src =
          rewriter.create<arith::ShRUIOp>(loc, srcType, matched_src, frac);
    }
  } else {
    shifted_src = matched_src;
  }

  // Step3 (optional): truncate dst_base
  if (truncate_dst) {
    auto res = rewriter.create<arith::TruncIOp>(loc, dstType, shifted_src);
    op->replaceAllUsesWith(res);
  } else {
    op->getResult(0).replaceAllUsesWith(shifted_src);
  }
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
  } else if (auto new_op = dyn_cast<DivFixedOp>(op)) {
    lowerFixedDiv(new_op);
  } else if (auto new_op = dyn_cast<CmpFixedOp>(op)) {
    lowerFixedCmp(new_op);
  } else if (auto new_op = dyn_cast<MinFixedOp>(op)) {
    lowerFixedMin(new_op);
  } else if (auto new_op = dyn_cast<MaxFixedOp>(op)) {
    lowerFixedMax(new_op);
  } else if (auto new_op = dyn_cast<AffineStoreOp>(op)) {
    updateAffineStore(new_op);
  } else if (auto new_op = dyn_cast<GetGlobalFixedOp>(op)) {
    lowerGetGlobalFixedOp(new_op);
  } else if (auto new_op = dyn_cast<FixedToFloatOp>(op)) {
    lowerFixedToFloat(new_op);
  } else if (auto new_op = dyn_cast<FloatToFixedOp>(op)) {
    lowerFloatToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToIntOp>(op)) {
    lowerFixedToInt(new_op);
  } else if (auto new_op = dyn_cast<IntToFixedOp>(op)) {
    lowerIntToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToFixedOp>(op)) {
    lowerFixedToFixed(new_op);
  }

  for (auto &region : op.getRegions()) {
    visitRegion(region);
  }
}

void visitBlock(Block &block) {
  SmallVector<Operation *, 10> opToRemove;
  for (auto &op : block.getOperations()) {
    visitOperation(op);
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, DivFixedOp, CmpFixedOp,
                  MinFixedOp, MaxFixedOp, IntToFixedOp, FixedToIntOp,
                  FloatToFixedOp, FixedToFloatOp, FixedToFixedOp,
                  GetGlobalFixedOp>(op)) {
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

/// Pass entry point
bool applyFixedPointToInteger(ModuleOp &mod) {

  for (FuncOp func : mod.getOps<FuncOp>()) {
    lowerPrintOp(func);
    markFixedArithOps(func);
    markFixedCastOps(func);
    FunctionType newFuncType = updateFunctionSignature(func);
    updateAffineLoad(func);
    updateAlloc(func);
    updateAffineLoad(func);
    visitRegion(func.getBody());
    updateAffineLoad(func);
    updateReturnOp(func);
    func.setType(newFuncType);
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