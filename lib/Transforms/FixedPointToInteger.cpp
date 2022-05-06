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
  std::string extra_itypes = "";
  if (funcOp->hasAttr("extra_itypes")) {
    extra_itypes =
        funcOp->getAttr("extra_itypes").cast<StringAttr>().getValue().str();
  }
  std::string extra_otypes = "";
  if (funcOp->hasAttr("extra_otypes")) {
    extra_otypes =
        funcOp->getAttr("extra_otypes").cast<StringAttr>().getValue().str();
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
        // update the extra_otypes
        if (et.isa<FixedType>() and v.index() < extra_otypes.length()) {
          extra_otypes[v.index()] = 's';
        } else if (et.isa<UFixedType>() and v.index() < extra_otypes.length()) {
          extra_otypes[v.index()] = 'u';
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
        // update the extra_itypes
        if (et.isa<FixedType>() and v.index() < extra_itypes.length()) {
          extra_itypes[v.index()] = 's';
        } else if (et.isa<UFixedType>() and v.index() < extra_itypes.length()) {
          extra_itypes[v.index()] = 'u';
        }
      } else {
        new_arg_types.push_back(memrefType);
      }
    } else { // If argument type is not memref, do not change it
      new_arg_types.push_back(t);
    }
  }

  funcOp->setAttr("extra_itypes",
                  StringAttr::get(funcOp.getContext(), extra_itypes));
  funcOp->setAttr("extra_otypes",
                  StringAttr::get(funcOp.getContext(), extra_otypes));

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
          std::string extra_otypes = "";
          if (funcOp->hasAttr("extra_otypes")) {
            extra_otypes = funcOp->getAttr("extra_otypes")
                               .cast<StringAttr>()
                               .getValue()
                               .str();
          }
          bool is_unsigned = false;
          if (i < extra_otypes.length()) {
            is_unsigned = extra_otypes[i] == 'u';
          }
          Value castedMemRef =
              castIntMemRef(builder, loc, arg, 64, is_unsigned, false);
          op->setOperand(i, castedMemRef);
          // if (auto allocOp = dyn_cast<memref::AllocOp>(arg.getDefiningOp()))
          // {
          //   allocOp->getResult(0).setType(newType);
          //   for (auto &use : allocOp->getResult(0).getUses()) {
          //     Value storeEle;
          //     bool isStore= false;
          //     if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
          //       storeEle = storeOp.getOperand(0);
          //       isStore = true;
          //     } else if (auto storeOp =
          //                    dyn_cast<AffineStoreOp>(use.getOwner())) {
          //       storeEle = storeOp.getOperand(0);
          //       isStore = true;
          //     }
          //     if (isStore) {
          //       // check storeEle's type and cast it if necessary
          //       OpBuilder builder(use.getOwner());
          //       Location loc = use.getOwner()->getLoc();
          //       unsigned width;
          //       if (etype.isa<FixedType>()) {
          //         width = etype.cast<FixedType>().getWidth();
          //       } else {
          //         width = etype.cast<UFixedType>().getWidth();
          //       }
          //       Type oldType = builder.getIntegerType(width);
          //       if (oldType != IntegerType::get(funcOp.getContext(), 64)) {
          //         // cast it
          //         Value casted =
          //             castInteger(builder, loc, storeEle, oldType,
          //                         IntegerType::get(funcOp.getContext(), 64),
          //                         etype.isa<FixedType>());
          //         use.getOwner()->setOperand(0, casted);
          //       }
          //     }
          //   }
          // }
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
    // if op is MulFixedOp, double lwidth, rwidth, and reswidth
    if (llvm::isa<MulFixedOp>(op)) {
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
    if (llvm::isa<AddFixedOp, SubFixedOp, MulFixedOp, CmpFixedOp, MinFixedOp,
                  MaxFixedOp>(op)) {
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
    lowerPrintOp(func);
    markFixedOperations(func);
    FunctionType newFuncType = updateFunctionSignature(func);
    updateAffineLoad(func);
    updateAlloc(func);
    updateAffineLoad(func);
    for (Operation &op : func.getOps()) {
      visitOperation(op);
    }
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