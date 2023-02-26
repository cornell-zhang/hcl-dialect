/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hcl/Conversion/Passes.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

/* Cast integer to target bitwidth
 */
Value castIntegerWidth(MLIRContext *ctx, OpBuilder &builder, Location loc,
                       Value v, size_t target_width, bool is_signed) {
  Value result;
  Type newType = IntegerType::get(ctx, target_width);
  if (!v.getType().isa<IntegerType>()) {
    llvm::errs()
        << "castIntegerWidth: input is not integer type, input value is: " << v
        << "\n";
    assert(false);
  }
  if (v.getType().cast<IntegerType>().getWidth() < target_width) {
    // extend bits
    if (is_signed) {
      result = builder.create<arith::ExtSIOp>(loc, newType, v);
    } else {
      result = builder.create<arith::ExtUIOp>(loc, newType, v);
    }
  } else if (v.getType().cast<IntegerType>().getWidth() > target_width) {
    // truncate bits
    result = builder.create<arith::TruncIOp>(loc, newType, v);
  } else {
    result = v;
  }
  return result;
}

/* Update the function signature and
 * Because we need to interact with numpy, which only supports up
 * to 64-bit int/uint, so we update the input/output arguments
 * to 64-bit signless integer type. When the input memref
 */
FunctionType updateFunctionSignature(func::FuncOp &funcOp) {
  bool isTop = funcOp.getName() == "top";
  FunctionType functionType = funcOp.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types = llvm::to_vector<8>(functionType.getInputs());

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
        size_t real_width = et.isa<FixedType>()
                                ? et.cast<FixedType>().getWidth()
                                : et.cast<UFixedType>().getWidth();
        size_t width = isTop ? 64 : real_width;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
        // update the otypes
        if (et.isa<FixedType>() && v.index() < otypes.length()) {
          otypes[v.index()] = 's';
        } else if (et.isa<UFixedType>() && v.index() < otypes.length()) {
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
        size_t real_width = et.isa<FixedType>()
                                ? et.cast<FixedType>().getWidth()
                                : et.cast<UFixedType>().getWidth();
        size_t width = isTop ? 64 : real_width;
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

  // Update func::FuncOp's block argument types
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.dyn_cast<MemRefType>()) {
        Type et = memrefType.getElementType();
        if (et.isa<FixedType, UFixedType>()) {
          size_t real_width = et.isa<FixedType>()
                                  ? et.cast<FixedType>().getWidth()
                                  : et.cast<UFixedType>().getWidth();
          size_t width = isTop ? 64 : real_width;
          Type newType = IntegerType::get(funcOp.getContext(), width);
          Type newMemRefType = memrefType.clone(newType);
          // Set block argument type to new memref type
          block.getArgument(i).setType(newMemRefType);
          // Truncate the memref type to real_width
          if (isTop && real_width != 64) {
            OpBuilder rewriter(funcOp.getBody());
            Type truncType = IntegerType::get(funcOp.getContext(), real_width);
            Value truncMemRef = rewriter.create<memref::AllocOp>(
                block.getArgument(i).getLoc(),
                memrefType.clone(truncType).cast<MemRefType>());
            block.getArgument(i).replaceAllUsesWith(truncMemRef);
            SmallVector<int64_t, 4> lbs(memrefType.getRank(), 0);
            SmallVector<int64_t, 4> steps(memrefType.getRank(), 1);
            buildAffineLoopNest(
                rewriter, block.getArgument(i).getLoc(), lbs,
                memrefType.getShape(), steps,
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                  Value v = nestedBuilder.create<AffineLoadOp>(
                      loc, block.getArgument(i), ivs);
                  Value truncated =
                      nestedBuilder.create<arith::TruncIOp>(loc, truncType, v);
                  nestedBuilder.create<AffineStoreOp>(loc, truncated,
                                                      truncMemRef, ivs);
                });
          }
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
void updateAffineLoad(func::FuncOp &f) {
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
void updateReturnOp(func::FuncOp &funcOp) {
  bool isTop = funcOp.getName() == "top";
  if (!isTop)
    return; // Only update top function
  // Update func::FuncOp's return types
  SmallVector<Operation *, 4> returnOps;
  funcOp.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<func::ReturnOp>(op)) {
      returnOps.push_back(op);
    }
  });
  // get the return type of the function
  FunctionType funcType = funcOp.getFunctionType();
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

void updateSelectOp(arith::SelectOp &selectOp) {
  // update the result of select op
  // from fixed-point type to integer type
  Type resType = selectOp.getResult().getType();
  if (resType.isa<FixedType, UFixedType>()) {
    int bitwidth = resType.isa<FixedType>()
                       ? resType.cast<FixedType>().getWidth()
                       : resType.cast<UFixedType>().getWidth();
    Type newType = IntegerType::get(selectOp.getContext(), bitwidth);
    selectOp.getResult().setType(newType);
  }
  // Check that the operands of select op have the same type
  Type op0Type = selectOp.getOperand(1).getType(); // true branch
  Type op1Type = selectOp.getOperand(2).getType(); // false branch
  assert(op0Type == op1Type);
  assert(op0Type == selectOp.getResult().getType());
}

/* Update hcl.print (PrintOp) operations.
 * Create a float64 memref to store the real value
 * of hcl.print's operand memref
 */
void lowerPrintMemRefOp(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto new_op = dyn_cast<PrintMemRefOp>(op)) {
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

void lowerPrintOp(func::FuncOp &funcOp) {
  SmallVector<Operation *, 4> printOps;
  funcOp.walk([&](Operation *op) {
    if (auto new_op = dyn_cast<PrintOp>(op)) {
      // Only lower fixed-point prints
      for (auto operand : new_op->getOperands()) {
        if (operand.getType().isa<FixedType, UFixedType>()) {
          printOps.push_back(op);
          break;
        }
      }
    }
  });

  for (auto *printOp : printOps) {
    for (auto opr : llvm::enumerate(printOp->getOperands())) {
      if (opr.value().getType().isa<FixedType, UFixedType>()) {
        OpBuilder builder(printOp);
        Value oldValue = opr.value();
        bool is_unsigned = opr.value().getType().isa<UFixedType>();
        Value newValue = castToF64(builder, oldValue, is_unsigned);
        printOp->setOperand(opr.index(), newValue);
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
void markFixedArithOps(func::FuncOp &f) {
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
      // check that opr_r, res are fixed-point
      if (!opr_r.getType().isa<FixedType>()) {
        llvm::errs() << "Error: lhs or rhs are not fixed-point: "
                     << "operation: " << *op << "\n"
                     << "lhs type: " << opr_l.getType()
                     << ", rhs type: " << opr_r.getType()
                     << ", result type: " << res.getType() << "\n";
        assert(false);
      }
      FixedType ltype = opr_l.getType().cast<FixedType>();
      FixedType rtype = opr_r.getType().cast<FixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
      if (auto resType = res.getType().dyn_cast<FixedType>()) {
        reswidth = resType.getWidth();
        resfrac = resType.getFrac();
      } else {
        reswidth = res.getType().getIntOrFloatBitWidth();
        resfrac = 0;
      }
    } else if (opr_l.getType().isa<UFixedType>()) { // ufixed
      // check that opr_r, res are unsigned fixed-point
      if (!opr_r.getType().isa<UFixedType>()) {
        llvm::errs() << "Error: lhs or rhs are not unsigned fixed-point: "
                     << "operation: " << *op << "\n"
                     << "lhs type: " << opr_l.getType()
                     << ", rhs type: " << opr_r.getType()
                     << ", result type: " << res.getType() << "\n";
        assert(false);
      }
      UFixedType ltype = opr_l.getType().cast<UFixedType>();
      UFixedType rtype = opr_r.getType().cast<UFixedType>();
      lwidth = ltype.getWidth();
      lfrac = ltype.getFrac();
      rwidth = rtype.getWidth();
      rfrac = rtype.getFrac();
      if (auto resType = res.getType().dyn_cast<UFixedType>()) {
        reswidth = resType.getWidth();
        resfrac = resType.getFrac();
      } else {
        reswidth = res.getType().getIntOrFloatBitWidth();
        resfrac = 0;
      }
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
void markFixedCastOps(func::FuncOp &f) {
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
void updateAlloc(func::FuncOp &f) {
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

void updateSCFIfOp(mlir::scf::IfOp &op) {
  for (auto res : op.getResults()) {
    if (res.getType().isa<FixedType>()) {
      res.setType(IntegerType::get(res.getContext(),
                                   res.getType().cast<FixedType>().getWidth()));
    } else if (res.getType().isa<UFixedType>()) {
      res.setType(IntegerType::get(
          res.getContext(), res.getType().cast<UFixedType>().getWidth()));
    } else if (auto memRefType = res.getType().dyn_cast<MemRefType>()) {
      Type eleTyp = memRefType.getElementType();
      if (eleTyp.isa<FixedType>()) {
        eleTyp = IntegerType::get(res.getContext(),
                                  eleTyp.cast<FixedType>().getWidth());
      } else if (eleTyp.isa<UFixedType>()) {
        eleTyp = IntegerType::get(res.getContext(),
                                  eleTyp.cast<UFixedType>().getWidth());
      }
      res.setType(memRefType.clone(eleTyp));
    }
  }
  // llvm::outs() << op << "\n";
}

// Lower AddFixedOp to AddIOp
void lowerFixedAdd(AddFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  OpBuilder rewriter(op);

  // llvm::outs() << "lhs" << op->getOperand(0) << "\n";
  // llvm::outs() << "rhs: " << op->getOperand(1) << "\n";
  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

  arith::AddIOp newOp = rewriter.create<arith::AddIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower FixedSubOp to SubIOp
void lowerFixedSub(SubFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

  arith::SubIOp newOp = rewriter.create<arith::SubIOp>(op->getLoc(), lhs, rhs);
  op->replaceAllUsesWith(newOp);
}

// Lower MulFixedop to MulIOp
void lowerFixedMul(MulFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  size_t frac =
      op->getAttr("lfrac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";

  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

  arith::MulIOp newOp = rewriter.create<arith::MulIOp>(op->getLoc(), lhs, rhs);

  // lhs<width, frac> * rhs<width, frac> -> res<width, 2*frac>
  // Therefore, we need to right shift the result for frac bit
  // Right shift needs to consider signed/unsigned
  Type opTy = op->getOperand(0).getType();
  IntegerType intTy = IntegerType::get(op->getContext(), width);
  IntegerType truncTy = IntegerType::get(op->getContext(), width / 2);
  auto fracAttr = rewriter.getIntegerAttr(intTy, frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);

  if (opTy.isa<FixedType>()) {
    // use signed right shift
    arith::ShRSIOp res =
        rewriter.create<arith::ShRSIOp>(op->getLoc(), newOp, fracCstOp);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  } else {
    // use unsigned right shift
    arith::ShRUIOp res =
        rewriter.create<arith::ShRUIOp>(op->getLoc(), newOp, fracCstOp);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  }
}

// Lower FixedDivOp to DivSIOp/DivUIOp
void lowerFixedDiv(DivFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  size_t frac =
      op->getAttr("lfrac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";

  OpBuilder rewriter(op);
  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);
  // lhs<width, frac> / rhs<width, frac> -> res<width, 0>
  // Therefore, we need to left shift the lhs for frac bit
  // lhs<width, 2 * frac> / rhs<width, frac> -> res<width, frac>
  Type opTy = op->getOperand(0).getType();
  IntegerType intTy = IntegerType::get(op->getContext(), width);
  IntegerType truncTy = IntegerType::get(op->getContext(), width / 2);
  auto fracAttr = rewriter.getIntegerAttr(intTy, frac);
  auto fracCstOp =
      rewriter.create<arith::ConstantOp>(op->getLoc(), intTy, fracAttr);
  arith::ShLIOp lhs_shifted =
      rewriter.create<arith::ShLIOp>(op->getLoc(), lhs, fracCstOp);
  if (opTy.isa<FixedType>()) { // fixed
    arith::DivSIOp res =
        rewriter.create<arith::DivSIOp>(op->getLoc(), lhs_shifted, rhs);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  } else { // ufixed
    arith::DivUIOp res =
        rewriter.create<arith::DivUIOp>(op->getLoc(), lhs_shifted, rhs);
    auto truncated =
        rewriter.create<arith::TruncIOp>(op->getLoc(), truncTy, res);
    op->replaceAllUsesWith(truncated);
  }
}

// Lower CmpFixedOp to CmpIOp
void lowerFixedCmp(CmpFixedOp &op) {
  // llvm::outs() << op << "\n";
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  // llvm::outs() << "width: " << width << "\n";
  // llvm::outs() << "sign: " << sign << "\n";
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

  // llvm::outs() << "lhs: " << lhs << "\n";
  // llvm::outs() << "rhs: " << rhs << "\n";

  // auto prednum =
  // op->getAttr("predicate").cast<IntegerAttr>().getValue().getSExtValue();
  auto prednum = op.getPredicate();
  auto loc = op->getLoc();
  arith::CmpIOp newOp;
  switch (prednum) {
  case hcl::CmpFixedPredicate::eq:
    newOp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
    break;
  case hcl::CmpFixedPredicate::ne:
    newOp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
    break;
  case hcl::CmpFixedPredicate::slt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::sle:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::sgt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::sge:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::ult:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::ule:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::ugt:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs,
                                           rhs);
    break;
  case hcl::CmpFixedPredicate::uge:
    newOp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, lhs,
                                           rhs);
    break;
  default:
    llvm::errs() << "unknown predicate code in CmpFixedOp\n";
  }

  // llvm::outs() << "newOp: " << newOp << "\n";

  op->replaceAllUsesWith(newOp);
}

// Lower MinFixedOp to MinSIOp or MinUIOp
void lowerFixedMin(MinFixedOp &op) {
  size_t width =
      op->getAttr("lwidth").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

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
  std::string sign = op->getAttr("sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  OpBuilder rewriter(op);

  Value lhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(0), width, isSigned);
  Value rhs = castIntegerWidth(op->getContext(), rewriter, op->getLoc(),
                               op->getOperand(1), width, isSigned);

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
  Type oldElementType = oldType.getElementType();
  bool isSigned;
  if (auto fixedTy = oldElementType.dyn_cast<FixedType>()) {
    isSigned = true;
  } else {
    isSigned = false;
  }
  auto memRefType = oldType.clone(IntegerType::get(op.getContext(), 64));
  auto symbolName = op.name();
  auto res = rewriter.create<memref::GetGlobalOp>(loc, memRefType, symbolName);
  // Truncate or Extend I64 memref to the width of the fixed-point
  size_t bitwidth;
  if (auto fixedType = oldElementType.dyn_cast<FixedType>()) {
    bitwidth = fixedType.getWidth();
  } else if (auto ufixedType = oldElementType.dyn_cast<UFixedType>()) {
    bitwidth = ufixedType.getWidth();
  } else {
    llvm::errs() << "unknown fixed-point type in GetGlobalFixedOp\n";
    return;
  }
  auto castedMemRefType =
      oldType.clone(IntegerType::get(op.getContext(), bitwidth))
          .cast<MemRefType>();
  auto castedMemRef = rewriter.create<memref::AllocOp>(loc, castedMemRefType);
  SmallVector<int64_t, 4> lbs(oldType.getRank(), 0);
  SmallVector<int64_t, 4> steps(oldType.getRank(), 1);
  buildAffineLoopNest(
      rewriter, loc, lbs, oldType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        auto v = nestedBuilder.create<AffineLoadOp>(loc, res, ivs);
        Value casted = castIntegerWidth(op.getContext(), nestedBuilder, loc, v,
                                        bitwidth, isSigned);
        nestedBuilder.create<AffineStoreOp>(loc, casted, castedMemRef, ivs);
      });

  op->replaceAllUsesWith(castedMemRef);
  // update affine.load operations res type to be consistent with castedMemRef's
  // element type
  for (auto &use : castedMemRef.getResult().getUses()) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(use.getOwner())) {
      for (auto v : llvm::enumerate(loadOp->getResults())) {
        Type newType =
            loadOp->getOperand(0).getType().cast<MemRefType>().getElementType();
        loadOp->getResult(v.index()).setType(newType);
      }
    }
  }
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
  size_t src_width =
      op->getAttr("src_width").cast<IntegerAttr>().getValue().getSExtValue();
  size_t src_frac =
      op->getAttr("src_frac").cast<IntegerAttr>().getValue().getSExtValue();
  std::string sign =
      op->getAttr("src_sign").cast<StringAttr>().getValue().str();
  bool isSigned = sign == "signed";
  auto dstType = dst.getType().cast<IntegerType>();
  auto srcType = IntegerType::get(op.getContext(), src_width);
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
  // size_t src_width = srcType.getWidth();
  auto frac = rewriter.create<arith::ConstantOp>(
      loc, dstType, rewriter.getIntegerAttr(dstType, dst_frac));

  Value bitAdjusted = castIntegerWidth(op->getContext(), rewriter, loc, src,
                                       dst_width, isSigned);
  auto lshifted =
      rewriter.create<arith::ShLIOp>(loc, dstType, bitAdjusted, frac);
  op->replaceAllUsesWith(lshifted);
}

void updateCallOp(func::CallOp &op) {
  // get the callee function signature type
  FunctionType callee_type = op.getCalleeType();
  auto callee = op.getCallee();
  assert(callee != "top" && "updateCallOp: assume callee is not top");
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(callee_type.getResults());
  llvm::SmallVector<Type, 4> new_result_types;
  llvm::SmallVector<Type, 4> new_arg_types;
  for (auto v : llvm::enumerate(result_types)) {
    Type t = v.value();
    if (MemRefType memrefType = t.dyn_cast<MemRefType>()) {
      Type et = memrefType.getElementType();
      if (et.isa<FixedType, UFixedType>()) {
        size_t width = et.isa<FixedType>() ? et.cast<FixedType>().getWidth()
                                           : et.cast<UFixedType>().getWidth();
        Type newElementType = IntegerType::get(op.getContext(), width);
        new_result_types.push_back(memrefType.clone(newElementType));
      } else {
        new_result_types.push_back(memrefType);
      }
    } else { // If result type is not memref, error out
      op.emitError("updateCallOp: result type is not memref");
    }
  }
  // set call op result types to new_result_types
  for (auto v : llvm::enumerate(new_result_types)) {
    op.getResult(v.index()).setType(v.value());
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

  // auto srcType = src.getType().cast<IntegerType>();
  // if (srcType.getWidth() != src_width) {
  //   llvm::errs() << "src_width != srcType.getWidth()\n";
  // }
  auto srcType = IntegerType::get(op.getContext(), src_width);
  auto dstType = IntegerType::get(op.getContext(), dst_width);

  // Step1: match bitwidth to max(src_width, dst_width)
  bool truncate_dst = false;
  bool match_to_dst = false;
  Value matched_src;
  if (dst_width > src_width) {
    // if (dst_width > src_width), no need to truncate dst_base at step3
    truncate_dst = false;
    match_to_dst = true;
    // extend src_base to dst_width
    if (isSignedSrc) {
      matched_src = rewriter.create<arith::ExtSIOp>(loc, dstType, src);
    } else {
      matched_src = rewriter.create<arith::ExtUIOp>(loc, dstType, src);
    }
  } else if (dst_width == src_width) {
    truncate_dst = false;
    match_to_dst = false;
    matched_src = src;
  } else {
    // if (dst_width < src_width), truncate dst_base at step3
    truncate_dst = true;
    match_to_dst = false;
    matched_src = src;
  }

  // Step2: shift src_base to get dst_base
  Value shifted_src;
  if (dst_frac > src_frac) {
    // if (dst_frac > src_frac), left shift (dst_frac - src_frac)
    Type shiftType = match_to_dst ? dstType : srcType;
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, shiftType,
        rewriter.getIntegerAttr(shiftType, dst_frac - src_frac));
    shifted_src =
        rewriter.create<arith::ShLIOp>(loc, shiftType, matched_src, frac);
  } else if (dst_frac < src_frac) {
    // if (dst_frac < src_frac), right shift (src_frac - dst_frac)
    Type shiftType = match_to_dst ? dstType : srcType;
    auto frac = rewriter.create<arith::ConstantOp>(
        loc, shiftType,
        rewriter.getIntegerAttr(shiftType, src_frac - dst_frac));
    if (isSignedSrc) {
      shifted_src =
          rewriter.create<arith::ShRSIOp>(loc, shiftType, matched_src, frac);
    } else {
      shifted_src =
          rewriter.create<arith::ShRUIOp>(loc, shiftType, matched_src, frac);
    }
  } else {
    shifted_src = matched_src;
  }

  // debug output
  // llvm::outs() << shifted_src << "\n";

  // Step3 (optional): truncate dst_base
  if (truncate_dst) {
    auto res = rewriter.create<arith::TruncIOp>(loc, dstType, shifted_src);
    op->replaceAllUsesWith(res);
  } else {
    op->getResult(0).replaceAllUsesWith(shifted_src);
  }
}

void validateLoweredFunc(func::FuncOp &func) {
  // check if result types and input types are not fixed or ufixed
  FunctionType functionType = func.getFunctionType();
  SmallVector<Type, 4> result_types =
      llvm::to_vector<4>(functionType.getResults());
  SmallVector<Type, 8> arg_types = llvm::to_vector<8>(functionType.getInputs());
  for (auto result_type : result_types) {
    if (result_type.isa<FixedType>() || result_type.isa<UFixedType>()) {
      func.emitError(
          "FuncOp: " + func.getName().str() +
          " has fixed-point type result type: " +
          " which means it is not lowered by FixedPointToInteger pass\n");
    }
  }
  for (auto arg_type : arg_types) {
    if (arg_type.isa<FixedType>() || arg_type.isa<UFixedType>()) {
      func.emitError(
          "FuncOp: " + func.getName().str() +
          " has fixed-point type arg type: " +
          " which means it is not lowered by FixedPointToInteger pass\n");
    }
  }

  // check if all operations are lowered
  for (auto &block : func.getBody().getBlocks()) {
    for (auto &op : block.getOperations()) {
      // check the result type and arg types of op
      if (op.getNumResults() > 0) {
        for (auto result : op.getResults()) {
          if (result.getType().isa<FixedType>() ||
              result.getType().isa<UFixedType>()) {
            op.emitError(
                "FuncOp: " + func.getName().str() +
                " has op: " + std::string(op.getName().getStringRef()) +
                " with fixed-point result type" +
                " which means it is not lowered by FixedPointToInteger pass\n");
            llvm::errs() << "op that failed validation: " << op << "\n";
          }
        }
      }
      // check the arg types of op
      for (auto arg : op.getOperands()) {
        if (arg.getType().isa<FixedType>() || arg.getType().isa<UFixedType>()) {
          op.emitError(
              "FuncOp: " + func.getName().str() +
              " has op: " + std::string(op.getName().getStringRef()) +
              " with fixed-point arg type" +
              " which means it is not lowered by FixedPointToInteger pass\n");
          llvm::errs() << "op that failed validation: " << op << "\n";
        }
      }
    }
  }
}

/// Visitors to recursively update all operations
void visitOperation(Operation &op);
void visitRegion(Region &region);
void visitBlock(Block &block);

void visitOperation(Operation &op) {
  if (auto new_op = dyn_cast<AddFixedOp>(op)) {
    // llvm::outs() << "AddFixedOp\n";
    lowerFixedAdd(new_op);
  } else if (auto new_op = dyn_cast<SubFixedOp>(op)) {
    // llvm::outs() << "SubFixedOp\n";
    lowerFixedSub(new_op);
  } else if (auto new_op = dyn_cast<MulFixedOp>(op)) {
    // llvm::outs() << "MulFixedOp\n";
    lowerFixedMul(new_op);
  } else if (auto new_op = dyn_cast<DivFixedOp>(op)) {
    // llvm::outs() << "DivFixedOp\n";
    lowerFixedDiv(new_op);
  } else if (auto new_op = dyn_cast<CmpFixedOp>(op)) {
    // llvm::outs() << "CmpFixedOp\n";
    lowerFixedCmp(new_op);
  } else if (auto new_op = dyn_cast<MinFixedOp>(op)) {
    // llvm::outs() << "MinFixedOp\n";
    lowerFixedMin(new_op);
  } else if (auto new_op = dyn_cast<MaxFixedOp>(op)) {
    // llvm::outs() << "MaxFixedOp\n";
    lowerFixedMax(new_op);
  } else if (auto new_op = dyn_cast<AffineStoreOp>(op)) {
    // llvm::outs() << "AffineStoreOp\n";
    updateAffineStore(new_op);
  } else if (auto new_op = dyn_cast<GetGlobalFixedOp>(op)) {
    // llvm::outs() << "GetGlobalFixedOp\n";
    lowerGetGlobalFixedOp(new_op);
  } else if (auto new_op = dyn_cast<FixedToFloatOp>(op)) {
    // llvm::outs() << "FixedToFloatOp\n";
    lowerFixedToFloat(new_op);
  } else if (auto new_op = dyn_cast<FloatToFixedOp>(op)) {
    // llvm::outs() << "FloatToFixedOp\n";
    lowerFloatToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToIntOp>(op)) {
    // llvm::outs() << "FixedToIntOp\n";
    lowerFixedToInt(new_op);
  } else if (auto new_op = dyn_cast<IntToFixedOp>(op)) {
    // llvm::outs() << "IntToFixedOp\n";
    lowerIntToFixed(new_op);
  } else if (auto new_op = dyn_cast<FixedToFixedOp>(op)) {
    // llvm::outs() << "FixedToFixedOp\n";
    // llvm::outs() << *op.getParentOp() << "\n";
    lowerFixedToFixed(new_op);
    // debug output
    // llvm::outs() << *op.getParentOp() << "\n";
  } else if (auto new_op = dyn_cast<scf::IfOp>(op)) {
    // llvm::outs() << "IfOp\n";
    updateSCFIfOp(new_op);
  } else if (auto new_op = dyn_cast<func::CallOp>(op)) {
    updateCallOp(new_op);
  } else if (auto new_op = dyn_cast<arith::SelectOp>(op)) {
    updateSelectOp(new_op);
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

  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    lowerPrintMemRefOp(func);
    lowerPrintOp(func);
    markFixedArithOps(func);
    markFixedCastOps(func);
    // llvm::outs() << "markFixedCastOps done\n";
    FunctionType newFuncType = updateFunctionSignature(func);
    // llvm::outs() << "updateFunctionSignature done\n";
    updateAffineLoad(func);
    // llvm::outs() << "updateAffineLoad done\n";
    updateAlloc(func);
    // llvm::outs() << "updateAlloc done\n";
    updateAffineLoad(func);
    // llvm::outs() << "updateAffineLoad done\n";
    visitRegion(func.getBody());
    // llvm::outs() << "visitRegion done\n";
    updateAffineLoad(func);
    // llvm::outs() << "updateAffineLoad done\n";
    updateReturnOp(func);
    // llvm::outs() << "updateReturnOp done\n";
    func.setType(newFuncType);
    // validate
    validateLoweredFunc(func);
  }

  // llvm::outs() << mod << "\n";

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