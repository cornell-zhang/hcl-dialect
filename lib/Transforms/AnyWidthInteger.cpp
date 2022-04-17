//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AnyWidthInteger Pass
// This pass is to support any-width integer input from numpy.
// The input program has any-width integer input/output arguments
// The output program has 64-bit integer input/output and casts
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

/* CastIntMemRef
 * Allocate a new Int MemRef of target width and build a
 * AffineForOp loop nest to load, cast, store the elements
 * from oldMemRef to newMemRef.
 */
Value castIntMemRef(OpBuilder &builder, Location loc, const Value &oldMemRef,
                    size_t newWidth, bool unsign=false, bool replace = true, const Value& dstMemRef=NULL) {
  // first, alloc new memref
  MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
  Type newElementType = builder.getIntegerType(newWidth);
  MemRefType newMemRefType =
      oldMemRefType.clone(newElementType).cast<MemRefType>();
  Value newMemRef;
  if (!dstMemRef) {
    newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
  }
  // replace all uses
  if (replace)
    oldMemRef.replaceAllUsesWith(newMemRef);
  // build loop nest
  SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
  SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
  size_t oldWidth =
      oldMemRefType.getElementType().cast<IntegerType>().getWidth();
  buildAffineLoopNest(
      builder, loc, lbs, oldMemRefType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
        Value casted;
        if (newWidth < oldWidth) {
          // trunc
          casted =
              nestedBuilder.create<arith::TruncIOp>(loc, newElementType, v);
        } else if (newWidth > oldWidth){
          // extend
          if (unsign) {
            casted = nestedBuilder.create<arith::ExtUIOp>(loc, newElementType, v);
          } else {  
            casted = nestedBuilder.create<arith::ExtSIOp>(loc, newElementType, v);
          }
        } else {
          casted = v; // no cast happened
        }
        if (dstMemRef) {
          nestedBuilder.create<AffineStoreOp>(loc, casted, dstMemRef, ivs);
        } else {
          nestedBuilder.create<AffineStoreOp>(loc, casted, newMemRef, ivs);
        }
      });
  return newMemRef;
}

void updateTopFunctionSignature(FuncOp &funcOp) {
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
      // If result memref element type is integer
      // change it to i64 to be compatible with numpy
      if (et.isa<IntegerType>()) {
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
      // If argument memref element type is integer
      // change it to i64 to be compatible with numpy
      if (et.isa<IntegerType>()) {
        size_t width = 64;
        Type newElementType = IntegerType::get(funcOp.getContext(), width);
        new_arg_types.push_back(memrefType.clone(newElementType));
      } else {
        new_arg_types.push_back(memrefType);
      }
    }
  }

  // Get signedness hint information
  std::string extra_itypes = "";
  if (funcOp->hasAttr("extra_itypes")) {
    extra_itypes = funcOp->getAttr("extra_itypes").cast<StringAttr>().getValue().str();
  }
  std::string extra_otypes = "";
  if (funcOp->hasAttr("extra_otypes")) {
    extra_otypes = funcOp->getAttr("extra_otypes").cast<StringAttr>().getValue().str();
  }

  // Update FuncOp's block argument types
  // Also build loop nest to cast the input args
  SmallVector<Value, 4> newMemRefs;
  SmallVector<Value, 4> blockArgs;
  OpBuilder builder(funcOp->getRegion(0));
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); i++) {
      Type argType = block.getArgument(i).getType();
      if (MemRefType memrefType = argType.cast<MemRefType>()) {
        Type et = memrefType.getElementType();
        if (et.isa<IntegerType>()) {
          size_t width = 64;
          Type newType = IntegerType::get(funcOp.getContext(), width);
          Type newMemRefType = memrefType.clone(newType);
          size_t oldWidth = et.cast<IntegerType>().getWidth();
          block.getArgument(i).setType(newMemRefType);
          bool is_unsigned = false;
          if (i < extra_itypes.length()) {
            is_unsigned = extra_itypes[i] == 'u';
          }
          Value newMemRef = castIntMemRef(builder, funcOp->getLoc(), block.getArgument(i),
                        oldWidth, is_unsigned);
          newMemRefs.push_back(newMemRef);
          blockArgs.push_back(block.getArgument(i));
        }
      }
    }
  }

  // Update FuncOp's return types
  SmallVector<Operation *, 4> returnOps;
  funcOp.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<ReturnOp>(op)) {
      returnOps.push_back(op);
    }
  });
  for (auto op : returnOps) {
    OpBuilder returnRewriter(op);
    // Cast the return values
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      Value arg = op->getOperand(i);
      MemRefType type = arg.getType().cast<MemRefType>();
      Type etype = type.getElementType();
      if (etype.isa<IntegerType>()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(arg.getDefiningOp())) {
          bool is_unsigned = false;
          if (i < extra_otypes.length()) {
            is_unsigned = extra_otypes[i] == 'u';
          }
          Value newMemRef = castIntMemRef(returnRewriter, op->getLoc(),
                                          allocOp.getResult(), 64, is_unsigned, false);
          // Only replace the single use of oldMemRef: returnOp
          op->setOperand(i, newMemRef);
        }
      }
    }
    // Cast the input arguments
    for (auto v : llvm::enumerate(newMemRefs)) {
      Value newMemRef = v.value();
      Value& blockArg = blockArgs[v.index()];
      bool is_unsigned = false;
      if (v.index() < extra_itypes.length()) {
        is_unsigned = extra_itypes[v.index()] == 'u';
      }
      castIntMemRef(returnRewriter, op->getLoc(), newMemRef, 64, is_unsigned, false, blockArg);
    }
  }

  // Update function signature
  FunctionType newFuncType =
      FunctionType::get(funcOp.getContext(), new_arg_types, new_result_types);
  funcOp.setType(newFuncType);
}

/// entry point
bool applyAnyWidthInteger(ModuleOp &mod) {
  // Find top-level function
  bool isFoundTopFunc = false;
  FuncOp *topFunc;
  for (FuncOp func : mod.getOps<FuncOp>()) {
    if (func->hasAttr("top")) {
      isFoundTopFunc = true;
      topFunc = &func;
      break;
    }
  }

  if (isFoundTopFunc && topFunc) {
    updateTopFunctionSignature(*topFunc);
  }

  return true;
}

} // namespace hcl
} // namespace mlir

namespace {

struct HCLAnyWidthIntegerTransformation
    : public AnyWidthIntegerBase<HCLAnyWidthIntegerTransformation> {

  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyAnyWidthInteger(mod))
      return signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass() {
  return std::make_unique<HCLAnyWidthIntegerTransformation>();
}

} // namespace hcl
} // namespace mlir