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
}

void lowerAffineLoad(FuncOp &f) {
  SmallVector<Operation *, 10> loads;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AffineLoadOp>(op)) {
      loads.push_back(op);
    }
  });

  for (auto op : loads) {
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

/* Add attributes to fixed-point operations
 * to preserve operands and result's fixed-type
 * information. After block arguments and
 * affine load operations are updated to integer
 * type, these information will not be directly
 * accessible through operands' types.
 */
void markFixedOperations(FuncOp &f) {
  SmallVector<Operation *, 10> addOps;
  f.walk([&](Operation *op) {
    if (auto add_op = dyn_cast<AddFixedOp>(op)) {
      addOps.push_back(op);
    }
  });
  // set attribute to addOps
  for (auto op : addOps) {
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
    }
    Type newType = IntegerType::get(f.getContext(), width);
    Type newMemRefType = memRefType.clone(newType);
    op->getResult(0).setType(newMemRefType); // alloc has only one result
  }
}

// Lower a Fixed-point add op to AddIOp
void lowerFixedAdd(AddFixedOp &op) {
  // FixedAddOps are binary ops, they have two operands
  Value opr_l = op->getOperand(0);
  Value opr_r = op->getOperand(1);
  // These values are of llvm::APInt type
  // Compare with operators in llvm::APInt class
  auto lwidth = op->getAttr("lwidth").cast<IntegerAttr>().getValue();
  auto lfrac = op->getAttr("lfrac").cast<IntegerAttr>().getValue();
  auto rwidth = op->getAttr("rwidth").cast<IntegerAttr>().getValue();
  auto rfrac = op->getAttr("rfrac").cast<IntegerAttr>().getValue();
  auto reswidth = op->getAttr("reswidth").cast<IntegerAttr>().getValue();
  auto resfrac = op->getAttr("resfrac").cast<IntegerAttr>().getValue();

  // Change result type
  IntegerType resType =
      IntegerType::get(op->getContext(), reswidth.getSExtValue());
  op->getResult(0).setType(resType);

  OpBuilder rewriter(op);
  auto loc = op->getLoc();
  arith::AddIOp newOp = rewriter.create<arith::AddIOp>(loc, opr_l, opr_r);
  op->replaceAllUsesWith(newOp);

  // Check width and cast

  // Check frac and shift
}

/// Visitors to recursively update all operations
void visitOperation(Operation &op);
void visitRegion(Region &region);
void visitBlock(Block &block);

void visitOperation(Operation &op) {
  SmallVector<Operation *, 10> opToRemove;

  if (auto new_op = dyn_cast<AddFixedOp>(op)) {
    lowerFixedAdd(new_op);
  }

  for (auto &region : op.getRegions()) {
    visitRegion(region);
  }
}

void visitBlock(Block &block) {
  SmallVector<Operation *, 10> opToRemove;
  for (auto &op : block.getOperations()) {
    visitOperation(op);
    if (llvm::isa<AddFixedOp>(op)) {
      opToRemove.push_back(&op);
    }
  }

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
    lowerAffineLoad(func);
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