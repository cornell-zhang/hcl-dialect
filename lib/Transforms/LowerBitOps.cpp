//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LowerBitOps Pass
// This file defines the lowering of bit operations.
// - GetBit
// - SetBit
// - GetSlice
// - SetSlice
// - BitReverse
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {

void lowerBitReverseOps(FuncOp &func) {
  SmallVector<Operation *, 8> bitReverseOps;
  func.walk([&](Operation *op) {
    if (auto bitReverseOp = dyn_cast<BitReverseOp>(op)) {
      bitReverseOps.push_back(bitReverseOp);
    }
  });

  for (auto op : bitReverseOps) {
    auto bitReverseOp = dyn_cast<BitReverseOp>(op);
    Value input = bitReverseOp.getOperand();
    Location loc = bitReverseOp.getLoc();
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    iwidth = iwidth - 1;
    OpBuilder rewriter(bitReverseOp);
    // Create two constants: number of bits, and zero
    Value const_width_i32 = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, iwidth, rewriter.getI32Type());
    Value const_width = rewriter.create<mlir::arith::IndexCastOp>(
        loc, const_width_i32, rewriter.getIndexType());
    SmallVector<Value> const_0_indices;
    const_0_indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // Create a single-element memref to store the result
    MemRefType memRefType = MemRefType::get({1}, input.getType());
    Value resultMemRef =
        rewriter.create<mlir::memref::AllocOp>(loc, memRefType);
    // Create a loop to iterate over the bits
    SmallVector<int64_t, 1> steps(1, 1);
    SmallVector<int64_t, 1> lbs(1, 0);
    SmallVector<int64_t, 1> ubs(1, iwidth);
    buildAffineLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          Value res =
              nestedBuilder.create<AffineLoadOp>(loc, resultMemRef, const_0_indices);
          // Get the bit at the width - current position
          Value reverse_idx = nestedBuilder.create<mlir::arith::SubIOp>(
              loc, const_width, ivs[0]);
          Type one_bit_type = nestedBuilder.getIntegerType(1);
          Value bit = nestedBuilder.create<mlir::hcl::GetIntBitOp>(
              loc, one_bit_type, input, reverse_idx);
          // Set the bit at the current position
          nestedBuilder.create<mlir::hcl::SetIntBitOp>(loc, res, ivs[0], bit);
          nestedBuilder.create<AffineStoreOp>(loc, res, resultMemRef, const_0_indices);
        });
    // Load the result from resultMemRef
    Value res = rewriter.create<mlir::AffineLoadOp>(loc, resultMemRef, const_0_indices);
    op->getResult(0).replaceAllUsesWith(res);
  }

  // remove the bitReverseOps
  std::reverse(bitReverseOps.begin(), bitReverseOps.end());
  for (auto op : bitReverseOps) {
    auto v = op->getResult(0);
    if (v.use_empty()) {
      op->erase();
    }
  }
}

/// Pass entry point
bool applyLowerBitOps(ModuleOp &mod) {
  for (FuncOp func : mod.getOps<FuncOp>()) {
    lowerBitReverseOps(func);
  }
  return true;
}
} // namespace hcl
} // namespace mlir

namespace {
struct HCLLowerBitOpsTransformation
    : public LowerBitOpsBase<HCLLowerBitOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerBitOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass() {
  return std::make_unique<HCLLowerBitOpsTransformation>();
}
} // namespace hcl
} // namespace mlir