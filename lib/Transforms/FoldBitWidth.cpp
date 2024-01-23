/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <utility>

#include "hcl/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {
template <typename OpTy> struct FoldWidth : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static std::optional<IntegerType> operandIfExtended(Value operand) {
    auto *definingOp = operand.getDefiningOp();
    if (!definingOp)
      return std::nullopt;

    if (!isa<IntegerType>(operand.getType()))
      return std::nullopt;

    if (auto extOp = dyn_cast<arith::ExtSIOp>(*definingOp))
      return cast<IntegerType>(extOp->getOperand(0).getType());
    if (auto extOp = dyn_cast<arith::ExtUIOp>(*definingOp))
      return cast<IntegerType>(extOp->getOperand(0).getType());

    return std::nullopt;
  }

  static std::optional<IntegerType>
  valIfTruncated(TypedValue<IntegerType> val) {
    if (!val.hasOneUse())
      return std::nullopt;
    auto *op = *val.getUsers().begin();
    if (auto trunc = dyn_cast<arith::TruncIOp>(*op))
      if (auto truncType = dyn_cast<IntegerType>(trunc.getType()))
        return truncType;

    return std::nullopt;
  }

  static bool opIsLegal(OpTy op) {
    if (op->getNumResults() != 1)
      return true;
    if (op->getNumOperands() <= 0)
      return true;
    if (!isa<IntegerType>(op->getResultTypes().front()))
      return true;

    auto outType =
        valIfTruncated(cast<TypedValue<IntegerType>>(op->getResult(0)));
    if (!outType.has_value())
      return true;

    auto operandType = operandIfExtended(op->getOperand(0));
    if (!operandType.has_value() || operandType != outType)
      return true;

    // Extension and trunc should be opt away
    SmallVector<Value> operands;
    for (auto operand : op->getOperands()) {
      auto oW = operandIfExtended(operand);
      if (oW != operandType)
        return true;
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (opIsLegal(op))
      return failure();

    auto outType =
        valIfTruncated(cast<TypedValue<IntegerType>>(op->getResult(0)));

    // Extension and trunc should be opt away
    SmallVector<Value> operands;
    for (auto operand : op->getOperands())
      operands.push_back(operand.getDefiningOp()->getOperand(0));

    SmallVector<Type> resultTypes = {*outType};
    auto newOp = rewriter.create<OpTy>(op.getLoc(), resultTypes, operands);
    auto trunc = *op->getUsers().begin();
    trunc->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    rewriter.eraseOp(trunc);
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename OpTy> struct FoldLinalgWidth : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static unsigned getIndex(mlir::Block::OpListType &opList, Operation *item) {

    for (auto op : enumerate(opList))
      if (&op.value() == item)
        return op.index();
    assert(false && "Op not in Op list");
  }

  static SmallVector<Operation *> getUsersSorted(Value memref) {
    SmallVector<Operation *> users(memref.getUsers().begin(),
                                   memref.getUsers().end());

    std::sort(users.begin(), users.end(),
              [&memref](Operation *a, Operation *b) {
                return getIndex(memref.getParentBlock()->getOperations(), a) <
                       getIndex(memref.getParentBlock()->getOperations(), b);
              });

    return users;
  }

  static std::optional<std::pair<TypedValue<MemRefType>, linalg::GenericOp>>
  operandIfExtended(TypedValue<MemRefType> memref) {
    if (memref.getUsers().empty())
      return std::nullopt;

    auto users = getUsersSorted(memref);
    // If a buffer is used for the sake of type-conversion it should only have 2
    // uses.
    if (users.size() != 2)
      return std::nullopt;

    // If this is an extended operand, the first use should be a GenericOp that
    // extends
    if (!isa<linalg::GenericOp>(users.front()))
      return std::nullopt;

    auto genericOp = cast<linalg::GenericOp>(users.front());

    // Check that the Generic Op is used to extend with memref as an output
    if (genericOp.getOutputs().front() != memref ||
        genericOp.getBody()->getOperations().size() != 2 ||
        genericOp.getInputs().size() != 1)
      return std::nullopt;

    auto &operation = genericOp.getBody()->front();
    if (!isa<arith::ExtSIOp>(operation) && !isa<arith::ExtUIOp>(operation))
      return std::nullopt;

    // Return the memory buffer that is being extended and the GenericOp too
    return std::pair(
        cast<TypedValue<MemRefType>>(genericOp.getInputs().front()), genericOp);
  }

  static std::optional<std::pair<TypedValue<MemRefType>, linalg::GenericOp>>
  valIfTruncated(TypedValue<MemRefType> memref) {
    if (memref.getUsers().empty())
      return std::nullopt;

    auto users = getUsersSorted(memref);
    // If a buffer is used for the sake of type-conversion it should only have 2
    // uses.
    if (users.size() != 2)
      return std::nullopt;

    // If this is an truncated operand, the last use should be a GenericOp that
    // truncates
    if (!isa<linalg::GenericOp>(users.back()))
      return std::nullopt;

    auto genericOp = cast<linalg::GenericOp>(users.back());

    // Check that the Generic Op is used to truncate the memref input
    if (genericOp.getInputs().front() != memref ||
        genericOp.getBody()->getOperations().size() != 2 ||
        genericOp.getOutputs().size() != 1)
      return std::nullopt;

    auto &operation = genericOp.getBody()->front();
    if (!isa<arith::TruncIOp>(operation))
      return std::nullopt;

    // Return the memory buffer that is being truncated and the GenericOp too
    return std::pair(
        cast<TypedValue<MemRefType>>(genericOp.getOutputs().front()),
        genericOp);
  }

  // Test if we should apply this pattern or not
  static bool opIsLegal(OpTy op) {

    // Should be a binary operation
    if (op.getInputs().size() != 2)
      return true;
    if (op.getOutputs().size() != 1)
      return true;

    auto outType =
        valIfTruncated(cast<TypedValue<MemRefType>>(op.getOutputs().front()));
    if (!outType.has_value())
      return true;

    auto inputs = op.getInputs();
    auto firstOperand =
        operandIfExtended(cast<TypedValue<MemRefType>>(inputs[0]));
    if (!firstOperand.has_value() ||
        firstOperand->first.getType() != outType->first.getType())
      return true;

    auto secondOperand =
        operandIfExtended(cast<TypedValue<MemRefType>>(inputs[1]));
    if (!secondOperand.has_value() ||
        firstOperand->first.getType() != secondOperand->first.getType())
      return true;

    // At this point, we know all memref types are equivalent so the pattern
    // should be applied
    return false;
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (opIsLegal(op))
      return failure();

    auto outType =
        valIfTruncated(cast<TypedValue<MemRefType>>(op.getOutputs().front()));
    auto inputs = op.getInputs();
    auto firstOperand =
        operandIfExtended(cast<TypedValue<MemRefType>>(inputs[0]));
    auto secondOperand =
        operandIfExtended(cast<TypedValue<MemRefType>>(inputs[1]));

    // Extension and trunc should be opt away
    SmallVector<Value> operands({firstOperand->first, secondOperand->first});

    SmallVector<Value> results({outType->first});

    // Create the new linalg operation, and move the output memory buffer up in
    // the instructions so that it dominates
    auto newop = rewriter.create<OpTy>(op->getLoc(), operands, results);
    newop.getOutputs().front().getDefiningOp()->moveBefore(newop);

    // It is safe to delete these operations, because we force that each
    // memory buffer only has 2 uses
    rewriter.eraseOp(outType->second);
    rewriter.eraseOp(firstOperand->second);
    rewriter.eraseOp(secondOperand->second);
    rewriter.eraseOp(op);
    assert(opIsLegal(newop));

    return success();
  }
};
} // namespace hcl
} // namespace mlir

namespace {
struct HCLFoldBitWidthTransformation
    : public FoldBitWidthBase<HCLFoldBitWidthTransformation> {
  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    // Patterns for scalar wraparound operations
    patterns.add<FoldWidth<arith::AddIOp>>(context);
    patterns.add<FoldWidth<arith::SubIOp>>(context);
    patterns.add<FoldWidth<arith::MulIOp>>(context);

    // Targets for scalar wraparound operations
    target.addDynamicallyLegalOp<arith::AddIOp>(
        FoldWidth<arith::AddIOp>::opIsLegal);
    target.addDynamicallyLegalOp<arith::SubIOp>(
        FoldWidth<arith::SubIOp>::opIsLegal);
    target.addDynamicallyLegalOp<arith::MulIOp>(
        FoldWidth<arith::MulIOp>::opIsLegal);

    // Patterns for linalg wraparound operations
    patterns.add<FoldLinalgWidth<linalg::AddOp>>(context);
    patterns.add<FoldLinalgWidth<linalg::SubOp>>(context);
    patterns.add<FoldLinalgWidth<linalg::MulOp>>(context);

    // Targets for linalg wraparound operations
    target.addDynamicallyLegalOp<linalg::AddOp>(
        FoldLinalgWidth<linalg::AddOp>::opIsLegal);
    target.addDynamicallyLegalOp<linalg::SubOp>(
        FoldLinalgWidth<linalg::SubOp>::opIsLegal);
    target.addDynamicallyLegalOp<linalg::MulOp>(
        FoldLinalgWidth<linalg::MulOp>::opIsLegal);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
    OpBuilder builder(getOperation());
    IRRewriter rewriter(builder);
    (void)runRegionDCE(rewriter, getOperation()->getRegions());
  }
};
} // namespace

namespace mlir {
namespace hcl {
std::unique_ptr<OperationPass<ModuleOp>> createFoldBitWidthPass() {
  return std::make_unique<HCLFoldBitWidthTransformation>();
}
} // namespace hcl
} // namespace mlir
