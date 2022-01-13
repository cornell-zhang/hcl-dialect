//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
#include "../PassDetail.h"
#include "hcl/Transforms/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "hcl/Transforms/Passes.h"

using namespace mlir;
using namespace hcl;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_CLASSES
#include "hcl/Transforms/Passes.h.inc"
} // end namespace

namespace {

/// Removes `unrealized_conversion_cast`s whose results are only used by other
/// `unrealized_conversion_cast`s converting back to the original type. This
/// pattern is complementary to the folder and can be used to process operations
/// starting from the first, i.e. the usual traversal order in dialect
/// conversion. The folder, on the other hand, can only apply to the last
/// operation in a chain of conversions because it is not expected to walk
/// use-def chains. One would need to declare cast ops as dynamically illegal
/// with a complex condition in order to eliminate them using the folder alone
/// in the dialect conversion infra.
struct UnrealizedConversionCastPassthrough
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // Match the casts that are _only_ used by other casts, with the overall
    // cast being a trivial noop: A->B->A.
    auto users = op->getUsers();
    if (!llvm::all_of(users, [&](Operation *user) {
          if (auto other = dyn_cast<UnrealizedConversionCastOp>(user))
            return other.getResultTypes() == op.inputs().getTypes() &&
                   other.inputs() == op.outputs();
          return false;
        })) {
      return rewriter.notifyMatchFailure(op, "live unrealized conversion cast");
    }

    for (Operation *user : users)
      rewriter.replaceOp(user, op.inputs());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pass to simplify and eliminate unrealized conversion casts.
struct ReconcileUnrealizedCasts
    : public ReconcileUnrealizedCastsBase<ReconcileUnrealizedCasts> {
  ReconcileUnrealizedCasts() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateReconcileUnrealizedCastsPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace hcl {

void populateReconcileUnrealizedCastsPatterns(RewritePatternSet &patterns) {
  patterns.add<UnrealizedConversionCastPassthrough>(patterns.getContext());
}

std::unique_ptr<Pass> createReconcileUnrealizedCastsPass() {
  return std::make_unique<ReconcileUnrealizedCasts>();
}

} // namespace hcl
} // namespace mlir