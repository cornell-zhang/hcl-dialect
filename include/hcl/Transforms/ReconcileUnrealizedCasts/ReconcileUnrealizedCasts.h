//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//


#ifndef HCL_TRANSFORMS_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_
#define HCL_TRANSFORMS_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>

namespace mlir {
namespace hcl {

/// Creates a pass that eliminates noop `unrealized_conversion_cast` operation
/// sequences.
std::unique_ptr<Pass> createReconcileUnrealizedCastsPass();

/// Populates `patterns` with rewrite patterns that eliminate noop
/// `unrealized_conversion_cast` operation sequences.
void populateReconcileUnrealizedCastsPatterns(RewritePatternSet &patterns);

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSFORMS_RECONCILEUNREALIZEDCASTS_RECONCILEUNREALIZEDCASTS_H_