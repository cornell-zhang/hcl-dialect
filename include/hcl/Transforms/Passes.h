//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSFORMS_PASSES_H
#define HCL_TRANSFORMS_PASSES_H

#include "hcl/Transforms/LoopTransformations.h"
#include "hcl/Transforms/ReconcileUnrealizedCasts.h"

namespace mlir {
namespace hcl {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "hcl/Transforms/Passes.h.inc"

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSFORMS_PASSES_H