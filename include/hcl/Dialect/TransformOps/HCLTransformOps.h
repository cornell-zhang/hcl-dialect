/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H
#define MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace hcl {
class ForOp;
} // namespace hcl
} // namespace mlir

#define GET_OP_CLASSES
#include "hcl/Dialect/TransformOps/HCLTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace hcl {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace hcl
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORMOPS_HCLTRANSFORMOPS_H
