//===- HCLTransformOps.cpp - Implementation of SCF transformation ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hcl/Dialect/TransformOps/HCLTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// HCLGetParentLoopOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLGetParentLoopOp::apply(transform::TransformResults &results,
                                     transform::TransformState &state) {
  SetVector<Operation *> parents;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    AffineForOp loop;
    Operation *current = target;
    for (unsigned i = 0, e = getNumLoops(); i < e; ++i) {
      loop = current->getParentOfType<AffineForOp>();
      if (!loop) {
        DiagnosedSilenceableFailure diag = emitSilenceableError()
                                           << "could not find an '"
                                           << AffineForOp::getOperationName()
                                           << "' parent";
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
      current = loop;
    }
    parents.insert(loop);
  }
  results.set(getResult().cast<OpResult>(), parents.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HCLLoopUnrollOp
//===----------------------------------------------------------------------===//

LogicalResult transform::HCLLoopUnrollOp::applyToOne(AffineForOp loop) {
  if (failed(loopUnrollByFactor(loop, getFactor())))
    return reportUnknownTransformError(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class HCLTransformDialectExtension
    : public transform::TransformDialectExtension<
          HCLTransformDialectExtension> {
public:
  HCLTransformDialectExtension() {
    declareDependentDialect<AffineDialect>();
    declareDependentDialect<func::FuncDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "hcl/Dialect/TransformOps/HCLTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "hcl/Dialect/TransformOps/HCLTransformOps.cpp.inc"

void mlir::hcl::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<HCLTransformDialectExtension>();
}
