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
// HCLParentLoopOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLParentLoopOp::apply(transform::TransformResults &results,
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
// HCLUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLUnrollOp::applyToOne(AffineForOp target,
                                   SmallVector<Operation *> &results,
                                   transform::TransformState &state) {
  if (failed(loopUnrollByFactor(target, getFactor()))) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to unroll";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// HCLSplitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLSplitOp::applyToOne(AffineForOp target,
                                  SmallVector<Operation *> &results,
                                  transform::TransformState &state) {
  SmallVector<AffineForOp, 2> splittedLoop;
  if (failed(tilePerfectlyNested({target}, {(unsigned)getFactor()},
                                 &splittedLoop))) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to split";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  results.append({splittedLoop.front(), splittedLoop.back()});
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// HCLPipelineOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HCLPipelineOp::applyToOne(AffineForOp target,
                                     SmallVector<Operation *> &results,
                                     transform::TransformState &state) {
  Builder b(target.getContext());
  target->setAttr("pipeline_ii", b.getI32IntegerAttr(getInitialInterval()));
  results.push_back(target);
  return DiagnosedSilenceableFailure(success());
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
