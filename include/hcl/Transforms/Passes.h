//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSFORMS_PASSES_H
#define HCL_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createMoveReturnToInputPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerCompositeTypePass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeCastPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveStrideMapPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformInterpreterPass();

bool applyLoopTransformation(ModuleOp &f);

bool applyFixedPointToInteger(ModuleOp &module);
bool applyAnyWidthInteger(ModuleOp &module);
bool applyMoveReturnToInput(ModuleOp &module);
bool applyLowerCompositeType(ModuleOp &module);
bool applyLowerBitOps(ModuleOp &module);
bool applyLegalizeCast(ModuleOp &module);
bool applyRemoveStrideMap(ModuleOp &module);

/// Registers all HCL transformation passes
void registerHCLPasses();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSFORMS_PASSES_H