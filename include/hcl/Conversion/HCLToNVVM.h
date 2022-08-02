//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCLTONVVM_PASSES_H
#define HCLTONVVM_PASSES_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

// HeteroCL Dialect -> SCF(Parallel) Dialect
void registerAffineMemOpParLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createAffineMemOpParLoweringPass();

bool applyAffineMemOpParLoweringPass(ModuleOp &module, MLIRContext &context);

} // namespace hcl
} // namespace mlir


namespace mlir {
namespace hcl {

// SCF(Parallel) -> GPU Dialect
void registerAffineToGPULoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createAffineToGPULoweringPass();

bool applyAffineToGPULoweringPass(ModuleOp &module, MLIRContext &context);

} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {

// SCF(Parallel) -> GPU Dialect
void registerGPUToNVVMLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createGPUToNVVMLoweringPass();

bool applyGPUToNVVMLoweringPass(ModuleOp &module, MLIRContext &context);

} // namespace hcl
} // namespace mlir

#endif // HCLTONVVM_PASSES_H
