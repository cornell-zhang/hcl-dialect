//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSFORMS_PASSES_H
#define HCL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLoopTransformationPass();
std::unique_ptr<OperationPass<ModuleOp>> createFixedPointToIntegerPass();
std::unique_ptr<OperationPass<ModuleOp>> createAnyWidthIntegerPass();

bool applyLoopTransformation(ModuleOp &f);

bool applyHostXcelSeparation(ModuleOp &host_mod, ModuleOp &xcel_mod,
                             ModuleOp &extern_mod,
                             std::map<std::string, std::string> &device_map,
                             std::vector<std::string> &graph_roots,
                             std::vector<std::string> &subgraph_inputs,
                             std::vector<std::string> &subgraph_outputs);

bool applyFixedPointToInteger(ModuleOp &module);
bool applyAnyWidthInteger(ModuleOp &module);

/// Registers all HCL transformation passes
void registerHCLPasses();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSFORMS_PASSES_H