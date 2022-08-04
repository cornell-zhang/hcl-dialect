//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCLTOLLVM_PASSES_H
#define HCLTOLLVM_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

// HeteroCL Dialect -> LLVM Dialect
std::unique_ptr<OperationPass<ModuleOp>> createHCLToLLVMLoweringPass();
bool applyHCLToLLVMLoweringPass(ModuleOp &module, MLIRContext &context);

void registerHCLConversionPasses();

#define GEN_PASS_CLASSES
#include "hcl/Conversion/Passes.h.inc"

} // namespace hcl
} // namespace mlir

#endif // HCLTOLLVM_PASSES_H
