//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCLTOLLVM_PASSES_H
#define HCLTOLLVM_PASSES_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

// HeteroCL Dialect -> LLVM Dialect
void registerHCLToLLVMLoweringPass();
std::unique_ptr<mlir::Pass> createHCLToLLVMLoweringPass();

} // namespace hcl
} // namespace mlir

#endif // HCLTOLLVM_PASSES_H