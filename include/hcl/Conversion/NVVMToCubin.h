//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCLNVVMTOCUBIIN_H 
#define HCLNVVMTOCUBIIN_H

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>> createNVVMToCubinPass();
// bool applyNVVMToCubinPass(ModuleOp &module, MLIRContext &context);


} // namespace hcl
} // namespace mlir

#endif // HCLNVVMTOCUBIIN_H
