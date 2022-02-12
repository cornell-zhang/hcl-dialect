//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/Registration.h"
#include "hcl/Transforms/Passes.h"
#include "hcl/Conversion/HCLToLLVM.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/InitAllDialects.h"

void hclMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hcl::HeteroCLDialect>();
  // registry.insert<mlir::hcl::HeteroCLDialect, mlir::StandardOpsDialect,
  //                 mlir::AffineDialect, mlir::math::MathDialect,
  //                 mlir::memref::MemRefDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void hclMlirRegisterAllPasses() {
  mlir::hcl::registerHCLPasses();
  mlir::hcl::registerHCLToLLVMLoweringPass();
}