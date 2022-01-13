//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/Registration.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/Passes.h"
#include "hcl/Conversion/HCLToLLVM.h"

void hclMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hcl::HeteroCLDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void hclMlirRegisterAllPasses() {
  mlir::hcl::registerHCLLoopTransformationPass();
  mlir::hcl::registerHCLToLLVMLoweringPass();
}