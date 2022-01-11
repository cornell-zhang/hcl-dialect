//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Registration.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/HeteroCLPasses.h"

void hclMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hcl::HeteroCLDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void hclMlirRegisterAllPasses() {
  mlir::hcl::registerHCLLoopTransformationPass();
}