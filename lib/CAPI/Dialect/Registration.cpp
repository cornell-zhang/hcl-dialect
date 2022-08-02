//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/Registration.h"
#include "hcl/Conversion/HCLToLLVM.h"
#include "hcl/Conversion/HCLToNVVM.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/InitAllDialects.h"

void hclMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::hcl::HeteroCLDialect, mlir::StandardOpsDialect,
                  mlir::arith::ArithmeticDialect, mlir::tensor::TensorDialect,
                  mlir::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void hclMlirRegisterAllPasses() {
  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerConversionPasses();

  // Dialect passes
  mlir::registerAffinePasses();
  mlir::arith::registerArithmeticPasses();
  mlir::LLVM::registerLLVMPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerStandardPasses();
  mlir::registerLinalgPasses();

  mlir::hcl::registerHCLPasses();
  mlir::hcl::registerHCLToLLVMLoweringPass();

  // GPU backend passes
  mlir::hcl::registerAffineToGPULoweringPass();
  mlir::hcl::registerGPUToNVVMLoweringPass();
}