//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
namespace {
#define GEN_PASS_REGISTRATION
#include "hcl/Conversion/Passes.h.inc"
} // end namespace

void mlir::hcl::registerHCLConversionPasses() { ::registerPasses(); }
