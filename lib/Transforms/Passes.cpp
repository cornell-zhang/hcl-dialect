//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
namespace {
#define GEN_PASS_REGISTRATION
#include "hcl/Transforms/Passes.h.inc"
} // end namespace

void mlir::hcl::registerHCLPasses() {
    ::registerPasses();
}