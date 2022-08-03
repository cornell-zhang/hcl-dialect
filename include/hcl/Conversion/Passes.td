//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_MLIR_PASSES
#define HCL_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def HCLToLLVMLowering : Pass<"hcl-lower-to-llvm", "ModuleOp"> {
  let summary = "HCL to LLVM conversion pass";
  let constructor = "mlir::hcl::createHCLToLLVMLoweringPass()";
}

#endif // HCL_MLIR_PASSES