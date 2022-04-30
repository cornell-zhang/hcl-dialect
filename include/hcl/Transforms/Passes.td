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

def LoopTransformation : Pass<"loop-opt", "ModuleOp"> {
  let summary = "Loop transformation pass";
  let constructor = "mlir::hcl::createLoopTransformationPass()";
}

def FixedToInteger : Pass<"fixed-to-integer", "ModuleOp"> {
  let summary = "Fixed-point operations to integer";
  let constructor = "mlir::hcl::createFixedPointToIntegerPass()";
}

def AnyWidthInteger : Pass<"anywidth-integer", "ModuleOp"> {
  let summary = "Transform anywidth-integer input to 64-bit";
  let constructor = "mlir::hcl::createAnyWidthIntegerPass()";
}

def MoveReturnToInput : Pass<"return-to-input", "ModuleOp"> {
  let summary = "Move return values to input argument list";
  let constructor = "mlir::hcl::createMoveReturnToInputPass()";
}

def LowerCompositeType : Pass<"lower-composite-type", "ModuleOp"> {
  let summary = "Lower composite types";
  let constructor = "mlir::hcl::createLowerCompositeTypePass()";
}

#endif // HCL_MLIR_PASSES