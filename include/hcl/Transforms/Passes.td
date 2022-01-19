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
  let constructor = "mlir::hcl::createHCLLoopTransformationPass()";
  let dependentDialects = ["hcl::HeteroCLDialect"];
}

//===----------------------------------------------------------------------===//
// ReconcileUnrealizedCasts
//===----------------------------------------------------------------------===//

def ReconcileUnrealizedCasts : Pass<"reconcile-unrealized-casts"> {
  let summary = "Simplify and eliminate unrealized conversion casts";
  let description = [{
    Eliminate `unrealized_conversion_cast` operations, commonly introduced by
    partial dialect conversions, that transitively convert a value to another
    value of the same type, that is:

    ```
    %0 = "producer.op"() : () -> !type.A
    %1 = unrealized_conversion_cast %0 : !type.A to !type.B
    %2 = unrealized_conversion_cast %1 : !type.B to !type.A
    "consumer.op"(%2) : (!type.A) -> ()
    ```

    Such situations appear when the consumer operation is converted by one pass
    and the producer operation is converted by another pass, each of which
    produces an unrealized cast. This pass can be used to clean up the IR.
  }];
  let constructor = "mlir::hcl::createReconcileUnrealizedCastsPass();";
}



#endif // HCL_MLIR_PASSES