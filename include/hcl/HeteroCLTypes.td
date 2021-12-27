//===- HeteroCLTypes.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TYPES
#define HCL_TYPES

include "HeteroCLDialect.td"

// // https://mlir.llvm.org/docs/Tutorials/DefiningAttributesAndTypes/#defining-custom-parsers-and-printers-using-assembly-formats
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td#L50
def LoopHandle : HeteroCL_Type<"LoopHandle"> {
  let summary = "loop handle";
  let mnemonic = "LoopHandle";
}

def StageHandle : HeteroCL_Type<"StageHandle"> {
  let summary = "stage handle";
  let mnemonic = "StageHandle";
}

#endif // HCL_TYPES