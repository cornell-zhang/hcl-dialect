//===- HeteroCLTypes.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TYPES
#define HCL_TYPES

// // https://mlir.llvm.org/docs/Tutorials/DefiningAttributesAndTypes/#defining-custom-parsers-and-printers-using-assembly-formats
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/TorchTypes.td#L50
def LoopType : HeteroCL_Type<"Loop"> {
  let summary = "loop handle";
  let mnemonic = "Loop";
  let parameters = (ins StringRefParameter<"loop name">:$loopName);
  let printer = [{
        $_printer << "<";
        llvm::printEscapedString(getImpl()->loopName, $_printer.getStream());
        $_printer << ">";
    }];
  let parser = [{
    if (parser.parseLess())
      return Type();
    StringRef loopName;
    if ($_parser.parseOptionalString(&loopName))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, loopName);
  }];
}

#endif // HCL_TYPES