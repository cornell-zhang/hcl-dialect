//===- HeteroCLTypes.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TYPES
#define HCL_TYPES

include "hcl/Dialect/HeteroCLDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

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

def Fixed : HeteroCL_Type<"Fixed", [MemRefElementTypeInterface]> {
  let summary = "fixed point";
  let mnemonic = "Fixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let printer = [{
    $_printer << "<" << getImpl()->width << ", " << getImpl()->frac << ">";
  }];
  let parser = [{
    if (parser.parseLess())
      return Type();
    std::size_t width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseComma())
      return Type();
    std::size_t frac;
    if ($_parser.parseInteger(frac))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width, frac);
  }];
}

def UFixed : HeteroCL_Type<"UFixed", [MemRefElementTypeInterface]> {
  let summary = "unsigned fixed point";
  let mnemonic = "UFixed";
  let parameters = (ins "std::size_t":$width, "std::size_t":$frac);
  let printer = [{
    $_printer << "<" << getImpl()->width << ", " << getImpl()->frac << ">";
  }];
  let parser = [{
    if (parser.parseLess())
      return Type();
    std::size_t width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseComma())
      return Type();
    std::size_t frac;
    if ($_parser.parseInteger(frac))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width, frac);
  }];
}

def Bfloat : HeteroCL_Type<"Bfloat", [MemRefElementTypeInterface]> {
  let summary = "Bfloat floating number";
  let mnemonic = "Bfloat";
  let parameters = (ins "std::size_t":$exp, "std::size_t":$frac);
  let printer = [{
    $_printer << "<" << getImpl()->exp<< ", " << getImpl()->frac<< ">";
  }];
  let parser = [{
    if (parser.parseLess())
      return Type();
    std::size_t exp;
    if ($_parser.parseInteger(exp))
      return Type();
    if ($_parser.parseComma())
      return Type();
    std::size_t frac;
    if ($_parser.parseInteger(frac))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, exp, frac);
  }];
}


def Struct : HeteroCL_Type<"Struct"> {
  let summary = "struct type";
  let mnemonic = "struct";
  let parameters = (ins ArrayRefParameter<"Type", "elementTypes">:$elementTypes);
  let printer = [{
    $_printer << "<";
    llvm::interleaveComma(getImpl()->elementTypes, $_printer);
    $_printer << '>';
  }];
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    SmallVector<mlir::Type, 1> elementTypes;
    do {
      mlir::Type elementType;
      if ($_parser.parseType(elementType))
        return nullptr;

      elementTypes.push_back(elementType);
    } while (succeeded($_parser.parseOptionalComma()));

    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, elementTypes);
  }];
}

#endif // HCL_TYPES