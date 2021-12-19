//===- HeteroCLAttrs.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_ATTRS
#define HCL_ATTRS

include "HeteroCLDialect.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
def CompletePartition: I32EnumAttrCase<"CompletePartition", 0>;
def BlockPartition: I32EnumAttrCase<"BlockPartition", 1>;
def CyclicPartition: I32EnumAttrCase<"CyclicPartition", 2>;

def PartitionKindEnum: I32EnumAttr<"PartitionKindEnum",
        "An example partition enum",
        [CompletePartition,BlockPartition,CyclicPartition]> {
    let cppNamespace = "mlir::hcl";
    let stringToSymbolFnName = "ConvertToEnum";
    let symbolToStringFnName = "ConvertToString";
}

def LoopAttr : HeteroCL_Attr<"LoopAttr"> {
  let mnemonic = "LoopAttr";
  let parameters = (ins "hcl::LoopHandleType":$loopName);
  let printer = [{
    StringRef loopName = getImpl()->loopName.getLoopName();
    $_printer << "Loop <" << loopName.str() << ">";
  }];
  let parser = [{
    if (parser.parseLess())
      return Attribute();
    StringRef loopName;
    if ($_parser.parseOptionalString(&loopName))
      return Attribute();
    if ($_parser.parseGreater())
      return Attribute();
    return get($_ctxt, hcl::LoopHandleType::get($_ctxt, loopName));
  }];
}

#endif // HCL_ATTRS