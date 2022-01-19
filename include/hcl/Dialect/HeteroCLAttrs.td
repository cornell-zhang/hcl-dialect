//===- HeteroCLAttrs.td - hcl dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HCL_ATTRS
#define HCL_ATTRS

include "hcl/Dialect/HeteroCLDialect.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
def CompletePartition: I32EnumAttrCase<"CompletePartition", 0>;
def BlockPartition: I32EnumAttrCase<"BlockPartition", 1>;
def CyclicPartition: I32EnumAttrCase<"CyclicPartition", 2>;

def PartitionKindEnum: I32EnumAttr<"PartitionKindEnum",
        "An example partition enum",
        [CompletePartition,BlockPartition,CyclicPartition]> {
    let cppNamespace = "mlir::hcl";
    let stringToSymbolFnName = "ConvertToPartitionEnum";
    let symbolToStringFnName = "ConvertToPartitionString";
}

def CPUDevice: I32EnumAttrCase<"CPUDevice", 0>;
def FPGADevice: I32EnumAttrCase<"FPGADevice", 1>;

def DeviceEnum: I32EnumAttr<"DeviceEnum",
        "A device enum",
        [CPUDevice, FPGADevice]> {
    let cppNamespace = "mlir::hcl";
    let stringToSymbolFnName = "ConvertToDeviceEnum";
    let symbolToStringFnName = "ConvertToDeviceString";
}

def FooAttr : HeteroCL_Attr<"FooAttr"> {
  let mnemonic = "FooAttr";
  let parameters = (ins );
  let printer = [{
    $_printer << "FooAttr";
  }];
  let parser = [{
    return get($_ctxt);
  }];
}

#endif // HCL_ATTRS