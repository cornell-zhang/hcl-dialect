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

def CMP_FIXED_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMP_FIXED_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMP_FIXED_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMP_FIXED_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMP_FIXED_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMP_FIXED_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMP_FIXED_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMP_FIXED_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMP_FIXED_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMP_FIXED_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpFixedPredicateAttr : I64EnumAttr<
    "CmpFixedPredicate", "",
    [CMP_FIXED_P_EQ, CMP_FIXED_P_NE, CMP_FIXED_P_SLT, CMP_FIXED_P_SLE, CMP_FIXED_P_SGT,
     CMP_FIXED_P_SGE, CMP_FIXED_P_ULT, CMP_FIXED_P_ULE, CMP_FIXED_P_UGT, CMP_FIXED_P_UGE]> {
  let cppNamespace = "mlir::hcl";
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