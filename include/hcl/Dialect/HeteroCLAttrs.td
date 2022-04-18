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

// NDRange dimension binding
def BlockIdxX: I32EnumAttrCase<"BlockIdxX", 0>;
def BlockIdxY: I32EnumAttrCase<"BlockIdxY", 1>;
def BlockIdxZ: I32EnumAttrCase<"BlockIdxZ", 2>;
def ThreadIdxX: I32EnumAttrCase<"ThreadIdxX", 3>;
def ThreadIdxY: I32EnumAttrCase<"ThreadIdxY", 4>;
def ThreadIdxZ: I32EnumAttrCase<"ThreadIdxZ", 5>;

def NDRangeDimKindEnum: I32EnumAttr<"NDRangeDimKindEnum",
        "A NDRange dimension binding enum",
        [BlockIdxX,BlockIdxY,BlockIdxZ,ThreadIdxX,ThreadIdxY,ThreadIdxZ]> {
    let cppNamespace = "mlir::hcl";
    let stringToSymbolFnName = "ConvertToNDRangeDimEnum";
    let symbolToStringFnName = "ConvertToNDRangeDimString";
}

def CPUDevice: I32EnumAttrCase<"CPUDevice", 0>;
def FPGADevice: I32EnumAttrCase<"FPGADevice", 1>;
def GPUDevice: I32EnumAttrCase<"GPUDevice", 2>;

def DeviceEnum: I32EnumAttr<"DeviceEnum",
        "A device enum",
        [CPUDevice, FPGADevice, GPUDevice]> {
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

def CMP_BFLOAT_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMP_BFLOAT_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMP_BFLOAT_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMP_BFLOAT_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMP_BFLOAT_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMP_BFLOAT_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMP_BFLOAT_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMP_BFLOAT_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMP_BFLOAT_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMP_BFLOAT_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpBfloatPredicateAttr : I64EnumAttr<
    "CmpBfloatPredicate", "",
    [CMP_BFLOAT_P_EQ, CMP_BFLOAT_P_NE, CMP_BFLOAT_P_SLT, CMP_BFLOAT_P_SLE, CMP_BFLOAT_P_SGT,
     CMP_BFLOAT_P_SGE, CMP_BFLOAT_P_ULT, CMP_BFLOAT_P_ULE, CMP_BFLOAT_P_UGT, CMP_BFLOAT_P_UGE]> {
  let cppNamespace = "mlir::hcl";
}


#endif // HCL_ATTRS