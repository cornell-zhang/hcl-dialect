//===- HeteroCLDialect.td - hcl dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HETEROCL_DIALECT
#define HETEROCL_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// hcl dialect definition.
//===----------------------------------------------------------------------===//

def HeteroCL_Dialect : Dialect {
    let name = "hcl";
    let summary = "A hcl out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let useDefaultTypePrinterParser = 1;
    let cppNamespace = "::mlir::hcl";
}

//===----------------------------------------------------------------------===//
// Base hcl operation definition.
//===----------------------------------------------------------------------===//

class HeteroCL_Op<string mnemonic, list<Trait> traits = []> :
        Op<HeteroCL_Dialect, mnemonic, traits>;

class HeteroCL_Type<string mnemonic, list<Trait> traits = []> : 
        TypeDef<HeteroCL_Dialect, mnemonic, traits>;

class HeteroCL_Attr<string mnemonic, list<Trait> traits = []> : 
        AttrDef<HeteroCL_Dialect, mnemonic, traits>;

#endif // HETEROCL_DIALECT
