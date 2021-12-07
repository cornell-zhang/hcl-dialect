//===- HeteroCLDialect.td - hcl dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

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
    let cppNamespace = "::mlir::hcl";
}

//===----------------------------------------------------------------------===//
// Base hcl operation definition.
//===----------------------------------------------------------------------===//

class HeteroCL_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<HeteroCL_Dialect, mnemonic, traits>;

class HeteroCL_Type<string name, list<Trait> traits = []> : 
        TypeDef<HeteroCL_Dialect, name, traits>;

#endif // STANDALONE_DIALECT
