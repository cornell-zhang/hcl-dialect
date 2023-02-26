/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: Polymer
 * https://github.com/kumasento/polymer
 */

//===- OslSymbolTable.h -----------------------------------------*- C++ -*-===//
//
// This file declares the OslSymbolTable class that stores the mapping between
// symbols and MLIR values.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_SUPPORT_OSLSYMBOLTABLE_H
#define HCL_SUPPORT_OSLSYMBOLTABLE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace mlir {
namespace hcl {

class OslScopStmtOpSet;

class OslSymbolTable {
public:
  using OpSet = OslScopStmtOpSet;
  using OpSetPtr = std::unique_ptr<OpSet>;

  enum SymbolType { LoopIV, Memref, StmtOpSet };

  Value getValue(StringRef key);

  OpSet getOpSet(StringRef key);

  void setValue(StringRef key, Value val, SymbolType type);

  void setOpSet(StringRef key, OpSet val, SymbolType type);

  unsigned getNumValues(SymbolType type);

  unsigned getNumOpSets(SymbolType type);

  void getValueSymbols(SmallVectorImpl<StringRef> &symbols);

  void getOpSetSymbols(SmallVectorImpl<StringRef> &symbols);

private:
  StringMap<OpSet> nameToStmtOpSet;
  StringMap<Value> nameToLoopIV;
  StringMap<Value> nameToMemref;
};

} // namespace hcl
} // namespace mlir

#endif
