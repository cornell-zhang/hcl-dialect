//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the Polymer project [https://github.com/kumasento/polymer]
//
//===----------------------------------------------------------------------===//

//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TARGET_OPENSCOP_H
#define HCL_TARGET_OPENSCOP_H

#include <memory>
#include <iostream>

#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#define SCOP_STMT_ATTR_NAME "scop.stmt"

namespace mlir {
namespace hcl {

class OslScop;
class OslSymbolTable;

std::unique_ptr<OslScop> createOpenScopFromFuncOp(
        FuncOp funcOp,
        OslSymbolTable &symTable
        );

} // namespace hcl
} // namespace mlir

#endif // HCL_TARGET_OPENSCOP_H
