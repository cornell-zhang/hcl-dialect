//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the Polymer project
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

//DEBJIT#include "pluto/internal/pluto.h"

#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace hcl {

class OslScop;

LogicalResult translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os);

LogicalResult emitOpenScop(
    ModuleOp module, 
    llvm::raw_ostream &os);

void registerToOpenScopTranslation();

} // namespace hcl
} // namespace mlir

#endif // HCL_TARGET_OPENSCOP_H
