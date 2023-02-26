/*
 * Copyright HeteroCL authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modification: Polymer
 * https://github.com/kumasento/polymer
 */

//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TARGET_OPENSCOP_H
#define HCL_TARGET_OPENSCOP_H

#include <memory>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#define SCOP_STMT_ATTR_NAME "scop.stmt"

namespace mlir {
class OwningModuleRef;
class MLIRContext;
class ModuleOp;
class FuncOp;
struct LogicalResult;
class Operation;
class Value;
} // namespace mlir

namespace mlir {
namespace hcl {

class OslScop;
class OslSymbolTable;

std::unique_ptr<OslScop> createOpenScopFromFuncOp(
        mlir::FuncOp funcOp,
        OslSymbolTable &symTable
        );

/// Create a function (FuncOp) from the given OpenScop object in the given
/// module (ModuleOp).
//mlir::Operation *
//createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, mlir::ModuleOp module,
//                         OslSymbolTable &symTable, mlir::MLIRContext *context,
//                         PlutoProg *prog = nullptr,
//                         const char *dumpClastAfterPluto = nullptr);
//
//mlir::OwningModuleRef translateOpenScopToModule(std::unique_ptr<OslScop> scop,
//                                                mlir::MLIRContext *context);
//
//mlir::LogicalResult translateModuleToOpenScop(
//    mlir::ModuleOp module,
//    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
//    llvm::raw_ostream &os);
//
//void registerToOpenScopTranslation();
//void registerFromOpenScopTranslation();

} // namespace hcl
} // namespace mlir

#endif
