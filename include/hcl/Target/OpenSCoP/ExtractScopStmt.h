//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the Polymer project [https://github.com/kumasento/polymer]
//
//===----------------------------------------------------------------------===//

//===- ExtractScopStmt.h - Extract scop stmt to func ------------------C++-===//
//
// This file declares the transformation that extracts scop statements into MLIR
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TARGET_EXTRACTSCOPSTMT_H
#define HCL_TARGET_EXTRACTSCOPSTMT_H

#include "mlir/IR/BuiltinOps.h"

#define SCOP_STMT_ATTR_NAME "scop.stmt"

namespace mlir {
class ModuleOp;
struct LogicalResult;
} // namespace mlir

namespace mlir {
namespace hcl {

LogicalResult extractOpenScop(
    ModuleOp module,
    llvm::raw_ostream &os);

void registerToOpenScopExtractTranslation();

} // namespace hcl
} // namespace mlir

#endif
