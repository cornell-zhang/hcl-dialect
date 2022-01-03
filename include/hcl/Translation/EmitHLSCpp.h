//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSLATION_EMITHLSCPP_H
#define HCL_TRANSLATION_EMITHLSCPP_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hcl {

LogicalResult emitHLSCpp(ModuleOp module, llvm::raw_ostream &os);
void registerEmitHLSCppTranslation();

} // namespace hcl
} // namespace mlir

#endif // HCL_TRANSLATION_EMITHLSCPP_H