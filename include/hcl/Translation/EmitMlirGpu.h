//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSLATION_EMITMLIRGPU_H
#define HCL_TRANSLATION_EMITMLIRGPU_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
  namespace hcl {

    LogicalResult emitMlirGpu(ModuleOp module, llvm::raw_ostream &os);
    void registerEmitMlirGpuTranslation();

  } // namespace hcl
} // namespace mlir

#endif // HCL_TRANSLATION_EMITMLIRGPU_H