//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitVivadoHLS.h"
#include "hcl-c/Translation/EmitVivadoHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace hcl;

MlirLogicalResult mlirEmitVivadoHls(MlirModule module, MlirStringCallback callback,
                                 void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitVivadoHLS(unwrap(module), stream));
}