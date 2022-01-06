//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/EmitHLSCpp.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "hcl/Translation/EmitHLSCpp.h"

using namespace mlir;
using namespace hcl;

MlirLogicalResult mlirEmitHlsCpp(MlirModule module, MlirStringCallback callback,
                                 void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitHLSCpp(unwrap(module), stream));
}