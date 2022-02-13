//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl-c/Dialect/HCLAttributes.h"
#include "hcl/Dialect/HeteroCLDialect.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace hcl;

// bool mlirAttributeIsAIntegerSet(MlirAttribute attr) {
//   return unwrap(attr).isa<IntegerSetAttr>();
// }

MlirAttribute mlirIntegerSetAttrGet(MlirIntegerSet set) {
  return wrap(IntegerSetAttr::get(unwrap(set)));
}