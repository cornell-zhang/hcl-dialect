//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HCLATTRS_H
#define HCLATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "hcl/Dialect/HeteroCLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hcl/Dialect/HeteroCLAttrs.h.inc"

#endif // HCLATTRS_H