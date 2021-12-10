//===- HeteroCLDialect.h - hcl dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HETEROCL_DIALECT_H
#define HETEROCL_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "hcl/HeteroCLOpsDialect.h.inc"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#define GET_TYPEDEF_CLASSES
#include "hcl/HeteroCLOpsTypes.h.inc"
#endif // HETEROCL_DIALECT_H
