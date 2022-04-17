//===- HeteroCLDialect.cpp - hcl dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLTypes.h"

#include "hcl/Dialect/HeteroCLAttrs.h"
#include "hcl/Dialect/HeteroCLOps.h"

using namespace mlir;
using namespace mlir::hcl;

#include "hcl/Dialect/HeteroCLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "hcl/Dialect/HeteroCLTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hcl/Dialect/HeteroCLAttrs.cpp.inc"

#include "hcl/Dialect/HeteroCLEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
void HeteroCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hcl/Dialect/HeteroCLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hcl/Dialect/HeteroCLTypes.cpp.inc"
      >();
  addAttributes< // test/lib/Dialect/Test/TestAttributes.cpp
#define GET_ATTRDEF_LIST
#include "hcl/Dialect/HeteroCLAttrs.cpp.inc"
      >();
}