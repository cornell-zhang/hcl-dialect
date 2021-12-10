//===- HeteroCLDialect.cpp - hcl dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"

using namespace mlir;
using namespace mlir::hcl;

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"
#include "hcl/HeteroCLOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "hcl/HeteroCLOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// hcl dialect.
//===----------------------------------------------------------------------===//

mlir::Type mlir::hcl::HeteroCLDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef ref;
    if (parser.parseKeyword(&ref))
    {
        return {};
    }
    mlir::Type genType;
    generatedTypeParser(getContext(), parser, ref, genType);
    return genType;
}

void mlir::hcl::HeteroCLDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
  auto res = generatedTypePrinter(type, printer);
}

void HeteroCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hcl/HeteroCLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hcl/HeteroCLOpsTypes.cpp.inc"
      >();
}
