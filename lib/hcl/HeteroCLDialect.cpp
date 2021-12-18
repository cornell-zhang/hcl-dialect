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

#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLTypes.h"

#include "hcl/HeteroCLAttrs.h"
#include "hcl/HeteroCLOps.h"

using namespace mlir;
using namespace mlir::hcl;

#include "hcl/HeteroCLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "hcl/HeteroCLTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hcl/HeteroCLAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
void HeteroCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hcl/HeteroCLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hcl/HeteroCLTypes.cpp.inc"
      >();
  addAttributes< // test/lib/Dialect/Test/TestAttributes.cpp
#define GET_ATTRDEF_LIST
#include "hcl/HeteroCLAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type-related Dialect methods.
//===----------------------------------------------------------------------===//

mlir::Type
mlir::hcl::HeteroCLDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  Type type;
  OptionalParseResult parseResult =
      generatedTypeParser(getContext(), parser, keyword, type);
  if (parseResult.hasValue())
    return type;

  parser.emitError(parser.getNameLoc(), "invalid 'hcl' type: `")
      << keyword << "'";
  return Type();
}

void mlir::hcl::HeteroCLDialect::printType(
    mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  if (failed(generatedTypePrinter(type, printer)))
    llvm_unreachable("unknown 'hcl' type");
}

Attribute HeteroCLDialect::parseAttribute(DialectAsmParser &parser,
                                          Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  {
    Attribute attr;
    auto parseResult =
        generatedAttributeParser(getContext(), parser, attrTag, type, attr);
    if (parseResult.hasValue())
      return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown hcl attribute");
  return Attribute();
}

void HeteroCLDialect::printAttribute(Attribute attr,
                                     DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}