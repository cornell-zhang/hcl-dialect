//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HCLTYPES_H
#define HCLTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "hcl/Dialect/HeteroCLTypes.h.inc"

namespace mlir {
namespace hcl {
namespace detail {
struct StructTypeStorage;
} // namespace detail
} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  using Base::Base;

  // Create an instance of a `StructType` with the given element types.
  // There must be at least one element type.
  static StructType get(ArrayRef<mlir::Type> elementTypes);

  /// Return the element types of this struct type.
  ArrayRef<mlir::Type> getElementTypes();

  /// Return the number of elements in this struct type.
  unsigned getNumElements() { return getElementTypes().size(); };

  /// Parser
  mlir::Type parse(mlir::DialectAsmParser &parser);

  /// Printer
  void print(mlir::DialectAsmPrinter &printer);
};
} // namespace hcl
} // namespace mlir

#endif // HCLTYPES_H