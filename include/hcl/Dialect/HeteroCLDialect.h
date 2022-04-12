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

#include "hcl/Dialect/HeteroCLDialect.h.inc"

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
  ArrayRef<mlir::Type> getElementTypes() const;

  /// Return the number of elements in this struct type.
  unsigned getNumElements() const { return getElementTypes().size(); };
};
} // namespace hcl
} // namespace mlir

#endif // HETEROCL_DIALECT_H
