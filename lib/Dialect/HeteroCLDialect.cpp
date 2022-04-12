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

//===----------------------------------------------------------------------===//
// HeteroCL Types.
//===----------------------------------------------------------------------===//
namespace mlir {
namespace hcl {
namespace detail {
struct StructTypeStorage : public TypeStorage {

  /// The element types of this struct type.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Comparison function
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Hash function for the key type
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};

} // namespace detail
} // namespace hcl
} // namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

/// Parse a struct type from the given string.
mlir::Type StructType::parse(mlir::DialectAsmParser &parser) {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    // SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

void StructType::print(mlir::DialectAsmPrinter &printer) {
  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(this->getElementTypes(), printer);
  printer << '>';
}
