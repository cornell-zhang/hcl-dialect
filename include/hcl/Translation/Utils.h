//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

#ifndef HCL_TRANSLATION_UTILS_H
#define HCL_TRANSLATION_UTILS_H

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace hcl;

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class HCLEmitterState {
public:
  explicit HCLEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;
  std::map<std::string, int> nameConflictCnt;

private:
  HCLEmitterState(const HCLEmitterState &) = delete;
  void operator=(const HCLEmitterState &) = delete;
};

/// This is the base class for all of the HLSCpp Emitter components.
class HCLEmitterBase {
public:
  explicit HCLEmitterBase(HCLEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  HCLEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  // TODO: update naming rule.
  SmallString<8> addName(Value val, bool isPtr = false, std::string name = "") {
    assert(!isDeclared(val) && "has been declared before.");

    SmallString<8> valName;
    if (isPtr)
      valName += "*";

    if (name != "") {
      if (state.nameConflictCnt.count(name) > 0) {
        state.nameConflictCnt[name]++;
        valName +=
            StringRef(name + std::to_string(state.nameConflictCnt[name]));
      } else { // first time
        state.nameConflictCnt[name] = 0;
        valName += name;
      }
    } else {
      valName += StringRef("v" + std::to_string(state.nameTable.size()));
    }
    state.nameTable[val] = valName;

    return valName;
  };

  SmallString<8> getName(Value val) {
    // For constant scalar operations, the constant number will be returned
    // rather than the value name.
    if (auto defOp = val.getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        auto constAttr = constOp.getValue();

        if (auto boolAttr = constAttr.dyn_cast<BoolAttr>()) {
          return SmallString<8>(std::to_string(boolAttr.getValue()));

        } else if (auto floatAttr = constAttr.dyn_cast<FloatAttr>()) {
          auto value = floatAttr.getValueAsDouble();
          if (std::isfinite(value))
            return SmallString<8>(std::to_string(value));
          else if (value > 0)
            return SmallString<8>("INFINITY");
          else
            return SmallString<8>("-INFINITY");

        } else if (auto intAttr = constAttr.dyn_cast<IntegerAttr>()) {
          auto value = intAttr.getInt();
          return SmallString<8>(std::to_string(value));
        }
      }
    }
    return state.nameTable.lookup(val);
  };

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    } else
      return true;
  }

private:
  HCLEmitterBase(const HCLEmitterBase &) = delete;
  void operator=(const HCLEmitterBase &) = delete;
};

#endif // HCL_TRANSLATION_UTILS_H