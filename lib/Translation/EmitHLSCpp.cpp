//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitHLSCpp.h"
// #include "hcl/Dialect/Visitor.h"
// #include "hcl/Dialect/InitAllDialects.h"
// #include "hcl/Support/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "hcl/Dialect/HeteroCLDialect.h"

using namespace mlir;
using namespace hcl;

namespace {
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

private:
  HCLEmitterState(const HCLEmitterState &) = delete;
  void operator=(const HCLEmitterState &) = delete;
};
} // namespace

namespace {
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
  SmallString<8> addName(Value val, bool isPtr = false);

  SmallString<8> addAlias(Value val, Value alias);

  SmallString<8> getName(Value val);

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
} // namespace

// TODO: update naming rule.
SmallString<8> HCLEmitterBase::addName(Value val, bool isPtr) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  valName += StringRef("v" + std::to_string(state.nameTable.size()));
  state.nameTable[val] = valName;

  return valName;
}

SmallString<8> HCLEmitterBase::addAlias(Value val, Value alias) {
  assert(!isDeclared(alias) && "has been declared before.");
  assert(isDeclared(val) && "hasn't been declared before.");

  auto valName = getName(val);
  state.nameTable[alias] = valName;

  return valName;
}

SmallString<8> HCLEmitterBase::getName(Value val) {
  // For constant scalar operations, the constant number will be returned rather
  // than the value name.
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<ConstantOp>(defOp)) {
      auto constAttr = constOp.value();

      if (auto floatAttr = constAttr.dyn_cast<FloatAttr>()) {
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

      } else if (auto boolAttr = constAttr.dyn_cast<BoolAttr>())
        return SmallString<8>(std::to_string(boolAttr.getValue()));
    }
  }
  return state.nameTable.lookup(val);
}

namespace {
class ModuleEmitter : public HCLEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HCLEmitterState &state)
      : HCLEmitterBase(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);
  void emitFunction(FuncOp func);
};
} // namespace

void ModuleEmitter::emitFunction(FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  // if (auto funcDirect = getFuncDirective(func))
  //   if (funcDirect.getTopFunc())
  //     os << "/// This is top function.\n";

  // if (auto timing = getTiming(func)) {
  //   os << "/// Latency=" << timing.getLatency();
  //   os << ", interval=" << timing.getInterval();
  //   os << "\n";
  // }

  // if (auto resource = getResource(func)) {
  //   os << "/// DSP=" << resource.getDsp();
  //   // os << ", BRAM=" << resource.getBram();
  //   // os << ", LUT=" << resource.getLut();
  //   os << "\n";
  // }

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  for (auto &arg : func.getArguments()) {
    indent();
    // if (arg.getType().isa<ShapedType>())
    // emitArrayDecl(arg);
    // else
    // emitValue(arg);

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  if (auto funcReturn = dyn_cast<ReturnOp>(func.front().getTerminator())) {
    for (auto result : funcReturn.getOperands()) {
      os << ",\n";
      indent();
      // TODO: a known bug, cannot return a value twice, e.g. return %0, %0 :
      // index, index. However, typically this should not happen.
      // if (result.getType().isa<ShapedType>())
      //   emitArrayDecl(result);
      // else
      //   // In Vivado HLS, pointer indicates the value is an output.
      //   emitValue(result, /*rank=*/0, /*isPtr=*/true);

      portList.push_back(result);
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  // emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  // emitFunctionDirectives(func, portList);
  // emitBlock(func.front());
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  os << R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  // for (auto &op : *module.getBody()) {
  //   if (auto func = dyn_cast<FuncOp>(op))
  //     emitFunction(func);
  //   else
  //     emitError(&op, "is unsupported operation.");
  // }
}

//===----------------------------------------------------------------------===//
// Entry of hcl-translate
//===----------------------------------------------------------------------===//

LogicalResult hcl::emitHLSCpp(ModuleOp module, llvm::raw_ostream &os) {
  HCLEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void hcl::registerEmitHLSCppTranslation() {
  static TranslateFromMLIRRegistration toHLSCpp(
      "emit-hlscpp", emitHLSCpp, [&](DialectRegistry &registry) {
        // registerAllDialects(registry);
        registry.insert<mlir::hcl::HeteroCLDialect, mlir::StandardOpsDialect,
                        tensor::TensorDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect>();
      });
}