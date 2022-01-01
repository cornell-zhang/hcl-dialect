//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitHLSCpp.h"
#include "hcl/Dialect/Visitor.h"
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

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static SmallString<16> getTypeName(Value val) {
  // Handle memref, tensor, and vector types.
  auto valType = val.getType();
  if (auto arrayType = val.getType().dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle float types.
  if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1)
      return SmallString<16>("bool");
    else {
      std::string signedness = "";
      if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
        signedness = "u";

      switch (intType.getWidth()) {
      case 8:
      case 16:
      case 32:
      case 64:
        return SmallString<16>(signedness + "int" +
                               std::to_string(intType.getWidth()) + "_t");
      default:
        return SmallString<16>("ap_" + signedness + "int<" +
                               std::to_string(intType.getWidth()) + ">");
      }
    }
  } else
    val.getDefiningOp()->emitError("has unsupported type.");

  return SmallString<16>();
}

//===----------------------------------------------------------------------===//
// Some Base Classes
//===----------------------------------------------------------------------===//

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

  /// SCF statement emitters.
  // void emitScfFor(scf::ForOp op);
  // void emitScfIf(scf::IfOp op);
  // void emitScfYield(scf::YieldOp op);

  // /// Affine statement emitters.
  void emitAffineFor(AffineForOp op);
  // void emitAffineIf(AffineIfOp op);
  // void emitAffineParallel(AffineParallelOp op);
  // void emitAffineApply(AffineApplyOp op);
  // template <typename OpType>
  // void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(AffineLoadOp op);
  void emitAffineStore(AffineStoreOp op);
  // void emitAffineYield(AffineYieldOp op);

  // /// Memref-related statement emitters.
  // template <typename OpType> void emitAlloc(OpType op);
  // void emitLoad(memref::LoadOp op);
  // void emitStore(memref::StoreOp op);

  // /// Tensor-related statement emitters.
  // void emitTensorLoad(memref::TensorLoadOp op);
  // void emitTensorStore(memref::TensorStoreOp op);
  // void emitTensorToMemref(memref::BufferCastOp op);
  // void emitDim(memref::DimOp op);
  // void emitRank(RankOp op);

  /// Standard expression emitters.
  void emitBinary(Operation *op, const char *syntax);
  void emitUnary(Operation *op, const char *syntax);

  // /// Special operation emitters.
  // void emitCall(CallOp op);
  // void emitSelect(SelectOp op);
  // void emitConstant(arith::ConstantOp op);
  // template <typename CastOpType> void emitCast(CastOpType op);

  // /// Structure operations emitters.
  // void emitAssign(AssignOp op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false);
  void emitArrayDecl(Value array);
  unsigned emitNestedLoopHead(Value val);
  // void emitNestedLoopTail(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  // void emitLoopDirectives(Operation *op);
  // void emitArrayDirectives(Value memref);
  // void emitFunctionDirectives(FuncOp func, ArrayRef<Value> portList);
  void emitFunction(FuncOp func);
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineEmitter Class
//===----------------------------------------------------------------------===//

namespace {
class AffineExprEmitter : public HCLEmitterBase,
                          public AffineExprVisitor<AffineExprEmitter> {
public:
  using operand_range = Operation::operand_range;
  explicit AffineExprEmitter(HCLEmitterState &state, unsigned numDim,
                             operand_range operands)
      : HCLEmitterBase(state), numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    emitAffineBinary(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    // This is super inefficient.
    os << "(";
    visit(expr.getLHS());
    os << " + ";
    visit(expr.getRHS());
    os << " - 1) / ";
    visit(expr.getRHS());
    os << ")";
  }

  void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }

  void visitDimExpr(AffineDimExpr expr) {
    os << getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    os << getName(operands[numDim + expr.getPosition()]);
  }

  /// Affine expression emitters.
  void emitAffineBinary(AffineBinaryOpExpr expr, const char *syntax) {
    os << "(";
    if (auto constRHS = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
      if ((unsigned)*syntax == (unsigned)*"*" && constRHS.getValue() == -1) {
        os << "-";
        visit(expr.getLHS());
        os << ")";
        return;
      }
      if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() < 0) {
        visit(expr.getLHS());
        os << " - ";
        os << -constRHS.getValue();
        os << ")";
        return;
      }
    }
    if (auto binaryRHS = expr.getRHS().dyn_cast<AffineBinaryOpExpr>()) {
      if (auto constRHS = binaryRHS.getRHS().dyn_cast<AffineConstantExpr>()) {
        if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          os << " - ";
          visit(binaryRHS.getLHS());
          os << ")";
          return;
        }
      }
    }
    visit(expr.getLHS());
    os << " " << syntax << " ";
    visit(expr.getRHS());
    os << ")";
  }

  void emitAffineExpr(AffineExpr expr) { visit(expr); }

private:
  unsigned numDim;
  operand_range operands;
};
} // namespace

//===----------------------------------------------------------------------===//
// StmtVisitor, ExprVisitor, and PragmaVisitor Classes
//===----------------------------------------------------------------------===//

namespace {
class StmtVisitor : public HLSCppVisitorBase<StmtVisitor, bool> {
public:
  StmtVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// SCF statements.
  // bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  // bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  // bool visitOp(scf::ParallelOp op) { return true; };
  // bool visitOp(scf::ReduceOp op) { return true; };
  // bool visitOp(scf::ReduceReturnOp op) { return true; };
  // bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// Affine statements.
  bool visitOp(AffineForOp op) { return emitter.emitAffineFor(op), true; }
  // bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  // bool visitOp(AffineParallelOp op) {
  //   return emitter.emitAffineParallel(op), true;
  // }
  // bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  // bool visitOp(AffineMaxOp op) {
  //   return emitter.emitAffineMaxMin<AffineMaxOp>(op, "max"), true;
  // }
  // bool visitOp(AffineMinOp op) {
  //   return emitter.emitAffineMaxMin<AffineMinOp>(op, "min"), true;
  // }
  bool visitOp(AffineLoadOp op) { return emitter.emitAffineLoad(op), true; }
  bool visitOp(AffineStoreOp op) { return emitter.emitAffineStore(op), true; }
  // bool visitOp(AffineYieldOp op) { return emitter.emitAffineYield(op), true; }

  /// Memref-related statements.
  // bool visitOp(memref::AllocOp op) {
  //   return emitter.emitAlloc<memref::AllocOp>(op), true;
  // }
  // bool visitOp(memref::AllocaOp op) {
  //   return emitter.emitAlloc<memref::AllocaOp>(op), true;
  // }
  // bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  // bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  // bool visitOp(memref::DeallocOp op) { return true; }

  // /// Tensor-related statements.
  // bool visitOp(memref::TensorLoadOp op) {
  //   return emitter.emitTensorLoad(op), true;
  // }
  // bool visitOp(memref::TensorStoreOp op) {
  //   return emitter.emitTensorStore(op), true;
  // }
  // bool visitOp(memref::BufferCastOp op) {
  //   return emitter.emitTensorToMemref(op), true;
  // }
  // bool visitOp(memref::DimOp op) { return emitter.emitDim(op), true; }
  // bool visitOp(RankOp op) { return emitter.emitRank(op), true; }

  // /// HLSCpp operations.
  // bool visitOp(AssignOp op) { return emitter.emitAssign(op), true; }
  // bool visitOp(CastOp op) { return emitter.emitCast<CastOp>(op), true; }
  // bool visitOp(MulOp op) { return emitter.emitBinary(op, "*"), true; }
  // bool visitOp(AddOp op) { return emitter.emitBinary(op, "+"), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSCppVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// Float binary expressions.
  bool visitOp(CmpFOp op);
  bool visitOp(AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(MulFOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(DivFOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(RemFOp op) { return emitter.emitBinary(op, "%"), true; }

  /// Integer binary expressions.
  // bool visitOp(CmpIOp op);
  // bool visitOp(AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  // bool visitOp(SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  // bool visitOp(MulIOp op) { return emitter.emitBinary(op, "*"), true; }
  // bool visitOp(DivSIOp op) { return emitter.emitBinary(op, "/"), true; }
  // bool visitOp(RemSIOp op) { return emitter.emitBinary(op, "%"), true; }
  // bool visitOp(DivUIOp op) { return emitter.emitBinary(op, "/"), true; }
  // bool visitOp(RemUIOp op) { return emitter.emitBinary(op, "%"), true; }
  // bool visitOp(XOrIOp op) { return emitter.emitBinary(op, "^"), true; }
  // bool visitOp(AndIOp op) { return emitter.emitBinary(op, "&"), true; }
  // bool visitOp(OrIOp op) { return emitter.emitBinary(op, "|"), true; }
  // bool visitOp(ShLIOp op) { return emitter.emitBinary(op, "<<"), true; }
  // bool visitOp(ShRSIOp op) { return emitter.emitBinary(op, ">>"), true; }
  // bool visitOp(ShRUIOp op) { return emitter.emitBinary(op, ">>"), true; }

  /// Unary expressions.
  // bool visitOp(math::AbsOp op) { return emitter.emitUnary(op, "abs"), true; }
  // bool visitOp(math::CeilOp op) { return emitter.emitUnary(op, "ceil"), true; }
  // bool visitOp(math::CosOp op) { return emitter.emitUnary(op, "cos"), true; }
  // bool visitOp(math::SinOp op) { return emitter.emitUnary(op, "sin"), true; }
  // bool visitOp(math::TanhOp op) { return emitter.emitUnary(op, "tanh"), true; }
  // bool visitOp(math::SqrtOp op) { return emitter.emitUnary(op, "sqrt"), true; }
  // bool visitOp(math::RsqrtOp op) {
  //   return emitter.emitUnary(op, "1.0 / sqrt"), true;
  // }
  // bool visitOp(math::ExpOp op) { return emitter.emitUnary(op, "exp"), true; }
  // bool visitOp(math::Exp2Op op) { return emitter.emitUnary(op, "exp2"), true; }
  // bool visitOp(math::LogOp op) { return emitter.emitUnary(op, "log"), true; }
  // bool visitOp(math::Log2Op op) { return emitter.emitUnary(op, "log2"), true; }
  // bool visitOp(math::Log10Op op) {
  //   return emitter.emitUnary(op, "log10"), true;
  // }
  // bool visitOp(NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  // /// Special operations.
  // bool visitOp(CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(ReturnOp op) { return true; }
  // bool visitOp(SelectOp op) { return emitter.emitSelect(op), true; }
  // bool visitOp(ConstantOp op) { return emitter.emitConstant(op), true; }
  // bool visitOp(IndexCastOp op) {
  //   return emitter.emitCast<IndexCastOp>(op), true;
  // }
  // bool visitOp(UIToFPOp op) {
  //   return emitter.emitCast<UIToFPOp>(op), true;
  // }
  // bool visitOp(SIToFPOp op) {
  //   return emitter.emitCast<SIToFPOp>(op), true;
  // }
  // bool visitOp(FPToUIOp op) {
  //   return emitter.emitCast<FPToUIOp>(op), true;
  // }
  // bool visitOp(FPToSIOp op) {
  //   return emitter.emitCast<FPToSIOp>(op), true;
  // }

private:
  ModuleEmitter &emitter;
};
} // namespace

bool ExprVisitor::visitOp(CmpFOp op) {
  switch (op.getPredicate()) {
  case CmpFPredicate::OEQ:
  case CmpFPredicate::UEQ:
    return emitter.emitBinary(op, "=="), true;
  case CmpFPredicate::ONE:
  case CmpFPredicate::UNE:
    return emitter.emitBinary(op, "!="), true;
  case CmpFPredicate::OLT:
  case CmpFPredicate::ULT:
    return emitter.emitBinary(op, "<"), true;
  case CmpFPredicate::OLE:
  case CmpFPredicate::ULE:
    return emitter.emitBinary(op, "<="), true;
  case CmpFPredicate::OGT:
  case CmpFPredicate::UGT:
    return emitter.emitBinary(op, ">"), true;
  case CmpFPredicate::OGE:
  case CmpFPredicate::UGE:
    return emitter.emitBinary(op, ">="), true;
  default:
    op.emitError("has unsupported compare type.");
    return false;
  }
}

// bool ExprVisitor::visitOp(CmpIOp op) {
//   switch (op.getPredicate()) {
//   case CmpIPredicate::eq:
//     return emitter.emitBinary(op, "=="), true;
//   case CmpIPredicate::ne:
//     return emitter.emitBinary(op, "!="), true;
//   case CmpIPredicate::slt:
//   case CmpIPredicate::ult:
//     return emitter.emitBinary(op, "<"), true;
//   case CmpIPredicate::sle:
//   case CmpIPredicate::ule:
//     return emitter.emitBinary(op, "<="), true;
//   case CmpIPredicate::sgt:
//   case CmpIPredicate::ugt:
//     return emitter.emitBinary(op, ">"), true;
//   case CmpIPredicate::sge:
//   case CmpIPredicate::uge:
//     return emitter.emitBinary(op, ">="), true;
//   }
// }

/// Affine statement emitters.
void ModuleEmitter::emitAffineFor(AffineForOp op) {
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  auto lowerMap = op.getLowerBoundMap();
  AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                 op.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      os << "max(";
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
      os << ", ";
      lowerEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  auto upperMap = op.getUpperBoundMap();
  AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                 op.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      os << "min(";
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
      os << ", ";
      upperEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  // emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getTypeName(val) << " ";

  // Add the new value to nameTable and emit its name.
  os << addName(val, isPtr);
  for (unsigned i = 0; i < rank; ++i)
    os << "[iv" << i << "]";
}


void ModuleEmitter::emitArrayDecl(Value array) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    emitValue(array);
    for (auto &shape : arrayType.getShape())
      os << "[" << shape << "]";
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true);
}

/// Standard expression emitters.
void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = ";
  emitValue(op->getOperand(0), rank);
  os << " " << syntax << " ";
  emitValue(op->getOperand(1), rank);
  os << ";";
  emitInfoAndNewLine(op);
  // emitNestedLoopTail(rank);
}

void ModuleEmitter::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  emitInfoAndNewLine(op);
  // emitNestedLoopTail(rank);
}

unsigned ModuleEmitter::emitNestedLoopHead(Value val) {
  unsigned rank = 0;

  if (auto type = val.getType().dyn_cast<ShapedType>()) {
    if (!type.hasStaticShape()) {
      emitError(val.getDefiningOp(), "is unranked or has dynamic shape.");
      return 0;
    }

    // Declare a new array.
    if (!isDeclared(val)) {
      indent();
      emitArrayDecl(val);
      os << ";\n";
    }

    // Create nested loop.
    unsigned dimIdx = 0;
    for (auto &shape : type.getShape()) {
      indent();
      os << "for (int iv" << dimIdx << " = 0; ";
      os << "iv" << dimIdx << " < " << shape << "; ";
      os << "++iv" << dimIdx++ << ") {\n";

      addIndent();
    }
    rank = type.getRank();
  }

  return rank;
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();

  // // Print schedule information.
  // if (auto timing = getTiming(op))
  //   os << ", [" << timing.getBegin() << "," << timing.getEnd() << ")";

  // // Print loop information.
  // if (auto loopInfo = getLoopInfo(op))
  //   os << ", iterCycle=" << loopInfo.getIterLatency()
  //      << ", II=" << loopInfo.getMinII();

  os << "\n";
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}

void ModuleEmitter::emitFunction(FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  // if (auto funcDirect = getFuncDirective(func))
  //   if (funcDirect.getTopFunc())
      os << "/// This is top function.\n";

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  for (auto &arg : func.getArguments()) {
    indent();
    if (arg.getType().isa<ShapedType>())
      emitArrayDecl(arg);
    else
      emitValue(arg);

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  // if (auto funcReturn = dyn_cast<ReturnOp>(func.front().getTerminator())) {
  //   for (auto result : funcReturn.getOperands()) {
  //     os << ",\n";
  //     indent();
  //     // TODO: a known bug, cannot return a value twice, e.g. return %0, %0 :
  //     // index, index. However, typically this should not happen.
  //     if (result.getType().isa<ShapedType>())
  //       emitArrayDecl(result);
  //     else
  //       // In Vivado HLS, pointer indicates the value is an output.
  //       emitValue(result, /*rank=*/0, /*isPtr=*/true);

  //     portList.push_back(result);
  //   }
  // } else
  //   emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  // emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  // emitFunctionDirectives(func, portList);
  emitBlock(func.front());
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

  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<FuncOp>(op))
      emitFunction(func);
    else
      emitError(&op, "is unsupported operation.");
  }
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
        registry.insert<mlir::hcl::HeteroCLDialect, mlir::StandardOpsDialect,
                        tensor::TensorDialect, mlir::AffineDialect,
                        mlir::math::MathDialect, mlir::memref::MemRefDialect>();
      });
}