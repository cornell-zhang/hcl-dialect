//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitHLSCpp.h"
#include "hcl/Dialect/Visitor.h"
// #include "hcl/Dialect/InitAllDialects.h"
#include "hcl/Support/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

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
  }

  // Handle (custom) fixed point types.
  else if (auto fixedType = valType.dyn_cast<hcl::FixedType>())
    return SmallString<16>(
        "ap_fixed<" + std::to_string(fixedType.getWidth()) + "," +
        std::to_string(fixedType.getWidth() - fixedType.getFrac()) + ">");

  else if (auto ufixedType = valType.dyn_cast<hcl::UFixedType>())
    return SmallString<16>(
        "ap_ufixed<" + std::to_string(ufixedType.getWidth()) + "," +
        std::to_string(ufixedType.getWidth() - ufixedType.getFrac()) + ">");
  else
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
  SmallString<8> addName(Value val, bool isPtr = false, std::string name = "");

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
SmallString<8> HCLEmitterBase::addName(Value val, bool isPtr,
                                       std::string name) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  if (name != "") {
    valName += StringRef(name + std::to_string(state.nameTable.size()));
  } else {
    valName += StringRef("v" + std::to_string(state.nameTable.size()));
  }
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

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public HCLEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HCLEmitterState &state) : HCLEmitterBase(state) {}

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  void emitAffineFor(AffineForOp op);
  void emitAffineIf(AffineIfOp op);
  void emitAffineParallel(AffineParallelOp op);
  void emitAffineApply(AffineApplyOp op);
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(AffineLoadOp op);
  void emitAffineStore(AffineStoreOp op);
  void emitAffineYield(AffineYieldOp op);

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);

  /// Tensor-related statement emitters.
  void emitTensorExtract(tensor::ExtractOp op);
  void emitTensorInsert(tensor::InsertOp op);
  void emitTensorLoad(memref::TensorLoadOp op);
  void emitTensorStore(memref::TensorStoreOp op);
  void emitTensorToMemref(memref::BufferCastOp op);
  void emitDim(memref::DimOp op);
  void emitRank(RankOp op);

  /// Standard expression emitters.
  void emitBinary(Operation *op, const char *syntax);
  void emitUnary(Operation *op, const char *syntax);

  /// Special operation emitters.
  void emitCall(CallOp op);
  void emitSelect(SelectOp op);
  void emitConstant(ConstantOp op);
  template <typename CastOpType> void emitCast(CastOpType op);
  void emitGeneralCast(UnrealizedConversionCastOp op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  unsigned emitNestedLoopHead(Value val);
  void emitNestedLoopTail(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  void emitLoopDirectives(Operation *op);
  void emitArrayDirectives(Value memref);
  void emitFunctionDirectives(FuncOp func, ArrayRef<Value> portList);
  void emitFunction(FuncOp func);
  void emitHostFunction(FuncOp func);
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
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  bool visitOp(scf::ParallelOp op) { return true; };
  bool visitOp(scf::ReduceOp op) { return true; };
  bool visitOp(scf::ReduceReturnOp op) { return true; };
  bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// Affine statements.
  bool visitOp(AffineForOp op) { return emitter.emitAffineFor(op), true; }
  bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  bool visitOp(AffineParallelOp op) {
    return emitter.emitAffineParallel(op), true;
  }
  bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  bool visitOp(AffineMaxOp op) {
    return emitter.emitAffineMaxMin<AffineMaxOp>(op, "max"), true;
  }
  bool visitOp(AffineMinOp op) {
    return emitter.emitAffineMaxMin<AffineMinOp>(op, "min"), true;
  }
  bool visitOp(AffineLoadOp op) { return emitter.emitAffineLoad(op), true; }
  bool visitOp(AffineStoreOp op) { return emitter.emitAffineStore(op), true; }
  bool visitOp(AffineYieldOp op) { return emitter.emitAffineYield(op), true; }

  /// Memref-related statements.
  bool visitOp(memref::AllocOp op) {
    return emitter.emitAlloc<memref::AllocOp>(op), true;
  }
  bool visitOp(memref::AllocaOp op) {
    return emitter.emitAlloc<memref::AllocaOp>(op), true;
  }
  bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  bool visitOp(memref::DeallocOp op) { return true; }

  /// Tensor-related statements.
  bool visitOp(tensor::ExtractOp op) {
    return emitter.emitTensorExtract(op), true;
  }
  bool visitOp(tensor::InsertOp op) {
    return emitter.emitTensorInsert(op), true;
  }
  bool visitOp(memref::TensorLoadOp op) {
    return emitter.emitTensorLoad(op), true;
  }
  bool visitOp(memref::TensorStoreOp op) {
    return emitter.emitTensorStore(op), true;
  }
  bool visitOp(memref::BufferCastOp op) {
    return emitter.emitTensorToMemref(op), true;
  }
  bool visitOp(memref::DimOp op) { return emitter.emitDim(op), true; }
  bool visitOp(RankOp op) { return emitter.emitRank(op), true; }

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
  bool visitOp(SignedDivIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(SignedFloorDivIOp op) {
    return emitter.emitBinary(op, "/"), true;
  }
  bool visitOp(RemFOp op) { return emitter.emitBinary(op, "%"), true; }

  /// Integer binary expressions.
  bool visitOp(CmpIOp op);
  bool visitOp(AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(MulIOp op) { return emitter.emitBinary(op, "*"), true; }

  /// Unary expressions.
  bool visitOp(math::CosOp op) { return emitter.emitUnary(op, "cos"), true; }
  bool visitOp(math::SinOp op) { return emitter.emitUnary(op, "sin"), true; }
  bool visitOp(math::TanhOp op) { return emitter.emitUnary(op, "tanh"), true; }
  bool visitOp(math::SqrtOp op) { return emitter.emitUnary(op, "sqrt"), true; }
  bool visitOp(math::RsqrtOp op) {
    return emitter.emitUnary(op, "1.0 / sqrt"), true;
  }
  bool visitOp(math::ExpOp op) { return emitter.emitUnary(op, "exp"), true; }
  bool visitOp(math::Exp2Op op) { return emitter.emitUnary(op, "exp2"), true; }
  bool visitOp(math::LogOp op) { return emitter.emitUnary(op, "log"), true; }
  bool visitOp(math::Log2Op op) { return emitter.emitUnary(op, "log2"), true; }
  bool visitOp(math::Log10Op op) {
    return emitter.emitUnary(op, "log10"), true;
  }
  bool visitOp(NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  /// Special operations.
  bool visitOp(CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(ReturnOp op) { return true; }
  bool visitOp(SelectOp op) { return emitter.emitSelect(op), true; }
  bool visitOp(ConstantOp op) { return emitter.emitConstant(op), true; }
  bool visitOp(IndexCastOp op) {
    return emitter.emitCast<IndexCastOp>(op), true;
  }
  bool visitOp(UIToFPOp op) { return emitter.emitCast<UIToFPOp>(op), true; }
  bool visitOp(SIToFPOp op) { return emitter.emitCast<SIToFPOp>(op), true; }
  bool visitOp(FPToUIOp op) { return emitter.emitCast<FPToUIOp>(op), true; }
  bool visitOp(FPToSIOp op) { return emitter.emitCast<FPToSIOp>(op), true; }
  bool visitOp(UnrealizedConversionCastOp op) {
    return emitter.emitGeneralCast(op), true;
  }

  /// HCL operations.
  bool visitOp(hcl::CreateLoopHandleOp op) { return true; }
  bool visitOp(hcl::CreateStageHandleOp op) { return true; }

  /// Fixed points
  bool visitOp(hcl::AddFixedOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(hcl::SubFixedOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(hcl::MulFixedOp op) { return emitter.emitBinary(op, "*"), true; }

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

bool ExprVisitor::visitOp(CmpIOp op) {
  switch (op.getPredicate()) {
  case CmpIPredicate::eq:
    return emitter.emitBinary(op, "=="), true;
  case CmpIPredicate::ne:
    return emitter.emitBinary(op, "!="), true;
  case CmpIPredicate::slt:
  case CmpIPredicate::ult:
    return emitter.emitBinary(op, "<"), true;
  case CmpIPredicate::sle:
  case CmpIPredicate::ule:
    return emitter.emitBinary(op, "<="), true;
  case CmpIPredicate::sgt:
  case CmpIPredicate::ugt:
    return emitter.emitBinary(op, ">"), true;
  case CmpIPredicate::sge:
  case CmpIPredicate::uge:
    return emitter.emitBinary(op, ">="), true;
  default:
    op.emitError("has unsupported compare type.");
    return false;
  }
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Definition
//===----------------------------------------------------------------------===//

/// SCF statement emitters.
void ModuleEmitter::emitScfFor(scf::ForOp op) {
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  emitValue(op.lowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.upperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.step());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitScfIf(scf::IfOp op) {
  // Declare all values returned by scf::YieldOp. They will be further handled
  // by the scf::YieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent();
  os << "if (";
  emitValue(op.condition());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(op.thenRegion().front());
  reduceIndent();

  if (!op.elseRegion().empty()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(op.elseRegion().front());
    reduceIndent();
  }

  indent();
  os << "}\n";
}

void ModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only and scf::If operations will use scf::Yield to return
  // generated values.
  if (auto parentOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
  }
}

/// Affine statement emitters.
void ModuleEmitter::emitAffineFor(AffineForOp op) {
  indent();
  auto iterVar = op.getInductionVar();
  std::string loop_name = "";
  if (op->hasAttr("loop_name")) { // loop label
    loop_name = op->getAttr("loop_name").cast<StringAttr>().getValue().str();
    std::replace(loop_name.begin(), loop_name.end(), '.', '_');
    os << "l_";
    os << addName(iterVar, false, loop_name);
    os << ": ";
  }
  os << "for (";

  // Emit lower bound.
  if (op->hasAttr("loop_name")) {
    os << getTypeName(iterVar) << " ";
  }
  emitValue(iterVar, 0, false, loop_name);
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
  emitValue(iterVar, 0, false, loop_name);
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
  emitValue(iterVar, 0, false, loop_name);
  os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineIf(AffineIfOp op) {
  // Declare all values returned by AffineYieldOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent();
  os << "if (";
  auto constrSet = op.getIntegerSet();
  AffineExprEmitter constrEmitter(state, constrSet.getNumDims(),
                                  op.getOperands());

  // Emit all constraints.
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    constrEmitter.emitAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      os << " == 0";
    else
      os << " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      os << " && ";
  }
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(*op.getThenBlock());
  reduceIndent();

  if (op.hasElse()) {
    indent();
    os << "} else {\n";
    addIndent();
    emitBlock(*op.getElseBlock());
    reduceIndent();
  }

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineParallel(AffineParallelOp op) {
  // Declare all values returned by AffineParallelOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  auto steps = getIntArrayAttrValue(op, op.getStepsAttrName());
  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    indent();
    os << "for (";
    auto iterVar = op.getBody()->getArgument(i);

    // Emit lower bound.
    emitValue(iterVar);
    os << " = ";
    auto lowerMap = op.getLowerBoundsValueMap().getAffineMap();
    AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                   op.getLowerBoundsOperands());
    lowerEmitter.emitAffineExpr(lowerMap.getResult(i));
    os << "; ";

    // Emit upper bound.
    emitValue(iterVar);
    os << " < ";
    auto upperMap = op.getUpperBoundsValueMap().getAffineMap();
    AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                   op.getUpperBoundsOperands());
    upperEmitter.emitAffineExpr(upperMap.getResult(i));
    os << "; ";

    // Emit increase step.
    emitValue(iterVar);
    os << " += " << steps[i] << ") {";
    emitInfoAndNewLine(op);

    addIndent();
  }

  emitBlock(*op.getBody());

  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    reduceIndent();

    indent();
    os << "}\n";
  }
}

void ModuleEmitter::emitAffineApply(AffineApplyOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter(state, affineMap.getNumDims(), op.getOperands())
      .emitAffineExpr(affineMap.getResult(0));
  os << ";";
  emitInfoAndNewLine(op);
}

template <typename OpType>
void ModuleEmitter::emitAffineMaxMin(OpType op, const char *syntax) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getOperands());
  for (unsigned i = 0, e = affineMap.getNumResults() - 1; i < e; ++i)
    os << syntax << "(";
  affineEmitter.emitAffineExpr(affineMap.getResult(0));
  for (auto &expr : llvm::drop_begin(affineMap.getResults(), 1)) {
    os << ", ";
    affineEmitter.emitAffineExpr(expr);
    os << ")";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
  }
  emitValue(op.getResult());
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, load_from_name);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    os << ".read(); // ";
    emitValue(memref, 0, false, load_from_name); // comment
  }
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
  std::string store_to_name = "";
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
  }
  auto memref = op.getMemRef();
  emitValue(memref, 0, false, store_to_name);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    os << ".write(";
    emitValue(op.getValueToStore());
    os << "); // ";
    emitValue(memref, 0, false, store_to_name); // comment
  }
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

// TODO: For now, all values created in the AffineIf region will be declared
// in the generated C++. However, values which will be returned by affine
// yield operation should not be declared again. How to "bind" the pair of
// values inside/outside of AffineIf region needs to be considered.
void ModuleEmitter::emitAffineYield(AffineYieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only AffineParallel and AffineIf operations will use
  // AffineYield to return generated values.
  if (auto parentOp = dyn_cast<AffineIfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
  } else if (auto parentOp = dyn_cast<AffineParallelOp>(op->getParentOp())) {
    indent();
    os << "if (";
    unsigned ivIdx = 0;
    for (auto iv : parentOp.getBody()->getArguments()) {
      emitValue(iv);
      os << " == 0";
      if (ivIdx++ != parentOp.getBody()->getNumArguments() - 1)
        os << " && ";
    }
    os << ") {\n";

    // When all induction values are 0, generated values will be directly
    // assigned to the current results, correspondingly.
    addIndent();
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
    reduceIndent();

    indent();
    os << "} else {\n";

    // Otherwise, generated values will be accumulated/reduced to the
    // current results with corresponding AtomicRMWKind operations.
    addIndent();
    auto RMWAttrs =
        getIntArrayAttrValue(parentOp, parentOp.getReductionsAttrName());
    resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result, rank);
      switch ((AtomicRMWKind)RMWAttrs[resultIdx]) {
      case (AtomicRMWKind::addf):
      case (AtomicRMWKind::addi):
        os << " += ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (AtomicRMWKind::assign):
        os << " = ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (AtomicRMWKind::maxf):
      case (AtomicRMWKind::maxs):
      case (AtomicRMWKind::maxu):
        os << " = max(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (AtomicRMWKind::minf):
      case (AtomicRMWKind::mins):
      case (AtomicRMWKind::minu):
        os << " = min(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (AtomicRMWKind::mulf):
      case (AtomicRMWKind::muli):
        os << " *= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      }
      os << ";";
      emitInfoAndNewLine(op);
      emitNestedLoopTail(rank);
    }
    reduceIndent();

    indent();
    os << "}\n";
  }
}

/// Memref-related statement emitters.
template <typename OpType> void ModuleEmitter::emitAlloc(OpType op) {
  // A declared result indicates that the memref is output of the function, and
  // has been declared in the function signature.
  if (isDeclared(op.getResult()))
    return;

  // Vivado HLS only supports static shape on-chip memory.
  if (!op.getType().hasStaticShape())
    emitError(op, "is unranked or has dynamic shape.");

  std::string name;
  if (op->hasAttr("name")) {
    auto attr = op->getAttr("name").template cast<StringAttr>();
    name = attr.getValue().str();
  }

  indent();
  emitArrayDecl(op.getResult(), false, name);
  os << ";";
  emitInfoAndNewLine(op);
  emitArrayDirectives(op.getResult());
}

void ModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    os << ".read(); // ";
    emitValue(memref); // comment
  }
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    os << ".write(";
    emitValue(op.getValueToStore());
    os << "); // ";
    emitValue(memref); // comment
  }
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitTensorExtract(tensor::ExtractOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.tensor());
  for (auto index : op.indices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitTensorInsert(tensor::InsertOp op) {
  indent();
  emitValue(op.dest());
  for (auto index : op.indices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.scalar());
  os << ";";
  emitInfoAndNewLine(op);
}

/// Tensor-related statement emitters.
void ModuleEmitter::emitTensorLoad(memref::TensorLoadOp op) {
  // TODO: stream interface for tensor?
  auto rank = emitNestedLoopHead(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getOperand(), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitTensorStore(memref::TensorStoreOp op) {
  // TODO: stream interface for tensor?
  auto rank = emitNestedLoopHead(op.getOperand(0));
  indent();
  emitValue(op.getOperand(1), rank);
  os << " = ";
  emitValue(op.getOperand(0), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitTensorToMemref(memref::BufferCastOp op) {
  // A declared result indicates that the memref is output of the function, and
  // has been declared in the function signature.
  if (isDeclared(op.getResult())) {
    auto rank = emitNestedLoopHead(op.getResult());
    indent();
    emitValue(op.getResult(), rank);
    os << " = ";
    emitValue(op.getOperand(), rank);
    os << ";";
    emitInfoAndNewLine(op);
    emitNestedLoopTail(rank);
  } else {
    addAlias(op.getOperand(), op.getResult());
    emitArrayDirectives(op.getResult());
  }
}

void ModuleEmitter::emitDim(memref::DimOp op) {
  if (auto constOp = dyn_cast<ConstantOp>(op.getOperand(1).getDefiningOp())) {
    auto constVal = constOp.value().cast<IntegerAttr>().getInt();
    auto type = op.getOperand(0).getType().cast<ShapedType>();

    if (type.hasStaticShape()) {
      if (constVal >= 0 && constVal < (int64_t)type.getShape().size()) {
        indent();
        emitValue(op.getResult());
        os << " = ";
        os << type.getShape()[constVal] << ";";
        emitInfoAndNewLine(op);
      } else
        emitError(op, "index is out of range.");
    } else
      emitError(op, "is unranked or has dynamic shape.");
  } else
    emitError(op, "index is not a constant.");
}

void ModuleEmitter::emitRank(RankOp op) {
  auto type = op.getOperand().getType().cast<ShapedType>();
  if (type.hasRank()) {
    indent();
    emitValue(op.getResult());
    os << " = ";
    os << type.getRank() << ";";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "is unranked.");
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
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

/// Special operation emitters.
void ModuleEmitter::emitSelect(SelectOp op) {
  unsigned rank = emitNestedLoopHead(op.getResult());
  unsigned conditionRank = rank;
  if (!op.getCondition().getType().isa<ShapedType>())
    conditionRank = 0;

  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getCondition(), conditionRank);
  os << " ? ";
  os << "(" << getTypeName(op.getTrueValue()) << ")";
  emitValue(op.getTrueValue(), rank);
  os << " : ";
  os << "(" << getTypeName(op.getFalseValue()) << ")";
  emitValue(op.getFalseValue(), rank);
  os << ";";
  emitInfoAndNewLine(op);
  emitNestedLoopTail(rank);
}

void ModuleEmitter::emitConstant(ConstantOp op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = op.value().dyn_cast<DenseElementsAttr>()) {
    indent();
    emitArrayDecl(op.getResult());
    os << " = {";
    auto type = op.getResult().getType().cast<ShapedType>().getElementType();

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value = element.cast<FloatAttr>().getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        os << element.cast<IntegerAttr>().getValue();
      else
        emitError(op, "array has unsupported element type.");

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "has unsupported constant type.");
}

template <typename CastOpType> void ModuleEmitter::emitCast(CastOpType op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getOperand());
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitGeneralCast(UnrealizedConversionCastOp op) {
  indent();
  emitValue(op.getResult(0));
  os << " = ";
  emitValue(op.getOperand(0));
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitCall(CallOp op) {
  // Handle returned value by the callee.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (result.getType().isa<ShapedType>())
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  // Emit the function call.
  indent();
  os << op.getCallee() << "(";

  // Handle input arguments.
  unsigned argIdx = 0;
  for (auto arg : op.getOperands()) {
    emitValue(arg);

    if (argIdx++ != op.getNumOperands() - 1)
      os << ", ";
  }

  // Handle output arguments.
  for (auto result : op.getResults()) {
    // The address should be passed in for scalar result arguments.
    if (result.getType().isa<ShapedType>())
      os << ", ";
    else
      os << ", &";

    emitValue(result);
  }

  os << ");";
  emitInfoAndNewLine(op);
}

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getTypeName(val) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void ModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    auto attr = array.getType().dyn_cast<MemRefType>().getMemorySpace();
    if (attr &&
        attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
      // Value has been declared before or is a constant number.
      if (isDeclared(array)) {
        os << getName(array);
        return;
      }

      // print stream type
      os << "hls::stream< " << getTypeName(array) << " > ";
      if (isFunc) {
        os << "&"; // pass by reference
      }

      // Add the new value to nameTable and emit its name.
      os << addName(array, /*isPtr=*/false, name);
      // Add original array declaration as comment
      os << " /* ";
      emitValue(array, 0, false, name);
      for (auto &shape : arrayType.getShape())
        os << "[" << shape << "]";
      os << " */";
    } else {
      emitValue(array, 0, false, name);
      for (auto &shape : arrayType.getShape())
        os << "[" << shape << "]";
    }
  } else
    emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
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

void ModuleEmitter::emitNestedLoopTail(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();

    indent();
    os << "}\n";
  }
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

void ModuleEmitter::emitLoopDirectives(Operation *op) {
  if (auto ii = getLoopDirective(op, "pipeline_ii")) {
    reduceIndent();
    indent();
    os << "#pragma HLS pipeline II=" << ii.cast<IntegerAttr>().getValue()
       << "\n";
    addIndent();
  }

  if (auto factor = getLoopDirective(op, "unroll")) {
    reduceIndent();
    indent();
    auto val = factor.cast<IntegerAttr>().getValue();
    if (val == 0)
      os << "#pragma HLS unroll"
         << "\n";
    else
      os << "#pragma HLS unroll factor=" << val << "\n";
    addIndent();
  }

  if (auto dataflow = getLoopDirective(op, "dataflow")) {
    reduceIndent();
    indent();
    os << "#pragma HLS dataflow\n";
    addIndent();
  }
}

void ModuleEmitter::emitArrayDirectives(Value memref) {
  bool emitPragmaFlag = false;
  auto type = memref.getType().cast<MemRefType>();

  if (auto layoutMap = getLayoutMap(type)) {
    // Emit array_partition pragma(s).
    SmallVector<int64_t, 8> factors;
    getPartitionFactors(type, &factors);

    for (int64_t dim = 0; dim < type.getRank(); ++dim) {
      if (!isFullyPartitioned(type, dim)) {
        if (factors[dim] != 1) {
          emitPragmaFlag = true;

          indent();
          os << "#pragma HLS array_partition";
          os << " variable=";
          emitValue(memref);

          // Emit partition type.
          if (layoutMap.getResult(dim).getKind() == AffineExprKind::FloorDiv)
            os << " block";
          else
            os << " cyclic";

          os << " factor=" << factors[dim];
          os << " dim=" << dim + 1 << "\n";
        }
      } else { // fully partitioned
        emitPragmaFlag = true;

        indent();
        os << "#pragma HLS array_partition";
        os << " variable=";
        emitValue(memref);

        // Emit partition type.
        os << " complete";
        os << " dim=" << dim + 1 << "\n";
      }
    }
  }

  // // Emit resource pragma when the array is not DRAM kind and is not fully
  // // partitioned.
  // auto kind = MemoryKind(type.getMemorySpaceAsInt());
  // if (kind != MemoryKind::DRAM && !isFullyPartitioned(type)) {
  //   emitPragmaFlag = true;

  //   indent();
  //   os << "#pragma HLS resource";
  //   os << " variable=";
  //   emitValue(memref);

  //   os << " core=";
  //   if (kind == MemoryKind::BRAM_1P)
  //     os << "ram_1p_bram";
  //   else if (kind == MemoryKind::BRAM_S2P)
  //     os << "ram_s2p_bram";
  //   else if (kind == MemoryKind::BRAM_T2P)
  //     os << "ram_t2p_bram";
  //   else
  //     os << "ram_s2p_bram";
  //   os << "\n";
  // }

  // streaming
  auto attr = type.getMemorySpace();
  if (attr) {
    std::string attr_str = attr.cast<StringAttr>().getValue().str();
    if (attr_str.substr(0, 6) == "stream") {
      indent();
      os << "#pragma HLS stream variable=";
      emitValue(memref);
      os << " depth=";
      os << attr_str.substr(7, std::string::npos);
      os << "\n";
    }
  }

  // Emit an empty line.
  if (emitPragmaFlag)
    os << "\n";
}

void ModuleEmitter::emitFunctionDirectives(FuncOp func,
                                           ArrayRef<Value> portList) {
  // auto funcDirect = getFuncDirective(func);
  // if (!funcDirect)
  //   return;

  // if (funcDirect.getPipeline()) {
  //   indent();
  //   os << "#pragma HLS pipeline II=" << funcDirect.getTargetInterval() <<
  //   "\n";

  //   // An empty line.
  //   os << "\n";
  // } else if (funcDirect.getDataflow()) {
  //   indent();
  //   os << "#pragma HLS dataflow\n";

  //   // An empty line.
  //   os << "\n";
  // }

  // // Only top function should emit interface pragmas.
  // if (funcDirect.getTopFunc()) {
  //   indent();
  //   os << "#pragma HLS interface s_axilite port=return bundle=ctrl\n";

  //   for (auto &port : portList) {
  //     // Array ports and scalar ports are handled separately. Here, we only
  //     // handle MemRef types since we assume the IR has be fully bufferized.
  //     if (auto memrefType = port.getType().dyn_cast<MemRefType>()) {
  //       // Only emit interface pragma when the array is not fully
  //       partitioned. if (!isFullyPartitioned(memrefType)) {
  //         indent();
  //         os << "#pragma HLS interface";
  //         // For now, we set the offset of all m_axi interfaces as slave.
  //         if (MemoryKind(memrefType.getMemorySpaceAsInt()) ==
  //         MemoryKind::DRAM)
  //           os << " m_axi offset=slave";
  //         else
  //           os << " bram";

  //         os << " port=";
  //         emitValue(port);
  //         os << "\n";
  //       }
  //     } else {
  //       indent();
  //       os << "#pragma HLS interface s_axilite";
  //       os << " port=";

  //       // TODO: This is a temporary solution.
  //       auto name = getName(port);
  //       if (name.front() == "*"[0])
  //         name.erase(name.begin());
  //       os << name;
  //       os << " bundle=ctrl\n";
  //     }
  //   }

  //   // An empty line.
  //   os << "\n";
  if (func->hasAttr("dataflow")) {
    indent();
    os << "#pragma HLS dataflow\n";
  }

  // Emit other pragmas for function ports.
  for (auto &port : portList)
    if (port.getType().isa<MemRefType>())
      emitArrayDirectives(port);
  // }
}

void ModuleEmitter::emitFunction(FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  if (func->hasAttr("top"))
    os << "/// This is top function.\n";

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        func->getAttr("inputs").cast<StringAttr>().getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names =
        func->getAttr("outputs").cast<StringAttr>().getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  for (auto &arg : func.getArguments()) {
    indent();
    if (input_args.size() == 0) {
      if (arg.getType().isa<ShapedType>())
        emitArrayDecl(arg, true);
      else
        emitValue(arg);
    } else {
      if (arg.getType().isa<ShapedType>())
        emitArrayDecl(arg, true, input_args[argIdx]);
      else
        emitValue(arg, 0, false, input_args[argIdx]);
    }

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  auto args = func.getArguments();
  if (auto funcReturn = dyn_cast<ReturnOp>(func.front().getTerminator())) {
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        os << ",\n";
        indent();
        if (output_names != "") {
          // TODO: a known bug, cannot return a value twice, e.g. return %0, %0 :
          // index, index. However, typically this should not happen.
          if (result.getType().isa<ShapedType>())
            emitArrayDecl(result, true);
          else
            // In Vivado HLS, pointer indicates the value is an output.
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
        } else {
          if (result.getType().isa<ShapedType>())
            emitArrayDecl(result, true, output_names);
          else
            // In Vivado HLS, pointer indicates the value is an output.
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);
  emitBlock(func.front());
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitHostFunction(FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  os << "/// This is top function.\n";

  // Emit function signature.
  os << "int main(int argc, char **argv) {\n";
  addIndent();

  emitBlock(func.front());

  os << "  return 0;\n";
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
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

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// vivado hls headers
#include "kernel.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().hasValue() && module.getName().getValue() == "host") {
    os << host_header;
    for (auto op : module.getOps<FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<FuncOp>(op))
        emitFunction(func);
      else
        emitError(&op, "is unsupported operation.");
    }
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