//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#include "hcl/Translation/EmitMlirGpu.h"
#include "hcl/Dialect/Visitor.h"
#include "hcl/Support/Utils.h"
#include "hcl/Translation/Utils.h"
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

// used for determine whether to generate C++ default types or ap_(u)int
static bool BIT_FLAG = false;

// TODO

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public HCLEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HCLEmitterState &state) : HCLEmitterBase(state) {}

  /// SCF statement emitters.
  // void emitScfFor(scf::ForOp op);
  // void emitScfIf(scf::IfOp op);
  // void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  void emitAffineFor(AffineForOp op);
  // void emitAffineIf(AffineIfOp op);
  // void emitAffineParallel(AffineParallelOp op);
  void emitAffineApply(AffineApplyOp op);
  // template <typename OpType>
  // void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(AffineLoadOp op);
  void emitAffineStore(AffineStoreOp op);
  void emitAffineYield(AffineYieldOp op);

  /// Memref-related statement emitters.
  // template <typename OpType> void emitAlloc(OpType op);
  // void emitLoad(memref::LoadOp op);
  // void emitStore(memref::StoreOp op);

  /// Tensor-related statement emitters.
  // void emitTensorExtract(tensor::ExtractOp op);
  // void emitTensorInsert(tensor::InsertOp op);
  // void emitTensorStore(memref::TensorStoreOp op);
  // void emitDim(memref::DimOp op);
  // void emitRank(memref::RankOp op);

  /// Standard expression emitters.
  void emitBinary(Operation *op, const char *syntax);
  // void emitUnary(Operation *op, const char *syntax);
  // void emitPower(Operation *op);
  // void emitMaxMin(Operation *op, const char *syntax);

  /// Special operation emitters.
  // void emitCall(CallOp op);
  // void emitSelect(SelectOp op);
  // void emitConstant(arith::ConstantOp op);
  // template <typename CastOpType> void emitCast(CastOpType op);
  // void emitGeneralCast(UnrealizedConversionCastOp op);
  // void emitGetBit(hcl::GetIntBitOp op);
  // void emitSetBit(hcl::SetIntBitOp op);
  // void emitGetSlice(hcl::GetIntSliceOp op);
  // void emitSetSlice(hcl::SetIntSliceOp op);
  // void emitBitReverse(hcl::BitReverseOp op);
  // void emitBitcast(arith::BitcastOp op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// C++ component emitters.
  // void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
  //                std::string name = "");
  void emitArrayDecl(Value array, bool isFunc = false, std::string name = "");
  // unsigned emitNestedLoopHead(Value val);
  // void emitNestedLoopTail(unsigned rank);
  void emitInfoAndNewLine(Operation *op);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  // void emitLoopDirectives(Operation *op);
  // void emitArrayDirectives(Value memref);
  // void emitFunctionDirectives(FuncOp func, ArrayRef<Value> portList);
  void emitFunction(FuncOp func);
  // void emitHostFunction(FuncOp func);
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineEmitter Class
//===----------------------------------------------------------------------===//

// TODO

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
  bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  // bool visitOp(AffineMaxOp op) {
  //   return emitter.emitAffineMaxMin<AffineMaxOp>(op, "max"), true;
  // }
  // bool visitOp(AffineMinOp op) {
  //   return emitter.emitAffineMaxMin<AffineMinOp>(op, "min"), true;
  // }
  bool visitOp(AffineLoadOp op) { return emitter.emitAffineLoad(op), true; }
  bool visitOp(AffineStoreOp op) { return emitter.emitAffineStore(op), true; }
  bool visitOp(AffineYieldOp op) { return emitter.emitAffineYield(op), true; }

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

  /// Tensor-related statements.
  // bool visitOp(tensor::ExtractOp op) {
  //   return emitter.emitTensorExtract(op), true;
  // }
  // bool visitOp(tensor::InsertOp op) {
  //   return emitter.emitTensorInsert(op), true;
  // }
  // bool visitOp(memref::TensorStoreOp op) {
  //   return emitter.emitTensorStore(op), true;
  // }
  // bool visitOp(memref::DimOp op) { return emitter.emitDim(op), true; }
  // bool visitOp(memref::RankOp op) { return emitter.emitRank(op), true; }

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
  // bool visitOp(arith::CmpFOp op);
  bool visitOp(arith::AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  // bool visitOp(arith::SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  // bool visitOp(arith::MulFOp op) { return emitter.emitBinary(op, "*"), true; }
  // bool visitOp(arith::DivFOp op) { return emitter.emitBinary(op, "/"), true; }
  // bool visitOp(arith::RemFOp op) { return emitter.emitBinary(op, "%"), true; }

  /// Integer binary expressions.
  // bool visitOp(arith::CmpIOp op);
  // bool visitOp(arith::AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  // bool visitOp(arith::SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  // bool visitOp(arith::MulIOp op) { return emitter.emitBinary(op, "*"), true; }
  // bool visitOp(arith::DivSIOp op) { return emitter.emitBinary(op, "/"), true; }
  // bool visitOp(arith::RemSIOp op) { return emitter.emitBinary(op, "%"), true; }
  // bool visitOp(arith::DivUIOp op) { return emitter.emitBinary(op, "/"), true; }
  // bool visitOp(arith::RemUIOp op) { return emitter.emitBinary(op, "%"), true; }
  // bool visitOp(arith::MaxSIOp op) {
  //   return emitter.emitMaxMin(op, "max"), true;
  // }
  // bool visitOp(arith::MinSIOp op) {
  //   return emitter.emitMaxMin(op, "min"), true;
  // }
  // bool visitOp(arith::MaxUIOp op) {
  //   return emitter.emitMaxMin(op, "max"), true;
  // }
  // bool visitOp(arith::MinUIOp op) {
  //   return emitter.emitMaxMin(op, "min"), true;
  // }

  /// Logical expressions.
  // bool visitOp(arith::XOrIOp op) { return emitter.emitBinary(op, "^"), true; }
  // bool visitOp(arith::AndIOp op) { return emitter.emitBinary(op, "&"), true; }
  // bool visitOp(arith::OrIOp op) { return emitter.emitBinary(op, "|"), true; }
  // bool visitOp(arith::ShLIOp op) { return emitter.emitBinary(op, "<<"), true; }
  // bool visitOp(arith::ShRSIOp op) { return emitter.emitBinary(op, ">>"), true; }
  // bool visitOp(arith::ShRUIOp op) { return emitter.emitBinary(op, ">>"), true; }
  // bool visitOp(hcl::GetIntBitOp op) { return emitter.emitGetBit(op), true; }
  // bool visitOp(hcl::SetIntBitOp op) { return emitter.emitSetBit(op), true; }
  // bool visitOp(hcl::GetIntSliceOp op) { return emitter.emitGetSlice(op), true; }
  // bool visitOp(hcl::SetIntSliceOp op) { return emitter.emitSetSlice(op), true; }
  // bool visitOp(hcl::BitReverseOp op) {
  //   return emitter.emitBitReverse(op), true;
  // }

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
  // bool visitOp(math::PowFOp op) { return emitter.emitPower(op), true; }
  // bool visitOp(math::LogOp op) { return emitter.emitUnary(op, "log"), true; }
  // bool visitOp(math::Log2Op op) { return emitter.emitUnary(op, "log2"), true; }
  // bool visitOp(math::Log10Op op) {
  //   return emitter.emitUnary(op, "log10"), true;
  // }
  // bool visitOp(arith::NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  /// Special operations.
  // bool visitOp(CallOp op) { return emitter.emitCall(op), true; }
  // bool visitOp(ReturnOp op) { return true; }
  // bool visitOp(SelectOp op) { return emitter.emitSelect(op), true; }
  // bool visitOp(arith::ConstantOp op) { return emitter.emitConstant(op), true; }
  // bool visitOp(arith::IndexCastOp op) {
  //   return emitter.emitCast<arith::IndexCastOp>(op), true;
  // }
  // bool visitOp(arith::UIToFPOp op) {
  //   return emitter.emitCast<arith::UIToFPOp>(op), true;
  // }
  // bool visitOp(arith::SIToFPOp op) {
  //   return emitter.emitCast<arith::SIToFPOp>(op), true;
  // }
  // bool visitOp(arith::FPToUIOp op) {
  //   return emitter.emitCast<arith::FPToUIOp>(op), true;
  // }
  // bool visitOp(arith::FPToSIOp op) {
  //   return emitter.emitCast<arith::FPToSIOp>(op), true;
  // }
  // bool visitOp(arith::TruncIOp op) {
  //   return emitter.emitCast<arith::TruncIOp>(op), true;
  // }
  // bool visitOp(arith::TruncFOp op) {
  //   return emitter.emitCast<arith::TruncFOp>(op), true;
  // }
  // bool visitOp(arith::ExtSIOp op) {
  //   return emitter.emitCast<arith::ExtSIOp>(op), true;
  // }
  // bool visitOp(arith::ExtUIOp op) {
  //   return emitter.emitCast<arith::ExtUIOp>(op), true;
  // }
  // bool visitOp(arith::ExtFOp op) {
  //   return emitter.emitCast<arith::ExtFOp>(op), true;
  // }
  // bool visitOp(arith::BitcastOp op) { return emitter.emitBitcast(op), true; }
  // bool visitOp(UnrealizedConversionCastOp op) {
  //   return emitter.emitGeneralCast(op), true;
  // }

  /// HCL operations.
  bool visitOp(hcl::CreateLoopHandleOp op) { return true; }
  bool visitOp(hcl::CreateStageHandleOp op) { return true; }

  /// Fixed points
  // bool visitOp(hcl::AddFixedOp op) { return emitter.emitBinary(op, "+"), true; }
  // bool visitOp(hcl::SubFixedOp op) { return emitter.emitBinary(op, "-"), true; }
  // bool visitOp(hcl::MulFixedOp op) { return emitter.emitBinary(op, "*"), true; }
  // bool visitOp(hcl::CmpFixedOp op);
  // bool visitOp(hcl::MinFixedOp op) {
  //   return emitter.emitMaxMin(op, "min"), true;
  // }
  // bool visitOp(hcl::MaxFixedOp op) {
  //   return emitter.emitMaxMin(op, "max"), true;
  // }

private:
  ModuleEmitter &emitter;
};
} // namespace

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Definition
//===----------------------------------------------------------------------===//

void ModuleEmitter::emitAffineFor(AffineForOp op) {
  indent();
  os << "FOR";
  // auto iterVar = op.getInductionVar();
  // std::string loop_name = "";
  // if (op->hasAttr("loop_name")) { // loop label
  //   loop_name = op->getAttr("loop_name").cast<StringAttr>().getValue().str();
  //   std::replace(loop_name.begin(), loop_name.end(), '.', '_');
  //   os << "l_";
  //   if (op->hasAttr("stage_name")) {
  //     std::string stage_name =
  //         op->getAttr("stage_name").cast<StringAttr>().getValue().str();
  //     std::replace(stage_name.begin(), stage_name.end(), '.', '_');
  //     os << stage_name << "_";
  //   }
  //   os << addName(iterVar, false, loop_name);
  //   os << ": ";
  // }
  // os << "(";

  // Emit lower bound.
  // if (op->hasAttr("loop_name")) {
  //   os << getTypeName(iterVar) << " ";
  // }
  // emitValue(iterVar, 0, false, loop_name);
  // os << " = ";
  // auto lowerMap = op.getLowerBoundMap();
  // AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
  //                                op.getLowerBoundOperands());
  // if (lowerMap.getNumResults() == 1)
  //   lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  // else {
  //   for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
  //     os << "max(";
  //   lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  //   for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
  //     os << ", ";
  //     lowerEmitter.emitAffineExpr(expr);
  //     os << ")";
  //   }
  // }
  // os << "; ";

  // Emit upper bound.
  // emitValue(iterVar, 0, false, loop_name);
  // os << " < ";
  // auto upperMap = op.getUpperBoundMap();
  // AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
  //                                op.getUpperBoundOperands());
  // if (upperMap.getNumResults() == 1)
  //   upperEmitter.emitAffineExpr(upperMap.getResult(0));
  // else {
  //   for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
  //     os << "min(";
  //   upperEmitter.emitAffineExpr(upperMap.getResult(0));
  //   for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
  //     os << ", ";
  //     upperEmitter.emitAffineExpr(expr);
  //     os << ")";
  //   }
  // }
  // os << "; ";

  // Emit increase step.
  // emitValue(iterVar, 0, false, loop_name);
  // if (op.getStep() == 1)
  //   os << "++) {";
  // else
  //   os << " += " << op.getStep() << ") {";
  emitInfoAndNewLine(op);

  addIndent();

  // emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitAffineApply(AffineApplyOp op) {
  indent();
  os << "APPLY " << op;
  // emitValue(op.getResult());
  // os << " = ";
  // auto affineMap = op.getAffineMap();
  // AffineExprEmitter(state, affineMap.getNumDims(), op.getOperands())
  //     .emitAffineExpr(affineMap.getResult(0));
  // os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  std::string load_from_name = "";
  if (op->hasAttr("from")) {
    load_from_name = op->getAttr("from").cast<StringAttr>().getValue().str();
  }
  // Value result = op.getResult();
  // fixUnsignedType(result, op->hasAttr("unsigned"));
  // emitValue(result);
  // os << " = ";
  auto memref = op.getMemRef();
  // emitValue(memref, 0, false, load_from_name);
  os << "*" << getName(memref) << "*";
  // auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  // if (attr &&
  //     attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
  //   os << ".read(); // ";
  //   emitValue(memref, 0, false, load_from_name); // comment
  // }
  // auto affineMap = op.getAffineMap();
  // AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
  //                                 op.getMapOperands());
  // auto arrayType = memref.getType().cast<ShapedType>();
  // if (arrayType.getShape().size() == 1 && arrayType.getShape()[0] == 1) {
  //   // do nothing;
  // } else {
  //   for (auto index : affineMap.getResults()) {
  //     os << "[";
  //     affineEmitter.emitAffineExpr(index);
  //     os << "]";
  //   }
  // }
  os << op;
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  std::string store_to_name = "";
  if (op->hasAttr("to")) {
    store_to_name = op->getAttr("to").cast<StringAttr>().getValue().str();
  }
  auto memref = op.getMemRef();
  // emitValue(memref, 0, false, store_to_name);
  os << "*" << getName(memref) << "*";
  // auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  // if (attr &&
  //     attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
  //   os << ".write(";
  //   emitValue(op.getValueToStore());
  //   os << "); // ";
  //   emitValue(memref, 0, false, store_to_name); // comment
  // }
  // auto affineMap = op.getAffineMap();
  // AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
  //                                 op.getMapOperands());
  // auto arrayType = memref.getType().cast<ShapedType>();
  // if (arrayType.getShape().size() == 1 && arrayType.getShape()[0] == 1) {
  //   // do nothing;
  // } else {
  //   for (auto index : affineMap.getResults()) {
  //     os << "[";
  //     affineEmitter.emitAffineExpr(index);
  //     os << "]";
  //   }
  // }
  // os << " = ";
  // emitValue(op.getValueToStore());
  // os << ";";
  os << op;
  emitInfoAndNewLine(op);
}

// TODO: For now, all values created in the AffineIf region will be declared
// in the generated C++. However, values which will be returned by affine
// yield operation should not be declared again. How to "bind" the pair of
// values inside/outside of AffineIf region needs to be considered.
void ModuleEmitter::emitAffineYield(AffineYieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // // For now, only AffineParallel and AffineIf operations will use
  // // AffineYield to return generated values.
  // if (auto parentOp = dyn_cast<AffineIfOp>(op->getParentOp())) {
  //   unsigned resultIdx = 0;
  //   for (auto result : parentOp.getResults()) {
  //     unsigned rank = emitNestedLoopHead(result);
  //     indent();
  //     emitValue(result, rank);
  //     os << " = ";
  //     emitValue(op.getOperand(resultIdx++), rank);
  //     os << ";";
  //     emitInfoAndNewLine(op);
  //     emitNestedLoopTail(rank);
  //   }
  // } else if (auto parentOp = dyn_cast<AffineParallelOp>(op->getParentOp())) {
  //   indent();
  //   os << "if (";
  //   unsigned ivIdx = 0;
  //   for (auto iv : parentOp.getBody()->getArguments()) {
  //     emitValue(iv);
  //     os << " == 0";
  //     if (ivIdx++ != parentOp.getBody()->getNumArguments() - 1)
  //       os << " && ";
  //   }
  //   os << ") {\n";

  //   // When all induction values are 0, generated values will be directly
  //   // assigned to the current results, correspondingly.
  //   addIndent();
  //   unsigned resultIdx = 0;
  //   for (auto result : parentOp.getResults()) {
  //     unsigned rank = emitNestedLoopHead(result);
  //     indent();
  //     emitValue(result, rank);
  //     os << " = ";
  //     emitValue(op.getOperand(resultIdx++), rank);
  //     os << ";";
  //     emitInfoAndNewLine(op);
  //     emitNestedLoopTail(rank);
  //   }
  //   reduceIndent();

  //   indent();
  //   os << "} else {\n";

  //   // Otherwise, generated values will be accumulated/reduced to the
  //   // current results with corresponding arith::AtomicRMWKind operations.
  //   addIndent();
  //   auto RMWAttrs =
  //       getIntArrayAttrValue(parentOp, parentOp.getReductionsAttrName());
  //   resultIdx = 0;
  //   for (auto result : parentOp.getResults()) {
  //     unsigned rank = emitNestedLoopHead(result);
  //     indent();
  //     emitValue(result, rank);
  //     switch ((arith::AtomicRMWKind)RMWAttrs[resultIdx]) {
  //     case (arith::AtomicRMWKind::addf):
  //     case (arith::AtomicRMWKind::addi):
  //       os << " += ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       break;
  //     case (arith::AtomicRMWKind::assign):
  //       os << " = ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       break;
  //     case (arith::AtomicRMWKind::maxf):
  //     case (arith::AtomicRMWKind::maxs):
  //     case (arith::AtomicRMWKind::maxu):
  //       os << " = max(";
  //       emitValue(result, rank);
  //       os << ", ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       os << ")";
  //       break;
  //     case (arith::AtomicRMWKind::minf):
  //     case (arith::AtomicRMWKind::mins):
  //     case (arith::AtomicRMWKind::minu):
  //       os << " = min(";
  //       emitValue(result, rank);
  //       os << ", ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       os << ")";
  //       break;
  //     case (arith::AtomicRMWKind::mulf):
  //     case (arith::AtomicRMWKind::muli):
  //       os << " *= ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       break;
  //     case (arith::AtomicRMWKind::ori):
  //       os << " |= ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       break;
  //     case (arith::AtomicRMWKind::andi):
  //       os << " &= ";
  //       emitValue(op.getOperand(resultIdx++), rank);
  //       break;
  //     }
  //     os << ";";
  //     emitInfoAndNewLine(op);
  //     emitNestedLoopTail(rank);
  //   }
  //   reduceIndent();

  //   indent();
  //   os << "}\n";
  // }
}

/// Standard expression emitters.
void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  // auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  // Value result = op->getResult(0);
  // fixUnsignedType(result, op->hasAttr("unsigned"));
  // emitValue(result, rank);
  // os << " = ";
  // emitValue(op->getOperand(0), rank);
  // os << " " << syntax << " ";
  // emitValue(op->getOperand(1), rank);
  os << "ADDF ";
  os << op << ";";
  emitInfoAndNewLine(op);
  // emitNestedLoopTail(rank);
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();
  os << "\n";
}

void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    // TODO: uncommenting yields errors
    // emitError(&op, "can't be correctly emitted.");
    // os << op << "\n";
  }
}

void ModuleEmitter::emitArrayDecl(Value array, bool isFunc, std::string name) {
  assert(!isDeclared(array) && "has been declared before.");

  auto arrayType = array.getType().cast<ShapedType>();
  if (arrayType.hasStaticShape()) {
    auto memref = array.getType().dyn_cast<MemRefType>();
    if (memref) {
      // auto attr = memref.getMemorySpace();
      // if (attr &&
      //     attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
      //   // Value has been declared before or is a constant number.
      //   if (isDeclared(array)) {
      //     os << getName(array);
      //     return;
      //   }
      //   // print stream type
      //   os << "hls::stream< " << getTypeName(array) << " > ";
      //   if (isFunc) {
      //     os << "&"; // pass by reference
      //   }
      //   // Add the new value to nameTable and emit its name.
      //   os << addName(array, /*isPtr=*/false, name);
      //   // Add original array declaration as comment
      //   os << " /* ";
      //   emitValue(array, 0, false, name);
      //   for (auto &shape : arrayType.getShape())
      //     os << "[" << shape << "]";
      //   os << " */";
      // }
      if (false) {
        // TODO
      } else {
        // emitValue(array, 0, false, name);
        os << "%" << addName(array, false) << ": " << memref;
        // os << ": " << memref;
        // if (arrayType.getShape().size() == 1 && arrayType.getShape()[0] == 1) {
        //   // do nothing;
        // } else {
        //   for (auto &shape : arrayType.getShape())
        //     os << "[" << shape << "]";
        // }
      }
    } 
    //else { // tensor
    //   emitValue(array, 0, false, name);
    // }
  } 
  // else
  //   emitValue(array, /*rank=*/0, /*isPtr=*/true, name);
}

void ModuleEmitter::emitFunction(FuncOp func) {
  if (func->hasAttr("bit"))
    BIT_FLAG = true;

  // if (func.getBlocks().size() != 1)
  //   emitError(func, "has zero or more than one basic blocks.");

  // if (func->hasAttr("top"))
  //   os << "/// This is top function.\n";

  // Emit function signature.
  addIndent();
  indent();
  os << "func @" << func.getName() << " (";

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit function arguments.
  unsigned int argIdx = 0;
  // std::vector<std::string> input_args;
  // if (func->hasAttr("inputs")) {
  //   std::string input_names =
  //       func->getAttr("inputs").cast<StringAttr>().getValue().str();
  //   input_args = split_names(input_names);
  // }
  // std::string output_names;
  // if (func->hasAttr("outputs")) {
  //   output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
  //   // suppose only one output
  //   input_args.push_back(output_names);
  // }

  for (auto &arg : func.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      // TODO: for now, input_args.size() == 0
      // if (input_args.size() == 0) {
      emitArrayDecl(arg, true);
      // } else {
        // emitArrayDecl(arg, true, input_args[argIdx]);
      // }
    } else {
      os << "N/A\n";
    }

    if (argIdx++ != func.getNumArguments() - 1)
      os << ", ";

    portList.push_back(arg);
  }

  os << ") {";
  emitInfoAndNewLine(func);
  os << "\n";

  // Emit function body
  addIndent();
  indent();
  os << "=====BEGIN BLOCK EMIT=====\n";
  emitBlock(func.front());
  indent();
  os << "=====END BLOCK EMIT=====\n\n";

  /// TODO: hard-coded codegen
  indent();
  os << "***HARDCODED***\n";
  indent();
  os << "// Define grid and block dimensions\n";

  indent(); os << "%gDimX = constant [0-9]+ : index\n";
  indent(); os << "%gDimY = constant [0-9]+ : index\n";
  indent(); os << "%gDimZ = constant [0-9]+ : index\n";
  indent(); os << "%bDimX = constant [0-9]+ : index\n";
  indent(); os << "%bDimY = constant [0-9]+ : index\n";
  indent(); os << "%bDimZ = constant [0-9]+ : index\n";

  indent();
  os << "// Register data to device\n";
  for (auto &arg : func.getArguments()) {
    indent();
    auto name = getName(arg);
    os << "%cast_" << name;
    os << " = ";
    os << "memref.cast %" << name;
    os << " : " << "memref<ranked> to memref<unranked>";
    os << "\n";
    indent();
    os << "gpu.host_register %cast_" << name;
    os << " : " << "memref<unranked>";
    os << "\n";
  }
  os << "\n";
  indent();
  os << "// GPU Kernel definition\n";
  indent();
  os << "gpu.launch ";
  os << "blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY, %grid_z = %gDimZ)\n";
  indent();
  os << "          threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY, %block_z = %bDimZ) {\n";
  addIndent();
  indent();
  os << "// Kernel body\n";
  indent(); os << "%a = memref.load %src0[%tx] : memref<ranked>\n";
  indent(); os << "%b = memref.load %src1[%tx] : memref<ranked>\n";
  indent(); os << "%sum = arith.addf %a, %b : f32\n";
  indent(); os << "memref.store %sum, %dest[%tx] : memref<ranked>\n";
  indent(); os << "gpu.terminate\n";
  reduceIndent();
  indent();
  os << "}\n";
  indent();
  os << "***HARDCODED***\n\n";
  /// END TODO

  indent();
  os << "return\n";
  reduceIndent();
  indent();
  os << "}\n";

  // End module
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string run_instr = R"XXX(// RUN: hcl-opt -opt %s | FileCheck %s)XXX";
  os << run_instr << "\n\n";
  std::string module_header = R"XXX(module {)XXX";
  os << module_header << "\n";
  for (auto op : module.getOps<FuncOp>()) {
    emitFunction(op);
  }
}

//===----------------------------------------------------------------------===//
// Entry of hcl-translate
//===----------------------------------------------------------------------===//

LogicalResult hcl::emitMlirGpu(ModuleOp module, llvm::raw_ostream &os) {
  HCLEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void hcl::registerEmitMlirGpuTranslation() {
  static TranslateFromMLIRRegistration toMlirGpu(
    "emit-gpu", emitMlirGpu, [&](DialectRegistry &registry) {
      // clang-format off
      registry.insert<
        mlir::hcl::HeteroCLDialect,
        mlir::StandardOpsDialect,
        mlir::arith::ArithmeticDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect,
        mlir::AffineDialect,
        mlir::math::MathDialect,
        mlir::memref::MemRefDialect
      >();
      // clang-format on
    });
}
