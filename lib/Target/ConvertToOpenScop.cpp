//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the Polymer project [https://github.com/kumasento/polymer]
//
//===----------------------------------------------------------------------===//
//===- EmitOpenScop.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

/* DEBJIT
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"
#include "polymer/Transforms/ExtractScopStmt.h"
*/

#include "hcl/Target/OpenScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/InitAllDialects.h" 
#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

/* Debjit
#include "osl/osl.h"
*/

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace hcl;
/* DEBJIT
using namespace polymer;

#define DEBUG_TYPE "oslscop"
*/

//Debjitnamespace {
//Debjit
//Debjit/// Build OslScop from FuncOp.
//Debjitclass OslScopBuilder {
//Debjitpublic:
//Debjit  OslScopBuilder() {}
//Debjit
//Debjit  /// Build a scop from a common FuncOp.
//Debjit  std::unique_ptr<OslScop> build(mlir::FuncOp f);
//Debjit
//Debjitprivate:
//Debjit  /// Find all statements that calls a scop.stmt.
//Debjit  void buildScopStmtMap(mlir::FuncOp f, OslScop::ScopStmtNames *scopStmtNames,
//Debjit                        OslScop::ScopStmtMap *scopStmtMap) const;
//Debjit
//Debjit  /// Build the scop context. The domain of each scop stmt will be updated, by
//Debjit  /// merging and aligning its IDs with the context as well.
//Debjit  void buildScopContext(OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
//Debjit                        FlatAffineValueConstraints &ctx) const;
//Debjit};
//Debjit
//Debjit} // namespace
//Debjit
//Debjit/// Sometimes the domain generated might be malformed. It is always better to
//Debjit/// inform this at an early stage.
//Debjitstatic void sanityCheckDomain(FlatAffineValueConstraints &dom) {
//Debjit  if (dom.isEmpty()) {
//Debjit    llvm::errs() << "A domain is found to be empty!";
//Debjit    dom.dump();
//Debjit  }
//Debjit}
//Debjit
//Debjit/// Build OslScop from a given FuncOp.
//Debjitstd::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {
//Debjit
//Debjit  /// Context constraints.
//Debjit  FlatAffineValueConstraints ctx;
//Debjit
//Debjit  // Initialize a new Scop per FuncOp. The osl_scop object within it will be
//Debjit  // created. It doesn't contain any fields, and this may incur some problems,
//Debjit  // which the validate function won't discover, e.g., no context will cause
//Debjit  // segfault when printing scop. Please don't just return this object.
//Debjit  auto scop = std::make_unique<OslScop>();
//Debjit  // Mapping between scop stmt names and their caller/callee op pairs.
//Debjit  OslScop::ScopStmtMap *scopStmtMap = scop->getScopStmtMap();
//Debjit  auto *scopStmtNames = scop->getScopStmtNames();
//Debjit
//Debjit  // Find all caller/callee pairs in which the callee has the attribute of name
//Debjit  // SCOP_STMT_ATTR_NAME.
//Debjit  buildScopStmtMap(f, scopStmtNames, scopStmtMap);
//Debjit  if (scopStmtMap->empty())
//Debjit    return nullptr;
//Debjit
//Debjit  // Build context in it.
//Debjit  buildScopContext(scop.get(), scopStmtMap, ctx);
//Debjit
//Debjit  // Counter for the statement inserted.
//Debjit  unsigned stmtId = 0;
//Debjit  for (const auto &scopStmtName : *scopStmtNames) {
//Debjit    const ScopStmt &stmt = scopStmtMap->find(scopStmtName)->second;
//Debjit    LLVM_DEBUG({
//Debjit      dbgs() << "Adding relations to statement: \n";
//Debjit      stmt.getCaller().dump();
//Debjit    });
//Debjit
//Debjit    // Collet the domain
//Debjit    FlatAffineValueConstraints domain = *stmt.getDomain();
//Debjit    sanityCheckDomain(domain);
//Debjit
//Debjit    LLVM_DEBUG({
//Debjit      dbgs() << "Domain:\n";
//Debjit      domain.dump();
//Debjit    });
//Debjit
//Debjit    // Collect the enclosing ops.
//Debjit    llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
//Debjit    stmt.getEnclosingOps(enclosingOps);
//Debjit    // Get the callee.
//Debjit    mlir::FuncOp callee = stmt.getCallee();
//Debjit
//Debjit    LLVM_DEBUG({
//Debjit      dbgs() << "Callee:\n";
//Debjit      callee.dump();
//Debjit    });
//Debjit
//Debjit    // Create a statement in OslScop and setup relations in it.
//Debjit    scop->createStatement();
//Debjit    scop->addDomainRelation(stmtId, domain);
//Debjit    scop->addScatteringRelation(stmtId, domain, enclosingOps);
//Debjit    callee.walk([&](mlir::Operation *op) {
//Debjit      if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
//Debjit        LLVM_DEBUG(dbgs() << "Creating access relation for: " << *op << '\n');
//Debjit
//Debjit        bool isRead = isa<mlir::AffineReadOpInterface>(op);
//Debjit        AffineValueMap vMap;
//Debjit        mlir::Value memref;
//Debjit
//Debjit        stmt.getAccessMapAndMemRef(op, &vMap, &memref);
//Debjit        scop->addAccessRelation(stmtId, isRead, memref, vMap, domain);
//Debjit      }
//Debjit    });
//Debjit
//Debjit    stmtId++;
//Debjit  }
//Debjit
//Debjit  // Setup the symbol table within the OslScop, which builds the mapping from
//Debjit  // mlir::Value to their names in the OpenScop representation, and maps them
//Debjit  // backward.
//Debjit  scop->initializeSymbolTable(f, &ctx);
//Debjit
//Debjit  // Insert body extension.
//Debjit  for (unsigned stmtId = 0; stmtId < scopStmtNames->size(); stmtId++) {
//Debjit    const ScopStmt &stmt = scopStmtMap->find(scopStmtNames->at(stmtId))->second;
//Debjit    scop->addBodyExtension(stmtId, stmt);
//Debjit  }
//Debjit  assert(scop->validate() && "The scop object created cannot be validated.");
//Debjit
//Debjit  // Additionally, setup the name of the function in the comment.
//Debjit  std::string funcName(f.getName());
//Debjit  scop->addExtensionGeneric("comment", funcName);
//Debjit
//Debjit  assert(scop->validate() && "The scop object created cannot be validated.");
//Debjit
//Debjit  return scop;
//Debjit}
//Debjit
//Debjit/// Find all statements that calls a scop.stmt.
//Debjitvoid OslScopBuilder::buildScopStmtMap(mlir::FuncOp f,
//Debjit                                      OslScop::ScopStmtNames *scopStmtNames,
//Debjit                                      OslScop::ScopStmtMap *scopStmtMap) const {
//Debjit  mlir::ModuleOp m = cast<mlir::ModuleOp>(f->getParentOp());
//Debjit
//Debjit  f.walk([&](mlir::Operation *op) {
//Debjit    if (mlir::CallOp caller = dyn_cast<mlir::CallOp>(op)) {
//Debjit      std::string calleeName(caller.getCallee());
//Debjit      mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(calleeName);
//Debjit
//Debjit      // If the callee is of scop.stmt, we create a new instance in the map
//Debjit      if (callee->getAttr(SCOP_STMT_ATTR_NAME)) {
//Debjit        scopStmtNames->push_back(std::string(calleeName));
//Debjit        scopStmtMap->insert(
//Debjit            std::make_pair(calleeName, ScopStmt(caller, callee)));
//Debjit      }
//Debjit    }
//Debjit  });
//Debjit}
//Debjit
//Debjitvoid OslScopBuilder::buildScopContext(OslScop *scop,
//Debjit                                      OslScop::ScopStmtMap *scopStmtMap,
//Debjit                                      FlatAffineValueConstraints &ctx) const {
//Debjit  LLVM_DEBUG(dbgs() << "--- Building SCoP context ...\n");
//Debjit
//Debjit  // First initialize the symbols of the ctx by the order of arg number.
//Debjit  // This simply aims to make mergeAndAlignIdsWithOthers work.
//Debjit  SmallVector<Value> symbols;
//Debjit  for (const auto &it : *scopStmtMap) {
//Debjit    auto domain = it.second.getDomain();
//Debjit    SmallVector<Value> syms;
//Debjit    domain->getValues(domain->getNumDimIds(), domain->getNumDimAndSymbolIds(),
//Debjit                      &syms);
//Debjit
//Debjit    for (Value sym : syms) {
//Debjit      // Find the insertion position.
//Debjit      auto it = symbols.begin();
//Debjit      while (it != symbols.end()) {
//Debjit        auto lhs = it->cast<BlockArgument>();
//Debjit        auto rhs = sym.cast<BlockArgument>();
//Debjit        if (lhs.getArgNumber() >= rhs.getArgNumber())
//Debjit          break;
//Debjit        ++it;
//Debjit      }
//Debjit      if (*it != sym)
//Debjit        symbols.insert(it, sym);
//Debjit    }
//Debjit  }
//Debjit  ctx.reset(/*numDims=*/0, /*numSymbols=*/symbols.size());
//Debjit  ctx.setValues(0, symbols.size(), symbols);
//Debjit
//Debjit  // Union with the domains of all Scop statements. We first merge and align the
//Debjit  // IDs of the context and the domain of the scop statement, and then append
//Debjit  // the constraints from the domain to the context. Note that we don't want to
//Debjit  // mess up with the original domain at this point. Trivial redundant
//Debjit  // constraints will be removed.
//Debjit  for (const auto &it : *scopStmtMap) {
//Debjit    FlatAffineValueConstraints *domain = it.second.getDomain();
//Debjit    FlatAffineValueConstraints cst(*domain);
//Debjit
//Debjit    LLVM_DEBUG(dbgs() << "Statement:\n");
//Debjit    LLVM_DEBUG(it.second.getCaller().dump());
//Debjit    LLVM_DEBUG(it.second.getCallee().dump());
//Debjit    LLVM_DEBUG(dbgs() << "Target domain: \n");
//Debjit    LLVM_DEBUG(domain->dump());
//Debjit
//Debjit    LLVM_DEBUG({
//Debjit      dbgs() << "Domain values: \n";
//Debjit      SmallVector<Value> values;
//Debjit      domain->getValues(0, domain->getNumDimAndSymbolIds(), &values);
//Debjit      for (Value value : values)
//Debjit        dbgs() << " * " << value << '\n';
//Debjit    });
//Debjit
//Debjit    ctx.mergeAndAlignIdsWithOther(0, &cst);
//Debjit    ctx.append(cst);
//Debjit    ctx.removeRedundantConstraints();
//Debjit
//Debjit    LLVM_DEBUG(dbgs() << "Updated context: \n");
//Debjit    LLVM_DEBUG(ctx.dump());
//Debjit
//Debjit    LLVM_DEBUG({
//Debjit      dbgs() << "Context values: \n";
//Debjit      SmallVector<Value> values;
//Debjit      ctx.getValues(0, ctx.getNumDimAndSymbolIds(), &values);
//Debjit      for (Value value : values)
//Debjit        dbgs() << " * " << value << '\n';
//Debjit    });
//Debjit  }
//Debjit
//Debjit  // Then, create the single context relation in scop.
//Debjit  scop->addContextRelation(ctx);
//Debjit
//Debjit  // Finally, given that ctx has all the parameters in it, we will make sure
//Debjit  // that each domain is aligned with them, i.e., every domain has the same
//Debjit  // parameter columns (Values & order).
//Debjit  SmallVector<mlir::Value, 8> symValues;
//Debjit  ctx.getValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(), &symValues);
//Debjit
//Debjit  // Add and align domain SYMBOL columns.
//Debjit  for (const auto &it : *scopStmtMap) {
//Debjit    FlatAffineValueConstraints *domain = it.second.getDomain();
//Debjit    // For any symbol missing in the domain, add them directly to the end.
//Debjit    for (unsigned i = 0; i < ctx.getNumSymbolIds(); ++i) {
//Debjit      unsigned pos;
//Debjit      if (!domain->findId(symValues[i], &pos)) // insert to the back
//Debjit        domain->appendSymbolId(symValues[i]);
//Debjit      else
//Debjit        LLVM_DEBUG(dbgs() << "Found " << symValues[i] << '\n');
//Debjit    }
//Debjit
//Debjit    // Then do the aligning.
//Debjit    LLVM_DEBUG(domain->dump());
//Debjit    for (unsigned i = 0; i < ctx.getNumSymbolIds(); i++) {
//Debjit      mlir::Value sym = symValues[i];
//Debjit      unsigned pos;
//Debjit      assert(domain->findId(sym, &pos));
//Debjit
//Debjit      unsigned posAsCtx = i + domain->getNumDimIds();
//Debjit      LLVM_DEBUG(dbgs() << "Swapping " << posAsCtx << " " << pos << "\n");
//Debjit      if (pos != posAsCtx)
//Debjit        domain->swapId(posAsCtx, pos);
//Debjit    }
//Debjit
//Debjit    // for (unsigned i = 0; i < ctx.getNumSymbolIds(); i++) {
//Debjit    //   mlir::Value sym = symValues[i];
//Debjit    //   unsigned pos;
//Debjit    //   // If the symbol can be found in the domain, we put it in the same
//Debjit    //   // position as the ctx.
//Debjit    //   if (domain->findId(sym, &pos)) {
//Debjit    //     if (pos != i + domain->getNumDimIds())
//Debjit    //       domain->swapId(i + domain->getNumDimIds(), pos);
//Debjit    //   } else {
//Debjit    //     domain->insertSymbolId(i, sym);
//Debjit    //   }
//Debjit    // }
//Debjit  }
//Debjit}
//Debjit
//Debjitstd::unique_ptr<OslScop>
//Debjitpolymer::createOpenScopFromFuncOp(mlir::FuncOp f, OslSymbolTable &symTable) {
//Debjit  return OslScopBuilder().build(f);
//Debjit}
//Debjit
//Debjitnamespace {
//Debjit
//Debjit/// This class maintains the state of a working emitter.
//Debjitclass OpenScopEmitterState {
//Debjitpublic:
//Debjit  explicit OpenScopEmitterState(raw_ostream &os) : os(os) {}
//Debjit
//Debjit  /// The stream to emit to.
//Debjit  raw_ostream &os;
//Debjit
//Debjit  bool encounteredError = false;
//Debjit  unsigned currentIdent = 0; // TODO: may not need this.
//Debjit
//Debjitprivate:
//Debjit  OpenScopEmitterState(const OpenScopEmitterState &) = delete;
//Debjit  void operator=(const OpenScopEmitterState &) = delete;
//Debjit};
//Debjit
//Debjit/// Base class for various OpenScop emitters.
//Debjitclass OpenScopEmitterBase {
//Debjitpublic:
//Debjit  explicit OpenScopEmitterBase(OpenScopEmitterState &state)
//Debjit      : state(state), os(state.os) {}
//Debjit
//Debjit  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
//Debjit    state.encounteredError = true;
//Debjit    return op->emitError(message);
//Debjit  }
//Debjit
//Debjit  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
//Debjit    state.encounteredError = true;
//Debjit    return op->emitOpError(message);
//Debjit  }
//Debjit
//Debjit  /// All of the mutable state we are maintaining.
//Debjit  OpenScopEmitterState &state;
//Debjit
//Debjit  /// The stream to emit to.
//Debjit  raw_ostream &os;
//Debjit
//Debjitprivate:
//Debjit  OpenScopEmitterBase(const OpenScopEmitterBase &) = delete;
//Debjit  void operator=(const OpenScopEmitterBase &) = delete;
//Debjit};
//Debjit
//Debjit/// Emit OpenScop representation from an MLIR module.
//Debjitclass ModuleEmitter : public OpenScopEmitterBase {
//Debjitpublic:
//Debjit  explicit ModuleEmitter(OpenScopEmitterState &state)
//Debjit      : OpenScopEmitterBase(state) {}
//Debjit
//Debjit  /// Emit OpenScop definitions for all functions in the given module.
//Debjit  void emitMLIRModule(ModuleOp module,
//Debjit                      llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);
//Debjit
//Debjitprivate:
//Debjit  /// Emit a OpenScop definition for a single function.
//Debjit  LogicalResult
//Debjit  emitFuncOp(FuncOp func,
//Debjit             llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);
//Debjit};
//Debjit
//DebjitLogicalResult ModuleEmitter::emitFuncOp(
//Debjit    mlir::FuncOp func, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
//Debjit  OslSymbolTable symTable;
//Debjit  auto scop = createOpenScopFromFuncOp(func, symTable);
//Debjit  if (scop)
//Debjit    scops.push_back(std::move(scop));
//Debjit  return success();
//Debjit}
//Debjit
//Debjit/// The entry function to the current OpenScop emitter.
//Debjitvoid ModuleEmitter::emitMLIRModule(
//Debjit    ModuleOp module, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
//Debjit  // Emit a single OpenScop definition for each function.
//Debjit  for (auto &op : *module.getBody()) {
//Debjit    if (auto func = dyn_cast<mlir::FuncOp>(op)) {
//Debjit      // Will only look at functions that are not attributed as scop.stmt
//Debjit      if (func->getAttr(SCOP_STMT_ATTR_NAME))
//Debjit        continue;
//Debjit      if (failed(emitFuncOp(func, scops))) {
//Debjit        state.encounteredError = true;
//Debjit        return;
//Debjit      }
//Debjit    }
//Debjit  }
//Debjit}
//Debjit} // namespace

/// TODO: should decouple emitter and openscop builder.
//Debjitmlir::LogicalResult hcl::translateModuleToOpenScop(
//Debjit    mlir::ModuleOp module,
//Debjit    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
//Debjit    llvm::raw_ostream &os) {
//Debjit  OpenScopEmitterState state(os);
//Debjit  ::ModuleEmitter(state).emitMLIRModule(module, scops);
//Debjit
//Debjit  return success();
//Debjit}

LogicalResult hcl::emitOpenScop(ModuleOp module, llvm::raw_ostream &os) {
  std::cout << "Link up done\n" << "\n";
  /*
  llvm::SmallVector<std::unique_ptr<OslScop>, 8> scops;

  if (failed(translateModuleToOpenScop(module, scops, os)))
    return failure();

  for (auto &scop : scops)
    scop->print();
  */

  return success();
}

void hcl::registerToOpenScopTranslation() {
  static TranslateFromMLIRRegistration toOpenScop("export-scop", emitOpenScop, [&](DialectRegistry &registry){
        registry.insert<mlir::hcl::HeteroCLDialect, mlir::StandardOpsDialect,
                        tensor::TensorDialect, mlir::AffineDialect,
                        mlir::math::MathDialect, mlir::memref::MemRefDialect>();
      });
}
