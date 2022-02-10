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

#include "hcl/Support/OslScop.h"
#include "hcl/Support/OslScopStmtOpSet.h"
#include "hcl/Support/OslSymbolTable.h"
#include "hcl/Support/ScopStmt.h"
#include "hcl/Target/OpenScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
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

#include "osl/osl.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace hcl;

namespace {

/// Build OslScop from FuncOp.
class OslScopBuilder {
public:
  OslScopBuilder() {}

  /// Build a scop from a common FuncOp.
  std::unique_ptr<OslScop> build(mlir::FuncOp f);

private:
  /// Find all statements that calls a scop.stmt.
  void buildScopStmtMap(mlir::FuncOp f, OslScop::ScopStmtNames *scopStmtNames,
                        OslScop::ScopStmtMap *scopStmtMap) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  void buildScopContext(OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
                        FlatAffineConstraints &ctx) const;
};

} // namespace

static FlatAffineConstraints mergeDomainAndContext(FlatAffineConstraints dom,
                                                   FlatAffineConstraints ctx) {
  FlatAffineConstraints cst(dom);

  SmallVector<mlir::Value, 4> dims;
  ctx.getIdValues(0, ctx.getNumDimIds(), &dims);

  for (mlir::Value dim : dims)
    ctx.projectOut(dim);
  ctx.removeIndependentConstraints(0, ctx.getNumDimAndSymbolIds());
  ctx.removeRedundantInequalities();
  ctx.removeRedundantConstraints();
  ctx.removeTrivialRedundancy();

  cst.mergeAndAlignIdsWithOther(0, &ctx);
  cst.append(ctx);

  return cst;
}

/// Build OslScop from a given FuncOp.
std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {

  /// Context constraints.
  FlatAffineConstraints ctx;
  // Initialize a new Scop per FuncOp. The osl_scop object within it will be
  // created. It doesn't contain any fields, and this may incur some problems,
  // which the validate function won't discover, e.g., no context will cause
  // segfault when printing scop. Please don't just return this object.
  auto scop = std::make_unique<OslScop>();
  // Mapping between scop stmt names and their caller/callee op pairs.
  OslScop::ScopStmtMap *scopStmtMap = scop->getScopStmtMap();
  auto *scopStmtNames = scop->getScopStmtNames();

  // Find all caller/callee pairs in which the callee has the attribute of name
  // SCOP_STMT_ATTR_NAME.

  buildScopStmtMap(f, scopStmtNames, scopStmtMap);
  if (scopStmtMap->empty())
    return nullptr;

  // Build context in it.
  buildScopContext(scop.get(), scopStmtMap, ctx);
  ctx.dump();

  // Counter for the statement inserted.
  unsigned stmtId = 0;
  for (const auto &scopStmtName : *scopStmtNames) {
    llvm::errs() << "Statement name: " << scopStmtName << "\n";
    const ScopStmt &stmt = scopStmtMap->find(scopStmtName)->second;

    // Collet the domain
    FlatAffineConstraints domain =
        mergeDomainAndContext(*stmt.getDomain(), ctx);
    // Collect the enclosing ops.
    llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
    stmt.getEnclosingOps(enclosingOps);
    // Get the callee.
    mlir::FuncOp callee = stmt.getCallee();

    // Create a statement in OslScop and setup relations in it.
    scop->createStatement();
    scop->addDomainRelation(stmtId, domain);
    scop->addScatteringRelation(stmtId, domain, enclosingOps);
    callee.walk([&](mlir::Operation *op) {
      if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
        bool isRead = isa<mlir::AffineReadOpInterface>(op);
        AffineValueMap vMap;
        mlir::Value memref;
        stmt.getAccessMapAndMemRef(op, &vMap, &memref);
        scop->addAccessRelation(stmtId, isRead, memref, vMap, domain);
      }
    });

    stmtId++;
  }

  // Setup the symbol table within the OslScop, which builds the mapping from
  // mlir::Value to their names in the OpenScop representation, and maps them
  // backward.
  scop->initializeSymbolTable(f, &ctx);

  // Insert body extension.
  for (unsigned stmtId = 0; stmtId < scopStmtNames->size(); stmtId++) {
    const ScopStmt &stmt = scopStmtMap->find(scopStmtNames->at(stmtId))->second;
    //osl_scop_print(stderr, scop->get());
    scop->addBodyExtension(stmtId, stmt);
  }
  assert(scop->validate() && "The scop object created cannot be validated.");

  // Additionally, setup the name of the function in the comment.
  std::string funcName(f.getName());
  scop->addExtensionGeneric("comment", funcName);

  //osl_scop_print(stderr, scop->get());
  assert(scop->validate() && "The scop object created cannot be validated.");
  return scop;
}

/// Find all statements that calls a scop.stmt.
void OslScopBuilder::buildScopStmtMap(mlir::FuncOp f,
                                      OslScop::ScopStmtNames *scopStmtNames,
                                      OslScop::ScopStmtMap *scopStmtMap) const {
  mlir::ModuleOp m = cast<mlir::ModuleOp>(f->getParentOp());

  f.walk([&](mlir::Operation *op) {
    if (mlir::CallOp caller = dyn_cast<mlir::CallOp>(op)) {
      std::string calleeName(caller.getCallee());
      mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(calleeName);

      // If the callee is of scop.stmt, we create a new instance in the map
      if (callee->getAttr(SCOP_STMT_ATTR_NAME)) {
        scopStmtNames->push_back(std::string(calleeName));
        scopStmtMap->insert(
            std::make_pair(calleeName, ScopStmt(caller, callee)));
      }
    }
  });
}

void OslScopBuilder::buildScopContext(OslScop *scop,
                                      OslScop::ScopStmtMap *scopStmtMap,
                                      FlatAffineConstraints &ctx) const {
  ctx.reset();

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (const auto &it : *scopStmtMap) {
    FlatAffineConstraints *domain = it.second.getDomain();
    FlatAffineConstraints cst(*domain);

    ctx.mergeAndAlignIdsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();
  }

  // Then, create the single context relation in scop.
  scop->addContextRelation(ctx);

  // Finally, given that ctx has all the parameters in it, we will make sure
  // that each domain is aligned with them, i.e., every domain has the same
  // parameter columns (Values & order).
  SmallVector<mlir::Value, 8> symValues;
  ctx.getIdValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(), &symValues);

  for (const auto &it : *scopStmtMap) {
    FlatAffineConstraints *domain = it.second.getDomain();

    for (unsigned i = 0; i < ctx.getNumSymbolIds(); i++) {
      mlir::Value sym = symValues[i];
      unsigned pos;
      if (domain->findId(sym, &pos)) {
        if (pos != i + domain->getNumDimIds())
          domain->swapId(i + domain->getNumDimIds(), pos);
      } else {
        domain->addSymbolId(i, sym);
      }
    }
  }
}

std::unique_ptr<OslScop>
hcl::createOpenScopFromFuncOp(FuncOp f, OslSymbolTable &symTable) {
  return OslScopBuilder().build(f);
}
