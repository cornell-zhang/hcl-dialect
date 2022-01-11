//===- hcl-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Affine/Passes.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Transforms/Passes.h"

#include <iostream>

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    preloadDialectsInContext("preload-dialects-in-context",
                             llvm::cl::desc("Preloads dialects in context"),
                             llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpt("opt",
                                     llvm::cl::desc("Enable HCL schedules"),
                                     llvm::cl::init(false));

static llvm::cl::opt<bool>
    enableNormalize("normalize",
                    llvm::cl::desc("Enable other common optimizations"),
                    llvm::cl::init(false));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  // Set up the input and output file
  std::string errorMessage;
  auto infile = mlir::openInputFile(inputFilename, &errorMessage);
  if (!infile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Parse the input MLIR
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(infile), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int main(int argc, char **argv) {

  // Register dialects and passes in current context
  mlir::MLIRContext context;
  auto registry = context.getDialectRegistry();
  mlir::registerAllDialects(context);
  context.allowUnregisteredDialects(true);
  context.printOpOnDiagnostic(true);
  context.loadAllAvailableDialects();
  context.getOrLoadDialect<mlir::hcl::HeteroCLDialect>();
  mlir::registerAllPasses();
  mlir::hcl::registerHCLLoopTransformationPass();

  // Parse pass names in main to ensure static initialization completed
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::OwningModuleRef module;
  if (int error = loadMLIR(context, module))
    return error;

  // Initialize a pass manager
  // https://mlir.llvm.org/docs/PassManagement/
  // Operation agnostic passes
  mlir::PassManager pm(&context);
  // Operation specific passes
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (enableOpt) {
    optPM.addPass(mlir::hcl::createHCLLoopTransformationPass());
  }

  if (enableNormalize) {
    // To make all loop steps to 1.
    optPM.addPass(mlir::createAffineLoopNormalizePass());

    // Sparse Conditional Constant Propagation (SCCP)
    pm.addPass(mlir::createSCCPPass());

    // To factor out the redundant AffineApply/AffineIf operations.
    // optPM.addPass(mlir::createCanonicalizerPass());
    // optPM.addPass(mlir::createSimplifyAffineStructuresPass());

    // To simplify the memory accessing.
    pm.addPass(mlir::createNormalizeMemRefsPass());

    // Generic common sub expression elimination.
    // pm.addPass(mlir::createCSEPass());
  }

  // Run the pass pipeline
  if (mlir::failed(pm.run(*module))) {
    return 4;
  }

  // print output
  std::string errorMessage;
  auto outfile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outfile) {
    llvm::errs() << errorMessage << "\n";
    return 2;
  }
  module->print(outfile->os());
  outfile->os() << "\n";
  return 0;
}