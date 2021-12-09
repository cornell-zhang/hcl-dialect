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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLPasses.h"

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

static llvm::cl::opt<bool> preloadDialectsInContext(
    "preload-dialects-in-context",
    llvm::cl::desc("Preloads dialects in context"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpt(
    "opt", llvm::cl::desc("Enable optimizations"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  
  // Register dialects and passes in current context
  mlir::registerAllDialects(context);
  mlir::registerAllPasses();
  context.getOrLoadDialect<mlir::hcl::HeteroCLDialect>();
  mlir::hcl::registerHCLLoopReorderPass();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");


  // Set up the input output file file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // initialize source manager
  llvm::SourceMgr sourceMgr;
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  // Parse input mlir assembly file
  mlir::OwningModuleRef module(mlir::parseSourceFile(sourceMgr, &context));
  // Initialize a pass manager
  mlir::PassManager pm(&context);
  // Add desired passes
  if (enableOpt) {
    // Add operation agnostic passes here
    // e.g. pm.addPass(mlir::hcl::createMyPass());

    // Add operation specific passes here
    // e.g. 
    // mlir::OpPassManager &optPM = pm.nest<mlir::AffineForOp>();
    // optPM.addPass(mlir::hcl::createMyPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::hcl::createHCLLoopReorderPass());
  }
  // Run the pass pipeline
  if (mlir::failed(pm.run(*module))){
    return 4;
  }

  // print output
  module->print(output->os());
  output->os() << "\n";
  return 0;
}
