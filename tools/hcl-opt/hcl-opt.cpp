//===- hcl-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "hcl/Dialect/HeteroCLDialect.h"

#include "hcl/Conversion/HCLToLLVM.h"
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

static llvm::cl::opt<bool> lowerToLLVM("lower-to-llvm",
                                       llvm::cl::desc("Lower to LLVM Dialect"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> lowerComposite("lower-composite",
                                       llvm::cl::desc("Lower composite types"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> lowerBitOps("lower-bitops",
                                       llvm::cl::desc("Lower bitops"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> legalizeCast("legalize-cast",
                                        llvm::cl::desc("Legalize cast"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool>
    enableNormalize("normalize",
                    llvm::cl::desc("Enable other common optimizations"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> runJiT("jit", llvm::cl::desc("Run JiT compiler"),
                                  llvm::cl::init(false));

static llvm::cl::opt<bool> fixedPointToInteger(
    "fixed-to-integer",
    llvm::cl::desc("Lower fixed-point operations to integer"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    anyWidthInteger("lower-anywidth-integer",
                    llvm::cl::desc("Lower anywidth integer to 64-bit integer"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool> moveReturnToInput(
    "return-to-input",
    llvm::cl::desc("Move return values to input argument list"),
    llvm::cl::init(false));

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  module = parseSourceFile(inputFilename, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int runJiTCompiler(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("top");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
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
  mlir::hcl::registerHCLPasses();
  mlir::hcl::registerHCLToLLVMLoweringPass();

  // Parse pass names in main to ensure static initialization completed
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadMLIR(context, module))
    return error;

  // Initialize a pass manager
  // https://mlir.llvm.org/docs/PassManagement/
  // Operation agnostic passes
  mlir::PassManager pm(&context);
  // Operation specific passes
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  if (enableOpt) {
    pm.addPass(mlir::hcl::createLoopTransformationPass());
  }

  if (lowerComposite) {
    pm.addPass(mlir::hcl::createLowerCompositeTypePass());
  }

  if (fixedPointToInteger) {
    pm.addPass(mlir::hcl::createFixedPointToIntegerPass());
  }

  if (anyWidthInteger) {
    pm.addPass(mlir::hcl::createAnyWidthIntegerPass());
  }

  if (moveReturnToInput) {
    pm.addPass(mlir::hcl::createMoveReturnToInputPass());
  }

  if (lowerBitOps) {
    pm.addPass(mlir::hcl::createLowerBitOpsPass());
  }

  if (legalizeCast) {
    pm.addPass(mlir::hcl::createLegalizeCastPass());
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
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());

    // Generic common sub expression elimination.
    // pm.addPass(mlir::createCSEPass());
  }

  if (runJiT || lowerToLLVM) {
    pm.addPass(mlir::hcl::createHCLToLLVMLoweringPass());
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

  // run JiT
  if (runJiT)
    return runJiTCompiler(*module);

  return 0;
}