#include "hcl/Conversion/HCLToLLVM.h"
#include "hcl/Conversion/CommonPatterns.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace hcl;

namespace {
struct HCLToLLVMLoweringPass
    : public HCLToLLVMLoweringBase<HCLToLLVMLoweringPass> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!applyHCLToLLVMLoweringPass(module, getContext()))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {
bool applyHCLToLLVMLoweringPass(ModuleOp &module, MLIRContext &context) {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&context);

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `hcl`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&context);
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOpHandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    return false;
  return true;
}
} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {
// void registerHCLToLLVMLoweringPass() {
//   PassRegistration<HCLToLLVMLoweringPass>();
// }

std::unique_ptr<OperationPass<ModuleOp>> createHCLToLLVMLoweringPass() {
  return std::make_unique<HCLToLLVMLoweringPass>();
}
} // namespace hcl
} // namespace mlir
