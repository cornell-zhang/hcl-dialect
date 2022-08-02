#include "hcl/Conversion/CommonPatterns.h"
#include "hcl/Conversion/HCLToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include <deque>
#include <iostream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "loops-to-gpu"

using namespace mlir;
using namespace hcl;

/// ----------------------------------------------------------------------------
///  Rewrite Patterns: lower affine load/store to memref load/store
/// ----------------------------------------------------------------------------
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

/// ----------------------------------------------------------------------------
///  Pass entry points: the apply functions
/// ----------------------------------------------------------------------------

namespace mlir {
namespace hcl {

bool applyGPUToNVVMLoweringPass(ModuleOp &module, MLIRContext &context) {

  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&context);
  RewritePatternSet patterns(&context);

  // target.addLegalDialect<scf::SCFDialect, StandardOpsDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateGpuToNVVMConversionPatterns(typeConverter, patterns);
  // populateGpuToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOpHandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return false;
  }
  return true;
}

bool applyAffineToGPULoweringPass(ModuleOp &module, MLIRContext &context) {

  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&context);
  RewritePatternSet patterns(&context);

  target.addLegalDialect<gpu::GPUDialect>();
  target.addLegalDialect<scf::SCFDialect, StandardOpsDialect>();

  unsigned numBlockDims = 1;
  unsigned numThreadDims = 1;
  module.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {
    std::cout << "in affine for loop" << std::endl;
    if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                numThreadDims))) {
      std::cout << "failed affine to gpu" << std::endl;
    }
  });

  populateReconcileUnrealizedCastsPatterns(patterns);
  patterns.add<AffineLoadLowering, AffineStoreLowering>(patterns.getContext());
  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOpHandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return false;
  }
  // finalizeParallelLoopToGPUConversion(module);
  return true;
}

} // namespace hcl
} // namespace mlir

/// ----------------------------------------------------------------------------
///  Pass interfaces: PassWrapper, PassRegistration
/// ----------------------------------------------------------------------------

namespace {
struct AffineToGPULoweringPass
    : public PassWrapper<AffineToGPULoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "scf-lower-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower SCF dialect to GPU dialect.";
  }
  void runOnOperation() final {
    auto module = getOperation();
    if (!applyAffineToGPULoweringPass(module, getContext()))
      signalPassFailure();
  }
};
struct GPUToNVVMLoweringPass
    : public PassWrapper<GPUToNVVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return "gpu-lower-to-nvvm"; }
  StringRef getDescription() const final {
    return "Lower SCF dialect to GPU dialect.";
  }

  void runOnOperation() final {
    auto module = getOperation();
    if (!applyGPUToNVVMLoweringPass(module, getContext()))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace hcl {

void registerAffineToGPULoweringPass() {
  PassRegistration<AffineToGPULoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineToGPULoweringPass() {
  return std::make_unique<AffineToGPULoweringPass>();
}

void registerGPUToNVVMLoweringPass() {
  PassRegistration<GPUToNVVMLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createGPUToNVVMLoweringPass() {
  return std::make_unique<GPUToNVVMLoweringPass>();
}

} // namespace hcl
} // namespace mlir