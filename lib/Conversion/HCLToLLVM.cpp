#include "hcl/Conversion/HCLToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace hcl;

namespace {
class CreateLoopHandleOpLowering : public ConversionPattern {
public:
  explicit CreateLoopHandleOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::CreateLoopHandleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class CreateStageHandleOpLowering : public ConversionPattern {
public:
  explicit CreateStageHandleOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::CreateStageHandleOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
struct HCLToLLVMLoweringPass
    : public PassWrapper<HCLToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const final { return "hcl-lower-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower HeteroCL dialect to LLVM dialect.";
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
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&context);
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateStageHandleOpLowering>(&context);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    return false;
  return true;
}
} // namespace hcl
} // namespace mlir

void HCLToLLVMLoweringPass::runOnOperation() {
  auto module = getOperation();
  if (!applyHCLToLLVMLoweringPass(module, getContext()))
    signalPassFailure();
}

namespace mlir {
namespace hcl {

void registerHCLToLLVMLoweringPass() {
  PassRegistration<HCLToLLVMLoweringPass>();
}

std::unique_ptr<mlir::Pass> createHCLToLLVMLoweringPass() {
  return std::make_unique<HCLToLLVMLoweringPass>();
}

} // namespace hcl
} // namespace mlir