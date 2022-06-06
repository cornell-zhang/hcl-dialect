#include "hcl/Conversion/HCLToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // If the PrintOp has string attribute, it is the format string
    std::string format_str = "%f \0";
    if (op->hasAttr("format")) {
      format_str = op->getAttr("format").cast<StringAttr>().getValue().str();
    }
    bool hasUnsignedAttr = op->hasAttr("unsigned");

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef(format_str), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<hcl::PrintOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.input(), loopIvs);
    // Cast element to f64
    auto casted = castToF64(rewriter, elementLoad, hasUnsignedAttr);
    rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, casted}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// To support printing MemRef with any element type, we cast
  /// Int, Float32 types to Float64.
  static Value castToF64(ConversionPatternRewriter &rewriter, const Value &src,
                         bool hasUnsignedAttr) {
    Type t = src.getType();
    Type F64Type = rewriter.getF64Type();
    Value casted;
    if (t.isa<IntegerType>()) {
      if (t.isUnsignedInteger() or hasUnsignedAttr) {
        casted = rewriter.create<arith::UIToFPOp>(src.getLoc(), F64Type, src);
      } else { // signed and signless integer
        casted = rewriter.create<arith::SIToFPOp>(src.getLoc(), F64Type, src);
      }
    } else if (t.isa<FloatType>()) {
      unsigned width = t.cast<FloatType>().getWidth();
      if (width < 64) {
        casted = rewriter.create<arith::ExtFOp>(src.getLoc(), F64Type, src);
      } else if (width > 64) {
        casted = rewriter.create<arith::TruncFOp>(src.getLoc(), F64Type, src);
      } else {
        casted = src;
      }
    } else {
      llvm::errs() << src.getLoc() << "could not cast value of type "
                   << src.getType() << " to F64.\n";
    }
    return casted;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

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

class SetIntBitOpLowering : public ConversionPattern {
public:
  explicit SetIntBitOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::SetIntBitOp::getOperationName(), 3, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // SetIntBitOp should be lowered to left shift and bitwise AND/OR
    Value input = operands[0];
    Value index = operands[1];
    Value val = operands[2];
    Location loc = op->getLoc();
    // Cast val to the same with as input
    unsigned width = input.getType().getIntOrFloatBitWidth();
    Value const_1 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, width);
    // Cast index to i32
    Type itype = rewriter.getIntegerType(width);
    Value idx_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, index, itype);
    Value bitmask =
        rewriter.create<mlir::arith::ShLIOp>(loc, const_1, idx_casted);
    // take the inverse of bitmask
    Value all_one_mask =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, width);
    Value inversed_mask =
        rewriter.create<mlir::arith::XOrIOp>(loc, all_one_mask, bitmask);
    // If val == 1, SetBit should be input OR bitmask (e.g. input || 000010000)
    Value Val1Res = rewriter.create<mlir::arith::OrIOp>(loc, input, bitmask);
    // If val == 0, SetBit should be input AND inversed bitmask
    // (e.g. input && 111101111)
    Value Val0Res =
        rewriter.create<mlir::arith::AndIOp>(loc, input, inversed_mask);
    Value trueRes = rewriter.create<SelectOp>(loc, val, Val1Res, Val0Res);
    op->getOperand(0).replaceAllUsesWith(trueRes);
    rewriter.eraseOp(op);
    return success();
  }
};

class GetIntBitOpLowering : public ConversionPattern {
public:
  explicit GetIntBitOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::GetIntBitOp::getOperationName(), 2, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // GetIntBitOp should be lowered to right shift and truncation
    Value input = operands[0];
    Value idx = operands[1];
    Location loc = op->getLoc();
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);
    Type i1 = rewriter.getI1Type();
    Value idx_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idx, itype);
    Value shifted =
        rewriter.create<mlir::arith::ShRSIOp>(loc, input, idx_casted);
    Value singleBit = rewriter.create<mlir::arith::TruncIOp>(loc, shifted, i1);
    op->getResult(0).replaceAllUsesWith(singleBit);
    rewriter.eraseOp(op);
    return success();
  }
};

class GetIntSliceOpLowering : public ConversionPattern {
public:
  explicit GetIntSliceOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::GetIntSliceOp::getOperationName(), 4, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = operands[0];
    Value hi = operands[1];
    Value lo = operands[2];
    // cast low and high index to int32 type
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Type itype = rewriter.getIntegerType(iwidth);
    Location loc = op->getLoc();
    Value lo_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, lo, itype);
    Value hi_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, hi, itype);
    Value width = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, input.getType().getIntOrFloatBitWidth() - 1, iwidth);
    Value lshift_width =
        rewriter.create<mlir::arith::SubIOp>(loc, width, hi_casted);
    // We do three shifts to extract the target bit slices
    Value shift1 =
        rewriter.create<mlir::arith::ShLIOp>(loc, input, lshift_width);
    Value shift2 =
        rewriter.create<mlir::arith::ShRUIOp>(loc, shift1, lshift_width);
    Value shift3 =
        rewriter.create<mlir::arith::ShRUIOp>(loc, shift2, lo_casted);
    op->getResult(0).replaceAllUsesWith(shift3);
    rewriter.eraseOp(op);
    return success();
  }
};

class SetIntSliceOpLowering : public ConversionPattern {
public:
  explicit SetIntSliceOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::SetIntSliceOp::getOperationName(), 4, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Three steps to implement setslice:
    // 1. Get higher slice
    // 2. Get lower slice
    // 3. Shift value and use bitwise OR to get result
    Value input = operands[0];
    Value hi = operands[1];
    Value lo = operands[2];
    Value val = operands[3];
    Location loc = op->getLoc();
    // Cast hi, lo to int32, cast val to same dtype as input
    // Note: val's width may be different than (hi-low+1), so
    // we need to clear the peripheral bits.
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    Value width =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, iwidth, iwidth);
    Type int_type = rewriter.getIntegerType(iwidth);
    Value lo_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, lo, int_type);
    Value hi_casted =
        rewriter.create<mlir::arith::IndexCastOp>(loc, hi, int_type);
    Value const1 =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, int_type);
    Value val_ext =
        rewriter.create<mlir::arith::ExtUIOp>(loc, val, input.getType());
    
    // Step 1: get higher slice - shift right, then shift left
    Value hi_shift_width =
        rewriter.create<mlir::arith::AddIOp>(loc, hi_casted, const1);
    Value hi_rshifted =
        rewriter.create<mlir::arith::ShRUIOp>(loc, input, hi_shift_width);
    Value hi_slice =
        rewriter.create<mlir::arith::ShLIOp>(loc, hi_rshifted, hi_shift_width);

    // Step 2: get lower slice - shift left, then shift right
    // Note: left shifting `width` bits would result in unchanged result
    // Therefore, we need to build a SelectOp:
    // shifted = shift < width ? lshift : zero
    Value shift_width =
        rewriter.create<mlir::arith::SubIOp>(loc, width, lo_casted);
    Value lo_lshifted =
        rewriter.create<mlir::arith::ShLIOp>(loc, input, shift_width);
    Value lo_slice_possible =
        rewriter.create<mlir::arith::ShRUIOp>(loc, lo_lshifted, shift_width);
    Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, iwidth);
    Value condition = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, shift_width, width);
    Value lo_slice =
        rewriter.create<SelectOp>(loc, condition, lo_slice_possible, zero);

    // Step 3: shift left val, and then use OR to "concat" three pieces
    Value val_shifted =
        rewriter.create<mlir::arith::ShLIOp>(loc, val_ext, lo_casted);
    Value peripheral_slices =
        rewriter.create<mlir::arith::OrIOp>(loc, hi_slice, lo_slice);
    Value res = rewriter.create<mlir::arith::OrIOp>(loc, peripheral_slices,
                                                    val_shifted);

    op->getOperand(0).replaceAllUsesWith(res);
    rewriter.eraseOp(op);
    return success();
  }
};

class BitReverseOpLowering : public ConversionPattern {
public:
  explicit BitReverseOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::BitReverseOp::getOperationName(), 1, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = operands[0];
    Location loc = op->getLoc();
    unsigned iwidth = input.getType().getIntOrFloatBitWidth();
    // Create two constants: number of bits, and zero
    Value const_0 =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, rewriter.getIndexType());
    Value const_width =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, iwidth, rewriter.getIndexType());
    
    
    // op->getResult(0).replaceAllUsesWith(res);
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
  // have a combination of `hcl`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&context);
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateStageHandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);
  patterns.add<BitReverseOpLowering>(&context);

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

std::unique_ptr<OperationPass<ModuleOp>> createHCLToLLVMLoweringPass() {
  return std::make_unique<HCLToLLVMLoweringPass>();
}

} // namespace hcl
} // namespace mlir
