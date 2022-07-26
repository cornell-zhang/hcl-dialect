#include "hcl/Conversion/HCLToLLVM.h"
#include <iostream>
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
#include "llvm/Support/Debug.h"
#include <deque>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "loops-to-gpu"


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

class CreateOphandleOpLowering : public ConversionPattern {
public:
  explicit CreateOphandleOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::CreateOpHandleOp::getOperationName(), 1,
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

/*
class GetIntSliceOpLowering : public ConversionPattern {
public:
  explicit GetIntSliceOpLowering(MLIRContext *context)
      : ConversionPattern(hcl::GetIntSliceOp::getOperationName(), 4, context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = operands[0];
    // Value hi = operands[1];
    Value lo = operands[2];
    // cast low and high index to int32 type
    Type i32 = rewriter.getI32Type();
    Location loc = op->getLoc();
    Value lo_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, lo, i32);
    // Value hi_casted = rewriter.create<mlir::arith::IndexCastOp>(loc, hi,
    // i32); Shift and truncate
    Type resType = op->getResult(0).getType();
    Value shifted =
        rewriter.create<mlir::arith::ShRSIOp>(loc, input, lo_casted);
    Value slice = rewriter.create<mlir::arith::TruncIOp>(loc, shifted, resType);
    op->getResult(0).replaceAllUsesWith(slice);
    rewriter.eraseOp(op);
    return success();
  }
};
*/

// Another way to implement GetIntSliceOp with just shifting
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

} // namespace


/// Affine to Memref Ops ////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////

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

namespace {
struct AffineMemOpParLoweringPass
    : public PassWrapper<AffineMemOpParLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const final { return "hcl-lower-to-scf"; }
  StringRef getDescription() const final {
    return "Lower HeteroCL dialect to SCF(Parallel) dialect.";
  }
};
} // namespace

namespace {
struct AffineToGPULoweringPass
    : public PassWrapper<AffineToGPULoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const final { return "scf-lower-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower SCF dialect to GPU dialect.";
  }
};
} // namespace

namespace {
struct GPUToNVVMLoweringPass
    : public PassWrapper<GPUToNVVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
  StringRef getArgument() const final { return "scf-lower-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower SCF dialect to GPU dialect.";
  }
};
} // namespace

namespace {
struct ParallelizationCandidate {
  ParallelizationCandidate(AffineForOp l, SmallVector<LoopReduction> &&r)
      : loop(l), reductions(std::move(r)) {}

  /// The potentially parallelizable loop.
  AffineForOp loop;
  /// Desciprtors of reductions that can be parallelized in the loop.
  SmallVector<LoopReduction> reductions;
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
  patterns.add<CreateOphandleOpLowering>(&context);
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
bool applyAffineMemOpParLoweringPass(ModuleOp &module, MLIRContext &context) {
  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&context);
  RewritePatternSet patterns(&context);
  target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect,scf::SCFDialect,gpu::GPUDialect, StandardOpsDialect>();
  
  
  patterns.add<AffineLoadLowering,AffineStoreLowering>(patterns.getContext());

/////////////////////////////////////////////////////////////////////////////////////////
// //Affine Parallelize
// unsigned maxNested = 16;
// std::vector<ParallelizationCandidate> parallelizableLoops;
//   module.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
//     SmallVector<LoopReduction> reductions;
//     if (isLoopParallel(loop, &reductions))
//       parallelizableLoops.emplace_back(loop, std::move(reductions));
//   });

//   for (const ParallelizationCandidate &candidate : parallelizableLoops) {
//     unsigned numParentParallelOps = 0;
//     AffineForOp loop = candidate.loop;
//     for (Operation *op = loop->getParentOp();
//          op != nullptr && !op->hasTrait<OpTrait::AffineScope>();
//          op = op->getParentOp()) {
//       if (isa<AffineParallelOp>(op))
//         ++numParentParallelOps;
//     }

//     if (numParentParallelOps < maxNested) {
//       if (failed(affineParallelize(loop, candidate.reductions))) {
//         return false;
//       }
//     } else {
//       return false;
//     }
//   }
// //module.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {convertAffineLoopNestToGPULaunch(loop, numBlockDims,numThreadDims);});
// populateAffineToStdConversionPatterns(patterns);
/////////////////////////////////////////////////////////////////////////////////////////////
  
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    {
      return false;
    }
  return true;
}
} // namespace hcl
} // namespace mlir


//////////////////*******Copy of Affine Loop nest to gpu with int instead of index ***********//////////////
namespace mlir {
namespace hcl {
static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

// Get the lower bound-related operands of a loop operation.
static Operation::operand_range getLowerBoundOperands(AffineForOp forOp) {
  return forOp.getLowerBoundOperands();
}

// Get the upper bound-related operands of a loop operation.
static Operation::operand_range getUpperBoundOperands(AffineForOp forOp) {
  return forOp.getUpperBoundOperands();
}

// Get a Value that corresponds to the loop step.  If the step is an attribute,
// materialize a corresponding constant using builder.
static Value getOrCreateStep(AffineForOp forOp, OpBuilder &builder) {
  return builder.create<arith::ConstantIndexOp>(forOp.getLoc(),
                                                forOp.getStep());
}

// Get a Value for the loop lower bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitLowerBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineLowerBound(forOp, builder);
}

// Get a Value for the loop upper bound.  If the value requires computation,
// materialize the instructions using builder.
static Value getOrEmitUpperBound(AffineForOp forOp, OpBuilder &builder) {
  return lowerAffineUpperBound(forOp, builder);
}

// Check the structure of the loop nest:
//   - there are enough loops to map to numDims;
//   - the loops are perfectly nested;
//   - the loop bounds can be computed above the outermost loop.
// This roughly corresponds to the "matcher" part of the pattern-based
// rewriting infrastructure.
static LogicalResult YasscheckAffineLoopNestMappableImpl(AffineForOp forOp,
                                                     unsigned numDims) {
  Region &limit = forOp.region();
  for (unsigned i = 0, e = numDims; i < e; ++i) {
    Operation *nested = &forOp.getBody()->front();
    if (!areValuesDefinedAbove(getLowerBoundOperands(forOp), limit) ||
        !areValuesDefinedAbove(getUpperBoundOperands(forOp), limit))
      return forOp.emitError(
          "loops with bounds depending on other mapped loops "
          "are not supported");

    // The innermost loop can have an arbitrary body, skip the perfect nesting
    // check for it.
    if (i == e - 1)
      break;

    auto begin = forOp.getBody()->begin(), end = forOp.getBody()->end();
    if (forOp.getBody()->empty() || std::next(begin, 2) != end)
      return forOp.emitError("expected perfectly nested loops in the body");

    if (!(forOp = dyn_cast<AffineForOp>(nested)))
      return nested->emitError("expected a nested loop");
  }
  return success();
}

static LogicalResult YasscheckAffineLoopNestMappable(AffineForOp forOp,
                                                 unsigned numBlockDims,
                                                 unsigned numThreadDims) {
  if (numBlockDims < 1 || numThreadDims < 1) {
    LLVM_DEBUG(llvm::dbgs() << "nothing to map");
    return success();
  }

  if (numBlockDims > 3) {
    return forOp.emitError("cannot map to more than 3 block dimensions");
  }
  if (numThreadDims > 3) {
    return forOp.emitError("cannot map to more than 3 thread dimensions");
  }
  return YasscheckAffineLoopNestMappableImpl(forOp, numBlockDims + numThreadDims);
}

namespace {
// Helper structure that holds common state of the loop to GPU kernel
// conversion.
struct YassAffineLoopToGpuConverter {
  Optional<AffineForOp> collectBounds(AffineForOp forOp, unsigned numLoops);

  void createLaunch(AffineForOp rootForOp, AffineForOp innermostForOp,
                    unsigned numBlockDims, unsigned numThreadDims);

  // Ranges of the loops mapped to blocks or threads.
  SmallVector<Value, 6> dims;
  // Lower bounds of the loops mapped to blocks or threads.
  SmallVector<Value, 6> lbs;
  // Induction variables of the loops mapped to blocks or threads.
  SmallVector<Value, 6> ivs;
  // Steps of the loops mapped to blocks or threads.
  SmallVector<Value, 6> steps;
};
} // namespace

// Return true if the value is obviously a constant "one".
static bool isConstantOne(Value value) {
  if (auto def = value.getDefiningOp<arith::ConstantIndexOp>())
    return def.value() == 1;
  return false;
}

// Collect ranges, bounds, steps and induction variables in preparation for
// mapping a loop nest of depth "numLoops" rooted at "forOp" to a GPU kernel.
// This may fail if the IR for computing loop bounds cannot be constructed, for
// example if an affine loop uses semi-affine maps. Return the last loop to be
// mapped on success, llvm::None on failure.
Optional<AffineForOp>
YassAffineLoopToGpuConverter::collectBounds(AffineForOp forOp, unsigned numLoops) {
  OpBuilder builder(forOp.getOperation());
  dims.reserve(numLoops);
  lbs.reserve(numLoops);
  ivs.reserve(numLoops);
  steps.reserve(numLoops);
  AffineForOp currentLoop = forOp;
  for (unsigned i = 0; i < numLoops; ++i) {
    Value lowerBound = getOrEmitLowerBound(currentLoop, builder);
    Value upperBound = getOrEmitUpperBound(currentLoop, builder);
    if (!lowerBound || !upperBound) {
      return llvm::None;
    }

    Value range = builder.create<arith::SubIOp>(currentLoop.getLoc(),
                                                upperBound, lowerBound);
    Value step = getOrCreateStep(currentLoop, builder);
    if (!isConstantOne(step))
      range = builder.create<arith::DivSIOp>(currentLoop.getLoc(), range, step);
    dims.push_back(range);

    lbs.push_back(lowerBound);
    ivs.push_back(currentLoop.getInductionVar());
    steps.push_back(step);

    if (i != numLoops - 1)
      currentLoop = cast<AffineForOp>(&currentLoop.getBody()->front());
  }
  return currentLoop;
}

// Replace the rooted at "rootForOp" with a GPU launch operation.  This expects
// "innermostForOp" to point to the last loop to be transformed to the kernel,
// and to have (numBlockDims + numThreadDims) perfectly nested loops between
// "rootForOp" and "innermostForOp".

void YassAffineLoopToGpuConverter::createLaunch(AffineForOp rootForOp,
                                            AffineForOp innermostForOp,
                                            unsigned numBlockDims,
                                            unsigned numThreadDims) {
                                              
  OpBuilder builder(rootForOp.getOperation());
  Type i32 = builder.getI32Type();

  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne =
      (numBlockDims < 3 || numThreadDims < 3)
          ? builder.create<arith::ConstantIndexOp>(rootForOp.getLoc(), 1)
          : nullptr;
  Value gridSizeX = numBlockDims > 0 ? dims[0] : constOne;
  Value gridSizeY = numBlockDims > 1 ? dims[1] : constOne;
  Value gridSizeZ = numBlockDims > 2 ? dims[2] : constOne;
  Value blockSizeX = numThreadDims > 0 ? dims[numBlockDims] : constOne;
  Value blockSizeY = numThreadDims > 1 ? dims[numBlockDims + 1] : constOne;
  Value blockSizeZ = numThreadDims > 2 ? dims[numBlockDims + 2] : constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.
  auto launchOp = builder.create<gpu::LaunchOp>(
      rootForOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX,
      blockSizeY, blockSizeZ);

  // Replace the loop terminator (loops contain only a single block) with the
  // gpu terminator and move the operations from the loop body block to the gpu
  // launch body block.  Do not move the entire block because of the difference
  // in block arguments.
  Operation &terminator = innermostForOp.getBody()->back();
  Location terminatorLoc = terminator.getLoc();
  terminator.erase();
  builder.setInsertionPointToEnd(innermostForOp.getBody());
  builder.create<gpu::TerminatorOp>(terminatorLoc, llvm::None);
  launchOp.body().front().getOperations().splice(
      launchOp.body().front().begin(),
      innermostForOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers instead.  Loops
  // may iterate from LB with step S whereas GPU thread/block ids always iterate
  // from 0 to N with step 1.  Therefore, loop induction variables are replaced
  // with (gpu-thread/block-id * S) + LB.
  builder.setInsertionPointToStart(&launchOp.body().front());
  auto *lbArgumentIt = lbs.begin();
  auto *stepArgumentIt = steps.begin();
  for (const auto &en : llvm::enumerate(ivs)) {
    Value id =
        en.index() < numBlockDims
            ? getDim3Value(launchOp.getBlockIds(), en.index())
            : getDim3Value(launchOp.getThreadIds(), en.index() - numBlockDims);
    Value step = steps[en.index()];
    if (!isConstantOne(step))
      id = builder.create<arith::MulIOp>(rootForOp.getLoc(), step, id);

    Value ivReplacement =
        builder.create<arith::AddIOp>(rootForOp.getLoc(), *lbArgumentIt, id);
    Value ivReplacement2 = builder.create<mlir::arith::IndexCastOp>(rootForOp.getLoc(), ivReplacement, i32);
    en.value().replaceAllUsesWith(ivReplacement2);
    std::advance(lbArgumentIt, 1);
    std::advance(stepArgumentIt, 1);
  }

  // We are done and can erase the original outermost loop.
  rootForOp.erase();
}

// Generic loop to GPU kernel conversion function.
static LogicalResult YassconvertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                                      unsigned numBlockDims,
                                                      unsigned numThreadDims) {
  if (failed(YasscheckAffineLoopNestMappable(forOp, numBlockDims, numThreadDims)))
    return failure();

  YassAffineLoopToGpuConverter converter;
  auto maybeInnerLoop =
      converter.collectBounds(forOp, numBlockDims + numThreadDims);
  if (!maybeInnerLoop)
    return failure();
  converter.createLaunch(forOp, *maybeInnerLoop, numBlockDims, numThreadDims);

  return success();
}
}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace hcl {
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
          std::cout<<"in affine for loop"<<std::endl;
          if (failed(convertAffineLoopNestToGPULaunch(forOp, numBlockDims,
                                                      numThreadDims)))
            {
              std::cout<<"failed affine to gpu"<<std::endl;
              
            }
        });
  
  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOphandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);

////////////////////////////////////////////////////////////////////////////
  // configureParallelLoopToGPULegality(target);
  // for (Region &region : module.getOperation()->getRegions())
  //     {
  //       greedilyMapParallelSCFToGPU(region);
        
  //     }  
  
  // populateParallelLoopToGPUPatterns(patterns);
  //////////////////////////////////////////////////////////////////////////////////////
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    {
      return false;
    }
  //finalizeParallelLoopToGPUConversion(module);
  return true;
}
} // namespace hcl
} // namespace mlir


namespace mlir {
namespace hcl {
bool applyGPUToNVVMLoweringPass(ModuleOp &module, MLIRContext &context) {
  
  LLVMConversionTarget target(context);
  target.addLegalOp<ModuleOp>();
  LLVMTypeConverter typeConverter(&context);
  RewritePatternSet patterns(&context);

  //target.addLegalDialect<scf::SCFDialect, StandardOpsDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  populateGpuToNVVMConversionPatterns(typeConverter,patterns);
  // populateGpuToLLVMConversionPatterns(typeConverter, patterns);
  
  populateReconcileUnrealizedCastsPatterns(patterns);

  patterns.add<CreateLoopHandleOpLowering>(&context);
  patterns.add<CreateOphandleOpLowering>(&context);
  patterns.add<PrintOpLowering>(&context);
  patterns.add<SetIntBitOpLowering>(&context);
  patterns.add<GetIntBitOpLowering>(&context);
  patterns.add<SetIntSliceOpLowering>(&context);
  patterns.add<GetIntSliceOpLowering>(&context);

////////////////////////////////////////////////////////////////////////////
  // configureParallelLoopToGPULegality(target);
  // for (Region &region : module.getOperation()->getRegions())
  //     {
  //       greedilyMapParallelSCFToGPU(region);
        
  //     }  
  
  // populateParallelLoopToGPUPatterns(patterns);
  //////////////////////////////////////////////////////////////////////////////////////
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    {
      return false;
    }
  //finalizeParallelLoopToGPUConversion(module);
  return true;
}
} // namespace hcl
} // namespace mlir

void HCLToLLVMLoweringPass::runOnOperation() {
  auto module = getOperation();
  if (!applyHCLToLLVMLoweringPass(module, getContext()))
   signalPassFailure();
}

void AffineMemOpParLoweringPass::runOnOperation() {
  auto module = getOperation();
  if (!applyAffineMemOpParLoweringPass(module, getContext()))
   signalPassFailure();
}

void AffineToGPULoweringPass::runOnOperation() {
  auto module = getOperation();
  if (!applyAffineToGPULoweringPass(module, getContext()))
   signalPassFailure();
}

void GPUToNVVMLoweringPass::runOnOperation() {
  auto module = getOperation();
  if (!applyGPUToNVVMLoweringPass(module, getContext()))
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

namespace mlir {
namespace hcl {

void registerAffineMemOpParLoweringPass() {
  PassRegistration<AffineMemOpParLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineMemOpParLoweringPass() {
  return std::make_unique<AffineMemOpParLoweringPass>();
}

} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {

void registerAffineToGPULoweringPass() {
  PassRegistration<AffineToGPULoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAffineToGPULoweringPass() {
  return std::make_unique<AffineToGPULoweringPass>();
}

} // namespace hcl
} // namespace mlir

namespace mlir {
namespace hcl {

void registerGPUToNVVMLoweringPass() {
  PassRegistration<GPUToNVVMLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createGPUToNVVMLoweringPass() {
  return std::make_unique<GPUToNVVMLoweringPass>();
}

} // namespace hcl
} // namespace mlir
