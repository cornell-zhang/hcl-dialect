//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// This pass lowers PrintOp to scf for loop nests to print out each element
// of a memref. This transformation is moved outside of the HCLToLLVM pass
// to support fixed-point number printing.
//===----------------------------------------------------------------------===//
#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace hcl;

namespace mlir {
namespace hcl {
FlatSymbolRefAttr getOrInsertPrintf(OpBuilder &rewriter, ModuleOp module);
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, ModuleOp module);
Value castToF64(OpBuilder &rewriter, const Value &src);

void lowerPrintToLoop(ModuleOp &parentModule, Operation *op,
                      OpBuilder &rewriter) {
  auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
  auto memRefShape = memRefType.getShape();
  auto loc = op->getLoc();

  // Get a symbol reference to the printf function, inserting it if necessary.
  auto printfRef = getOrInsertPrintf(rewriter, parentModule);
  Value formatSpecifierCst = getOrCreateGlobalString(
      loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
  Value newLineCst = getOrCreateGlobalString(
      loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

  // Create a loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    // TODO(Niansong): why remove them?
    // for (Operation &nested : *loop.getBody())
    //   rewriter.eraseOp(&nested);
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
  auto casted = castToF64(rewriter, elementLoad);
  rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                          ArrayRef<Value>({formatSpecifierCst, casted}));
  //   llvm::outs() << parentModule << "\n";
}

/// To support printing MemRef with any element type, we cast
/// Int, Float32 types to Float64.
Value castToF64(OpBuilder &rewriter, const Value &src) {
  Type t = src.getType();
  Type F64Type = rewriter.getF64Type();
  Value casted;
  if (t.isa<IntegerType>()) {
    if (t.isUnsignedInteger()) {
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
  } else if (t.isa<FixedType, UFixedType>()) {
    casted = rewriter.create<hcl::FixedToFloatOp>(src.getLoc(), F64Type, src);
  } else {
    llvm::errs() << src.getLoc() << "could not cast value of type "
                 << src.getType() << " to F64.\n";
  }
  return casted;
}

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
FlatSymbolRefAttr getOrInsertPrintf(OpBuilder &rewriter, ModuleOp module) {
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
  OpBuilder::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(context, "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, ModuleOp module) {
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

/// Entry point of lower print pass
bool applyLowerPrintPass(ModuleOp &module) {

  // Get all PrintOps
  SmallVector<Operation *, 10> printOps;
  module.walk([&](Operation *op) {
    if (auto p = dyn_cast<PrintOp>(op)) {
      printOps.push_back(op);
    }
  });

  // Lower each printOP
  for (Operation *printOp : printOps) {
    OpBuilder rewriter(printOp);
    lowerPrintToLoop(module, printOp, rewriter);
  }

  // Remove old print ops
  std::reverse(printOps.begin(), printOps.end());
  for (Operation *op : printOps) {
    op->erase();
  }

  return true;
}

} // namespace hcl
} // namespace mlir

namespace {
struct LowerPrintPass
    : public PassWrapper<LowerPrintPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerPrintPass(mod))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "lower-print"; }
  StringRef getDescription() const final {
    return "Lower HeteroCL dialect to LLVM dialect.";
  }
};
} // namespace

namespace mlir {
namespace hcl {

std::unique_ptr<OperationPass<ModuleOp>> createLowerPrintPass() {
  return std::make_unique<LowerPrintPass>();
}
} // namespace hcl
} // namespace mlir