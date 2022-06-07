//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The HCL-MLIR Authors.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LowerBitOps Pass
// This file defines the lowering of bit operations. 
// - GetBit
// - SetBit
// - GetSlice
// - SetSlice
// - BitReverse
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Dialect/HeteroCLTypes.h"
#include "hcl/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
 using namespace hcl;

 namespace mlir {
 namespace hcl {
   /// Pass entry point
   bool applyLowerBitOps(ModuleOp &mod) {
     return true;
   }
 } // namespace hcl
 } // namespace mlir

 namespace {
 struct HCLLowerBitOpsTransformation
     : public LowerBitOpsBase<HCLLowerBitOpsTransformation> {
   void runOnOperation() override {
     auto mod = getOperation();
     if (!applyLowerBitOps(mod)) {
       return signalPassFailure();
     }
   }
 };
 } // namespace

 namespace mlir {
 namespace hcl {

 std::unique_ptr<OperationPass<ModuleOp>> createLowerBitOpsPass() {
   return std::make_unique<HCLLowerBitOpsTransformation>();
 }
 } // namespace hcl
 } // namespace mlir 