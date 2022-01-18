//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#ifndef HCL_DIALECT_HLSCPP_VISITOR_H
#define HCL_DIALECT_HLSCPP_VISITOR_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"

namespace mlir {
namespace hcl {

/// This class is a visitor for SSACFG operation nodes.
template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class HLSCppVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // SCF statements.
            scf::ForOp, scf::IfOp, scf::ParallelOp, scf::ReduceOp,
            scf::ReduceReturnOp, scf::YieldOp,
            // Affine statements.
            AffineForOp, AffineIfOp, AffineParallelOp, AffineApplyOp,
            AffineMaxOp, AffineMinOp, AffineLoadOp, AffineStoreOp,
            AffineYieldOp, AffineVectorLoadOp, AffineVectorStoreOp,
            AffineDmaStartOp, AffineDmaWaitOp,
            // Memref-related statements.
            memref::AllocOp, memref::AllocaOp, memref::LoadOp, memref::StoreOp,
            memref::DeallocOp, memref::DmaStartOp, memref::DmaWaitOp,
            memref::ViewOp, memref::SubViewOp, AtomicRMWOp, GenericAtomicRMWOp,
            AtomicYieldOp,
            // Tensor-related statements.
            tensor::ExtractOp, tensor::InsertOp, memref::TensorLoadOp,
            memref::TensorStoreOp, memref::BufferCastOp, SplatOp, memref::DimOp,
            RankOp,
            // Unary expressions.
            math::CosOp, math::SinOp, math::TanhOp, math::SqrtOp, math::RsqrtOp,
            math::ExpOp, math::Exp2Op, math::LogOp, math::Log2Op, math::Log10Op,
            NegFOp,
            // Float binary expressions.
            CmpFOp, AddFOp, SubFOp, MulFOp, DivFOp, RemFOp, SignedDivIOp,
            SignedFloorDivIOp,
            // Integer binary expressions.
            CmpIOp, AddIOp, SubIOp, MulIOp,
            // Special operations.
            CallOp, ReturnOp, SelectOp, ConstantOp, IndexCastOp, UIToFPOp,
            SIToFPOp, FPToSIOp, FPToUIOp,
            UnrealizedConversionCastOp,
            // HCL operations.
            hcl::CreateLoopHandleOp, hcl::CreateStageHandleOp,
            hcl::AddFixedOp, hcl::SubFixedOp, hcl::MulFixedOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitOp(opNode, args...);
            })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation.");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // SCF statements.
  HANDLE(scf::ForOp);
  HANDLE(scf::IfOp);
  HANDLE(scf::ParallelOp);
  HANDLE(scf::ReduceOp);
  HANDLE(scf::ReduceReturnOp);
  HANDLE(scf::YieldOp);

  // Affine statements.
  HANDLE(AffineForOp);
  HANDLE(AffineIfOp);
  HANDLE(AffineParallelOp);
  HANDLE(AffineApplyOp);
  HANDLE(AffineMaxOp);
  HANDLE(AffineMinOp);
  HANDLE(AffineLoadOp);
  HANDLE(AffineStoreOp);
  HANDLE(AffineYieldOp);
  HANDLE(AffineVectorLoadOp);
  HANDLE(AffineVectorStoreOp);
  HANDLE(AffineDmaStartOp);
  HANDLE(AffineDmaWaitOp);

  // Memref-related statements.
  HANDLE(memref::AllocOp);
  HANDLE(memref::AllocaOp);
  HANDLE(memref::LoadOp);
  HANDLE(memref::StoreOp);
  HANDLE(memref::DeallocOp);
  HANDLE(memref::DmaStartOp);
  HANDLE(memref::DmaWaitOp);
  HANDLE(AtomicRMWOp);
  HANDLE(GenericAtomicRMWOp);
  HANDLE(AtomicYieldOp);
  HANDLE(memref::ViewOp);
  HANDLE(memref::SubViewOp);

  // Tensor-related statements.
  HANDLE(tensor::ExtractOp);
  HANDLE(tensor::InsertOp);
  HANDLE(memref::TensorLoadOp);
  HANDLE(memref::TensorStoreOp);
  HANDLE(memref::BufferCastOp);
  HANDLE(SplatOp);
  HANDLE(memref::DimOp);
  HANDLE(RankOp);

  // Unary expressions.
  HANDLE(math::CosOp);
  HANDLE(math::SinOp);
  HANDLE(math::TanhOp);
  HANDLE(math::SqrtOp);
  HANDLE(math::RsqrtOp);
  HANDLE(math::ExpOp);
  HANDLE(math::Exp2Op);
  HANDLE(math::LogOp);
  HANDLE(math::Log2Op);
  HANDLE(math::Log10Op);
  HANDLE(NegFOp);

  // Float binary expressions.
  HANDLE(CmpFOp);
  HANDLE(AddFOp);
  HANDLE(SubFOp);
  HANDLE(MulFOp);
  HANDLE(DivFOp);
  HANDLE(RemFOp);

  // Integer binary expressions.
  HANDLE(CmpIOp);
  HANDLE(AddIOp);
  HANDLE(SubIOp);
  HANDLE(MulIOp);
  HANDLE(SignedDivIOp);
  HANDLE(SignedFloorDivIOp);

  // Special operations.
  HANDLE(CallOp);
  HANDLE(ReturnOp);
  HANDLE(SelectOp);
  HANDLE(ConstantOp);
  HANDLE(IndexCastOp);
  HANDLE(UIToFPOp);
  HANDLE(SIToFPOp);
  HANDLE(FPToUIOp);
  HANDLE(FPToSIOp);
  HANDLE(UnrealizedConversionCastOp);

  // HCL operations
  HANDLE(hcl::CreateLoopHandleOp);
  HANDLE(hcl::CreateStageHandleOp);

  // Fixed point operations
  HANDLE(hcl::AddFixedOp);
  HANDLE(hcl::SubFixedOp);
  HANDLE(hcl::MulFixedOp);

#undef HANDLE
};
} // namespace hcl
} // namespace mlir

#endif // HCL_DIALECT_HLSCPP_VISITOR_H