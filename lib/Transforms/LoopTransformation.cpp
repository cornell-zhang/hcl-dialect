#include "hcl/Dialect/HeteroCLDialect.h"
#include "hcl/Dialect/HeteroCLOps.h"
#include "hcl/Support/Utils.h"
#include "hcl/Transforms/HeteroCLPasses.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
#include <map>
#include <set>

using namespace mlir;
using namespace hcl;

using AffineLoopBand = SmallVector<AffineForOp, 6>;

namespace {

struct HCLLoopTransformation
    : public PassWrapper<HCLLoopTransformation, FunctionPass> {

  void runOnFunction() override;
  void applyLoopTransformation(FuncOp &f);

  LogicalResult runSplitting(FuncOp &f, SplitOp &splitOp);
  LogicalResult runTiling(FuncOp &f, TileOp &tileOp);
  LogicalResult runReordering(FuncOp &f, ReorderOp &reorderOp);
  LogicalResult runUnrolling(FuncOp &f, UnrollOp &unrollOp);
  LogicalResult runPipelining(FuncOp &f, PipelineOp &pipelineOp);
  LogicalResult runParallel(FuncOp &f, ParallelOp &parallelOp);
  LogicalResult runFusing(FuncOp &f, FuseOp &fuseOp);
  LogicalResult runComputeAt(FuncOp &f, ComputeAtOp &computeAtOp);
  LogicalResult runPartition(FuncOp &f, PartitionOp &partitionOp);
  LogicalResult runReuseAt(FuncOp &f, ReuseAtOp &reuseAtOp);
  LogicalResult runBufferAt(FuncOp &f, BufferAtOp &bufferAtOp);

  StringRef getArgument() const final { return "hcl-loop-transformation"; }
  StringRef getDescription() const final {
    return "Loop transformation in HeteroCL";
  }
};

} // namespace

struct ExprCompare {
  int findConstantExpr(const AffineExpr &exp) const {
    int value = -1;
    // TODO: only support one constant now
    exp.walk([&](AffineExpr inner) {
      if (inner.isa<AffineConstantExpr>())
        value = inner.cast<AffineConstantExpr>().getValue();
    });
    return value;
  }
  bool operator()(const AffineExpr &exp1, const AffineExpr &exp2) const {
    int val1 = findConstantExpr(exp1);
    int val2 = findConstantExpr(exp2);
    return val1 < val2;
  }
};

Attribute createZeroAttr(OpBuilder &builder, mlir::Type elementType) {
  if (elementType.isa<FloatType>())
    return builder.getFloatAttr(elementType, 0.0);
  if (elementType.isa<IntegerType>())
    return builder.getIntegerAttr(elementType, 0);
  return {};
}

LogicalResult HCLLoopTransformation::runSplitting(FuncOp &f, SplitOp &splitOp) {
  // 1) Get the schedule
  unsigned int factor = splitOp.factor();
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(splitOp.loop().getDefiningOp()).loop_name();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(splitOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loop
  bool isOuterMost = false;
  AffineLoopBand band;
  rootForOp->walk([&](AffineForOp forOp) {
    if (band.size() == 0 && loop_name == getLoopName(forOp)) {
      band.push_back(forOp);
      if (forOp->hasAttr("stage_name"))
        isOuterMost = true;
    }
  });
  // handle exception
  if (band.size() == 0) {
    splitOp.emitError("Cannot find Loop ")
        << loop_name.str() << " in Stage " << stage_name.str();
    return failure();
  }
  if (factor >= band[0].getConstantUpperBound()) {
    splitOp.emitError("The requested tiling factor (")
        << factor << ") is larger than the upper bound ("
        << band[0].getConstantUpperBound() << ") of the loop";
    return failure();
  }

  // 4) Split the loop
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(factor);
  AffineLoopBand tiledNest;
  if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest)))
    return failure();
  if (isOuterMost)
    rootForOp = tiledNest[0];

  // 5) Loop normalization
  // Note: 5) & 6) are used for making the loop bound constants
  //       Otherwise, loops are not perfectly nested
  normalizeAffineFor(tiledNest[0]);
  normalizeAffineFor(tiledNest[1]);
  auto ub = tiledNest[1].getUpperBound();
  auto ubMap = ub.getMap();
  if (ubMap.isConstant()) {
    // Exception case that cannot change loop bound:
    // #map1 = affine_map<(d0, d1) -> (7, -d0 + 1024)>
    // %5 = affine.apply #map0(%arg3)
    // affine.for %arg4 = 0 to min #map1(%5, %5)
    auto cstUb = ubMap.getResult(0).dyn_cast<AffineConstantExpr>().getValue();
    OpBuilder opBuilder(tiledNest[1]);
    tiledNest[1].setUpperBound({}, opBuilder.getConstantAffineMap(cstUb));
  }

  // 6) Sink AffineApply Operations
  auto fstApply = *(tiledNest[0].getOps<AffineApplyOp>().begin());
  auto sndApply = *(tiledNest[1].getOps<AffineApplyOp>().begin());
  bool isDone = false;
  rootForOp->walk([&](AffineForOp forOp) { // from the innermost
    if (isDone)
      return;
    sndApply->moveBefore(&(*forOp.getBody()->getOperations().begin()));
    // definition should come before reference
    bool isDominance = true;
    for (auto user : sndApply->getUsers()) {
      DominanceInfo domInfo;
      if (!domInfo.properlyDominates(sndApply->getResult(0), user)) {
        isDominance = false;
        break;
      }
    }
    if (isDominance)
      isDone = true;
  });
  if (isDone && ubMap.isConstant())
    fstApply->moveBefore(sndApply);

  // 7) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(loop_name.str() + ".outer");
  newNameArr.push_back(loop_name.str() + ".inner");
  setLoopNames(tiledNest, newNameArr);
  if (isOuterMost)
    setStageName(tiledNest[0], stage_name);

  // 8) Create new loop handles
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto outer = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      StringAttr::get(firstOp->getContext(), newNameArr[0]));
  auto inner = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      StringAttr::get(firstOp->getContext(), newNameArr[1]));

  // 9) Link the loop handles with SSA values
  splitOp.getResult(0).replaceAllUsesWith(outer);
  splitOp.getResult(1).replaceAllUsesWith(inner);

  return success();
}

LogicalResult HCLLoopTransformation::runTiling(FuncOp &f, TileOp &tileOp) {
  // 1) Get the schedule
  unsigned int x_factor = tileOp.x_factor();
  unsigned int y_factor = tileOp.y_factor();
  const auto x_loop =
      dyn_cast<CreateLoopHandleOp>(tileOp.x_loop().getDefiningOp()).loop_name();
  const auto y_loop =
      dyn_cast<CreateLoopHandleOp>(tileOp.y_loop().getDefiningOp()).loop_name();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(tileOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loops
  bool isFound = false;
  bool isOuterMost = false;
  SmallVector<StringRef, 6> nameArr;
  nameArr.push_back(x_loop);
  nameArr.push_back(y_loop);
  AffineLoopBand band;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!isFound && findContiguousNestedLoops(forOp, band, nameArr))
      isFound = true;
    return;
  });
  // handle exception
  if (!isFound) {
    tileOp.emitError("Cannot find contiguous nested loops starting from Loop ")
        << x_loop.str();
    return failure();
  }
  if (x_factor >= band[0].getConstantUpperBound()) {
    tileOp.emitError("The requested tiling factor (")
        << x_factor << ") is larger than the upper bound ("
        << band[0].getConstantUpperBound() << ") of the loop";
    return failure();
  }
  if (y_factor >= band[1].getConstantUpperBound()) {
    tileOp.emitError("The requested tiling factor (")
        << y_factor << ") is larger than the upper bound ("
        << band[1].getConstantUpperBound() << ") of the loop";
    return failure();
  }
  if (band[0]->hasAttr("stage_name"))
    isOuterMost = true;

  // 4) Tile the loops
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(x_factor);
  tileSizes.push_back(y_factor);
  AffineLoopBand tiledNest;
  if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest)))
    return failure();
  if (isOuterMost)
    rootForOp = tiledNest[0];

  // 5) Loop normalization
  // Note: 5) & 6) are used for making the loop bound constants
  //       Otherwise, loops are not perfectly nested
  for (int i = 0; i < 4; ++i)
    normalizeAffineFor(tiledNest[i]);
  // the tiled factor loops are the inner two
  for (int i = 2; i < 4; ++i) {
    auto ub = tiledNest[i].getUpperBound();
    auto ubMap = ub.getMap();
    if (ubMap.isConstant()) {
      auto cstUb = ubMap.getResult(0).dyn_cast<AffineConstantExpr>().getValue();
      OpBuilder opBuilder(tiledNest[i]);
      tiledNest[i].setUpperBound({}, opBuilder.getConstantAffineMap(cstUb));
    }
  }

  // 6) Sink AffineApply Operations
  for (int i = 1; i >= 0; --i) { // from inner to outer
    auto fstApply = *(tiledNest[i].getOps<AffineApplyOp>().begin());
    auto sndApply = *(tiledNest[i + 2].getOps<AffineApplyOp>().begin());
    bool isDone = false;
    rootForOp->walk([&](AffineForOp forOp) { // from the innermost
      if (isDone)
        return;
      sndApply->moveBefore(&(*forOp.getBody()->getOperations().begin()));
      // definition should come before reference
      bool isDominance = true;
      for (auto user : sndApply->getUsers()) {
        DominanceInfo domInfo;
        if (!domInfo.properlyDominates(sndApply->getResult(0), user)) {
          isDominance = false;
          break;
        }
      }
      if (isDominance)
        isDone = true;
    });
    if (isDone && tiledNest[i + 2].getUpperBound().getMap().isConstant())
      fstApply->moveBefore(sndApply);
  }

  // 7) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(x_loop.str() + ".outer");
  newNameArr.push_back(x_loop.str() + ".inner");
  newNameArr.push_back(y_loop.str() + ".outer");
  newNameArr.push_back(y_loop.str() + ".inner");
  setLoopNames(tiledNest, newNameArr);
  if (isOuterMost)
    setStageName(tiledNest[0], stage_name);

  // 8) Create new loop handles &
  //    Link the loop handles with SSA values
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  for (int i = 0; i < 4; ++i) {
    auto handle = builder.create<CreateLoopHandleOp>(
        firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
        StringAttr::get(firstOp->getContext(), newNameArr[i]));
    tileOp.getResult(i).replaceAllUsesWith(handle);
  }

  return success();
}

LogicalResult HCLLoopTransformation::runReordering(FuncOp &f,
                                                   ReorderOp &reorderOp) {
  // 1) Get the schedule
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(reorderOp.stage().getDefiningOp())
          .stage_name();
  const auto loopsToReorder = reorderOp.loops(); // operand_range
  if (loopsToReorder.size() < 2) {
    reorderOp.emitError("Should at least input 2 loops to be reordered");
    return failure();
  }

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Traverse all the loops in the stage
  //    Get a mapping from loop name to id
  std::map<std::string, unsigned> oldName2ID;
  SmallVector<std::string> oldLoopNames;
  unsigned int curr_depth = 0;
  rootForOp.walk([&](AffineForOp forOp) { // from the inner most!
    std::string loop_name = getLoopName(forOp).str();
    oldName2ID[loop_name] = curr_depth;
    oldLoopNames.push_back(loop_name);
    curr_depth++;
  });
  // since .walk() method traverse the nested loops
  // from the inner-most, the names and mapping should be reversed
  std::reverse(oldLoopNames.begin(), oldLoopNames.end());
  for (auto name : oldLoopNames) {
    oldName2ID[name] = curr_depth - 1 - oldName2ID[name];
  }

  // 4) Traverse all the input arguments that need to be reordered and
  // construct permMap
  // Possible inputs:
  // a) # arguments = # loops: (i,j,k)->(k,j,i)
  // b) # arguments != # loops: input (k,i), but should be the same as a)

  // 4.1) Map input arguments to the corresponding loop names
  SmallVector<std::string> nameOfLoopsToReorder;
  for (auto loop : loopsToReorder) {
    nameOfLoopsToReorder.push_back(loop.getDefiningOp()
                                       ->getAttr("loop_name")
                                       .cast<StringAttr>()
                                       .getValue()
                                       .str());
  }

  // 4.2) Make Case b) to Case a)
  //      i.e. fill in all the missing loops in Case b)
  SmallVector<std::string> nameOfAllLoopsWithNewOrder;
  unsigned int cntInArgs = 0;
  for (unsigned int i = 0, e = oldLoopNames.size(); i < e; ++i) {
    auto name = oldLoopNames[i];
    auto iterator = std::find(nameOfLoopsToReorder.begin(),
                              nameOfLoopsToReorder.end(), name);
    if (iterator != nameOfLoopsToReorder.end()) { // name in the arguments
      nameOfAllLoopsWithNewOrder.push_back(nameOfLoopsToReorder[cntInArgs++]);
    } else { // not in
      nameOfAllLoopsWithNewOrder.push_back(name);
    }
  }

  // 4.3) Traverse the original loop nests and create a new order (permMap) for
  // the loops, where permMap[i] means the ith loop in the original nests will
  // become the permMap[i]-th loop
  unsigned int outerMostIdx = 0;
  SmallVector<unsigned, 6> permMap;
  for (unsigned int i = 0, e = oldLoopNames.size(); i < e; ++i) {
    auto name = oldLoopNames[i];
    auto iterator = std::find(nameOfAllLoopsWithNewOrder.begin(),
                              nameOfAllLoopsWithNewOrder.end(), name);
    unsigned int idx = iterator - nameOfAllLoopsWithNewOrder.begin();
    permMap.push_back(idx);
    if (idx == 0) {
      outerMostIdx = i;
    }
  }

  // 5) Permute the loops
  // TODO: imperfect loops
  // 5.1) Get the maximal perfect nest
  AffineLoopBand nest;
  getPerfectlyNestedLoops(nest, rootForOp);
  // 5.2) Permute if the nest's size is consistent with the specified
  // permutation
  if (nest.size() >= 2 && nest.size() == permMap.size()) {
    if (outerMostIdx != 0)
      nest[0]->removeAttr("stage_name");
    permuteLoops(nest, permMap);
  } else {
    reorderOp.emitError("Cannot permute the loops because the size of the "
                        "perfectly nested loop band (")
        << nest.size() << ") "
        << "is not consistent with the size of permutation mapping ("
        << permMap.size() << ")";
    return failure();
  }

  // 6) Rename the stage if the outermost loop moves inward
  if (outerMostIdx != 0) {
    nest[outerMostIdx]->setAttr(
        "stage_name",
        StringAttr::get(nest[outerMostIdx]->getContext(), stage_name));
  }

  return success();
}

LogicalResult HCLLoopTransformation::runUnrolling(FuncOp &f,
                                                  UnrollOp &unrollOp) {
  // 1) Get the schedule
  auto optional_factor = unrollOp.factor();
  unsigned int factor;
  if (optional_factor.hasValue()) {
    factor = optional_factor.getValue();
  } else {
    factor = 0; // fully unroll
  }
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(unrollOp.loop().getDefiningOp()).loop_name();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(unrollOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  bool isFound = false;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!isFound && loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)factor};
      setIntAttr(band, attr_arr, "unroll");
      isFound = true;
    }
  });
  // handle exception
  if (!isFound) {
    unrollOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  return success();
}

LogicalResult HCLLoopTransformation::runParallel(FuncOp &f,
                                                 ParallelOp &parallelOp) {
  // 1) Get the schedule
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(parallelOp.loop().getDefiningOp())
          .loop_name();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(parallelOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  bool isFound = false;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!isFound && loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{1};
      setIntAttr(band, attr_arr, "parallel");
      isFound = true;
    }
  });
  // handle exception
  if (!isFound) {
    parallelOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }

  return success();
}

LogicalResult HCLLoopTransformation::runPipelining(FuncOp &f,
                                                   PipelineOp &pipelineOp) {
  // 1) Get the schedule
  auto optional_ii = pipelineOp.ii();
  unsigned int ii;
  if (optional_ii.hasValue()) {
    ii = optional_ii.getValue();
  } else {
    ii = 1;
  }
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(pipelineOp.loop().getDefiningOp())
          .loop_name();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(pipelineOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loop and attach attribute
  bool isFound = false;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!isFound && loop_name == getLoopName(forOp)) {
      AffineLoopBand band{forOp};
      SmallVector<int, 6> attr_arr{(int)ii};
      setIntAttr(band, attr_arr, "pipeline_ii");
      isFound = true;
    }
  });
  // handle exception
  if (!isFound) {
    pipelineOp.emitError("Cannot find Loop ") << loop_name.str();
    return failure();
  }
  return success();
}

// modified from lib/Transforms/Utils/LoopUtils.cpp
LogicalResult coalesceLoops(MutableArrayRef<AffineForOp> loops,
                            AffineForOp stageLoop) {
  if (loops.size() < 2)
    return failure();

  AffineForOp innermost = loops.back();
  AffineForOp outermost = loops.front();
  AffineBound ub = outermost.getUpperBound();
  Location loc = outermost.getLoc();
  OpBuilder builder(outermost);
  for (AffineForOp loop : loops) {
    // We only work on normalized loops.
    if (loop.getStep() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0)
      return failure();
    // TODO: support AffineMap loop bounds
    if (!loop.hasConstantUpperBound())
      return failure();
  }
  SmallVector<Value, 4> upperBoundSymbols;
  SmallVector<Value, 4> ubOperands(ub.getOperands().begin(),
                                   ub.getOperands().end());

  // 1. Store the upper bound of the outermost loop in a variable.
  // 2. Emit code computing the upper bound of the coalesced loop as product of
  // the number of iterations of all loops.
  int64_t prod = 1;
  for (AffineForOp loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    prod *= cstUb;
    auto cstOp = builder.create<ConstantIndexOp>(loc, cstUb);
    upperBoundSymbols.push_back(cstOp);
    // hoist to the outermost
    cstOp->moveBefore(stageLoop);
  }
  outermost.setConstantUpperBound(prod);

  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  Value previous = outermost.getInductionVar();
  SmallVector<Operation *> opToSink;
  for (unsigned idx = loops.size(); idx > 0; --idx) {
    if (idx != loops.size()) {
      SmallVector<Value, 4> operands;
      operands.push_back(previous);
      operands.push_back(upperBoundSymbols[idx]);
      previous = builder.create<AffineApplyOp>(
          loc,
          AffineMap::get(
              /*numDims=*/1, /*numSymbols=*/1,
              builder.getAffineDimExpr(0).floorDiv(
                  builder.getAffineSymbolExpr(0))),
          operands);
      opToSink.push_back(previous.getDefiningOp());
    }
    // Modified value of the induction variables of the nested loops after
    // coalescing.
    Value inductionVariable;
    if (idx == 1) {
      inductionVariable = previous;
    } else {
      SmallVector<Value, 4> applyOperands;
      applyOperands.push_back(previous);
      applyOperands.push_back(upperBoundSymbols[idx - 1]);
      inductionVariable = builder.create<AffineApplyOp>(
          loc,
          AffineMap::get(
              /*numDims=*/1, /*numSymbols=*/1,
              builder.getAffineDimExpr(0) % builder.getAffineSymbolExpr(0)),
          applyOperands);
      opToSink.push_back(inductionVariable.getDefiningOp());
    }
    replaceAllUsesInRegionWith(loops[idx - 1].getInductionVar(),
                               inductionVariable, loops.back().region());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  AffineForOp secondOutermostLoop = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(secondOutermostLoop.getOperation()),
      innermost.getBody()->getOperations());
  secondOutermostLoop.erase();

  // 5. Sink AffineApply operations
  bool isDone = false;
  std::reverse(opToSink.begin(), opToSink.end());
  loops[0]->walk([&](AffineForOp forOp) { // from the innermost
    if (forOp == loops[0] || isDone)
      return;
    bool isDominance = true;
    for (auto applyOp : opToSink) {
      applyOp->moveBefore(&(*forOp.getBody()->getOperations().begin()));
      // definition should come before reference
      for (auto user : applyOp->getUsers()) {
        DominanceInfo domInfo;
        if (!domInfo.properlyDominates(applyOp->getResult(0), user)) {
          isDominance = false;
          break;
        }
      }
    }
    if (isDominance)
      isDone = true;
  });
  return success();
}

// Notice hcl.fuse (fuses nested loops) is different from affine.fuse,
// which fuses contiguous loops. This is actually the case of hcl.compute_at.
LogicalResult HCLLoopTransformation::runFusing(FuncOp &f, FuseOp &fuseOp) {
  // 1) Get the schedule
  const auto loopsToFuse = fuseOp.loops(); // operand_range
  unsigned int sizeOfFusedLoops = loopsToFuse.size();
  if (sizeOfFusedLoops < 2) {
    fuseOp.emitError("Should at least input 2 loops to be fused");
    return failure();
  }
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(fuseOp.stage().getDefiningOp())
          .stage_name();
  SmallVector<StringRef, 6> nameArr;
  for (auto loop : loopsToFuse) {
    nameArr.push_back(
        dyn_cast<CreateLoopHandleOp>(loop.getDefiningOp()).loop_name());
  }

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Find the requested loops
  bool isFound = false;
  bool isOuterMost = false;
  AffineLoopBand band;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!isFound && findContiguousNestedLoops(forOp, band, nameArr))
      isFound = true;
    return;
  });
  // handle exception
  if (!isFound) {
    fuseOp.emitError("Cannot find contiguous nested loops starting from Loop ")
        << nameArr[0].str()
        << ". Please specify the loop to be fused from outermost to innermost.";
    return failure();
  }
  if (band[0]->hasAttr("stage_name"))
    isOuterMost = true;

  // 3) Construct new loop
  MutableArrayRef<AffineForOp> fusedLoops =
      llvm::makeMutableArrayRef(band.data(), sizeOfFusedLoops);
  if (failed(coalesceLoops(fusedLoops, rootForOp)))
    return failure();
  if (isOuterMost)
    rootForOp = fusedLoops[0];

  // 5) Add name to the new loop
  std::string new_name;
  for (auto name : nameArr) {
    new_name += name.str() + "_";
  }
  new_name += "fused";
  setLoopName(fusedLoops[0], new_name);
  if (isOuterMost)
    setStageName(fusedLoops[0], stage_name);

  // 6) Create new loop handles &
  //    Link the loop handles with SSA values
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto fused = builder.create<CreateLoopHandleOp>(
      firstOp->getLoc(), LoopHandleType::get(firstOp->getContext()),
      StringAttr::get(firstOp->getContext(), new_name));
  fuseOp.getResult().replaceAllUsesWith(fused);

  return success();
}

LogicalResult HCLLoopTransformation::runComputeAt(FuncOp &f,
                                                  ComputeAtOp &computeAtOp) {
  // 1) Get the schedule
  const auto loop_name =
      dyn_cast<CreateLoopHandleOp>(computeAtOp.axis().getDefiningOp())
          .loop_name();
  const auto producer_name =
      dyn_cast<CreateStageHandleOp>(computeAtOp.producer().getDefiningOp())
          .stage_name();
  const auto consumer_name =
      dyn_cast<CreateStageHandleOp>(computeAtOp.consumer().getDefiningOp())
          .stage_name();

  // 2) Traverse all the outer-most loops and find the requested one
  AffineForOp producerFor;
  AffineForOp consumerFor;
  std::pair<bool, bool> isFound{false, false};
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    auto curr_name =
        rootForOp->getAttr("stage_name").cast<StringAttr>().getValue();
    if (producer_name == curr_name) {
      producerFor = rootForOp;
      isFound.first = true;
    } else if (consumer_name == curr_name) {
      consumerFor = rootForOp;
      isFound.second = true;
    }
  }
  if (!isFound.first || !isFound.second) {
    computeAtOp.emitError("Cannot find corresponding producer and consumer");
    return failure();
  }

  // 3) Find the requested loops
  int cnt_depth = 0;
  int requested_depth = 0;
  consumerFor.walk([&](AffineForOp forOp) {
    cnt_depth++;
    Attribute attr = forOp->getAttr("loop_name");
    if (loop_name == attr.cast<StringAttr>().getValue()) {
      requested_depth = cnt_depth;
    }
  });
  requested_depth = cnt_depth - requested_depth + 1;

  // 4) Try to merge two loops
  // TODO: bug: 1) cannot support tensor type
  //            2) gemm merge result seems incorrect
  SmallVector<Dependency, 4> dependency;
  if (!analyzeDependency(producerFor, consumerFor, dependency)) {
    std::string err_msg =
        "Does not support compute_at of stage with if operation.";
    computeAtOp.emitError("analyzeDependency Failed: ") << err_msg;
  }

  FusionStrategy strategy(FusionStrategy::Generic);
  if (dependency.size() > 0) {
    if (std::find(dependency.begin(), dependency.end(), Dependency::RAW) !=
        dependency.end()) {
      strategy = FusionStrategy::ProducerConsumer;
    } else {
      strategy = FusionStrategy::Generic;
    }
  } else {
    strategy = FusionStrategy::Sibling;
  }

  ComputationSliceState sliceUnion;
  FusionResult result = canFuseLoops(producerFor, consumerFor, requested_depth,
                                     &sliceUnion, strategy);
  std::string err_msg;
  if (result.value == FusionResult::Success) {
    fuseLoops(producerFor, consumerFor, sliceUnion);
    producerFor.erase();
    return success();
  } else if (result.value == FusionResult::FailPrecondition) {
    err_msg = "failed precondition for fusion (e.g. same block)";
  } else if (result.value == FusionResult::FailBlockDependence) {
    err_msg = "fusion would violate another dependence in block";
  } else if (result.value == FusionResult::FailFusionDependence) {
    err_msg = "fusion would reverse dependences between loops";
  } else if (result.value == FusionResult::FailComputationSlice) {
    err_msg = "unable to compute src loop computation slice";
  } else if (result.value == FusionResult::FailIncorrectSlice) {
    err_msg = "slice is computed, but it is incorrect";
  }
  computeAtOp.emitError("Cannot merge these two loops because ") << err_msg;

  return failure();
}

// https://github.com/hanchenye/scalehls/blob/master/lib/Transforms/Directive/ArrayPartition.cpp
LogicalResult HCLLoopTransformation::runPartition(FuncOp &f,
                                                  PartitionOp &partitionOp) {
  // 1) Get the schedule
  auto memref = partitionOp.target(); // return a Value type
  auto kind = partitionOp.partition_kind();
  unsigned int target_dim = partitionOp.dim();
  auto optional_factor = partitionOp.factor();
  int factor = 0;
  if (optional_factor.hasValue()) {
    factor = optional_factor.getValue();
  } else {
    factor = -1;
    if (kind != PartitionKindEnum::CompletePartition) {
      partitionOp.emitError("Should pass in `factor' for array partition");
      return failure();
    }
  }

  // 2) Find the requested array
  Value array;
  if (!memref.getDefiningOp()) { // in func args
    bool isFound = false;
    for (auto arg : f.getArguments()) {
      if (memref == arg) { // found the corresponding array
        array = arg;
        isFound = true;
        break;
      }
    }
    if (!isFound) {
      partitionOp.emitError(
          "Cannot find the requested array to be partitioned");
      return failure();
    }
  } else {
    array = memref;
  }

  // 3) Construct new memory layout map
  auto builder = Builder(array.getContext());
  auto arrayType = array.getType().dyn_cast<MemRefType>();
  auto layouts = arrayType.getAffineMaps();

  // Walk through each dimension of the current memory
  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;

  // first N: partition index
  // last N : physical index
  for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
    if (target_dim == 0 || (target_dim > 0 && dim == target_dim - 1)) {
      if (layouts.size() != 0) {
        // TODO: not sure why warning does not work (no output)
        // partitionOp.emitWarning
        partitionOp.emitError("Partition on the array partitioned before."
                              "The original layout map will be rewritten!");
      }
      if (kind == PartitionKindEnum::CyclicPartition) {
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 1, 0, 1
        partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
        addressIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(factor));
      } else if (kind == PartitionKindEnum::BlockPartition) {
        // * block factor N means partition into N blocks
        //   each block has shape[dim] / factor elements
        //   (not N elements in each block!)
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 0, 1, 1
        auto blockFactor =
            (arrayType.getShape()[dim] + factor - 1) / factor; // ceil
        partitionIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(blockFactor));
        addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);
      } else if (kind == PartitionKindEnum::CompletePartition) {
        // original index:  0, 1, 2, 3
        // bank (factor 2): 0, 1, 2, 3
        partitionIndices.push_back(builder.getAffineDimExpr(dim));
        addressIndices.push_back(builder.getAffineConstantExpr(0));
      } else {
        partitionOp.emitError("No this partition kind");
        return failure();
      }
    } else {
      if (layouts.size() == 0) {
        partitionIndices.push_back(builder.getAffineConstantExpr(0));
        addressIndices.push_back(builder.getAffineDimExpr(dim));
      } else { // already had one layout map before
        partitionIndices.push_back(layouts[0].getResult(dim));
        addressIndices.push_back(layouts[0].getResult(dim));
      }
    }
  }

  // Construct new layout map
  partitionIndices.append(addressIndices.begin(), addressIndices.end());
  auto layoutMap = AffineMap::get(arrayType.getRank(), 0, partitionIndices,
                                  builder.getContext());

  // Construct new array type
  auto newType =
      MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                      layoutMap, arrayType.getMemorySpace());

  // Set new type
  array.setType(newType);

  // 4) update function signature
  auto resultTypes = f.front().getTerminator()->getOperandTypes();
  auto inputTypes = f.front().getArgumentTypes();
  f.setType(builder.getFunctionType(inputTypes, resultTypes));

  return success();
}

LogicalResult HCLLoopTransformation::runReuseAt(FuncOp &f,
                                                ReuseAtOp &reuseAtOp) {
  // 1) Get the schedule
  auto target = reuseAtOp.target(); // return a Value type
  unsigned int axis = reuseAtOp.axis();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(reuseAtOp.stage().getDefiningOp())
          .stage_name();
  auto arrayType = target.getType().dyn_cast<MemRefType>();
  unsigned int rank = arrayType.getRank();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Obtain AffineMaps of load instructions
  SmallVector<AffineMap, 6> loadMap;
  std::set<AffineExpr, ExprCompare> requestedVars;
  // TODO: eliminate order in inputs
  reuseAtOp.emitWarning("Need to guarantee the loads have orders");
  rootForOp.walk([&](AffineLoadOp loadOp) {
    if (loadOp.getOperand(0) == target) {
      auto map = loadOp.getAffineMap();
      loadMap.push_back(map);
      requestedVars.insert(map.getResult(axis));
    }
  });

  // 4) Find reduction loops
  AffineLoopBand band;
  rootForOp.walk([&](AffineForOp forOp) {
    if (!forOp->hasAttr("reduction"))
      band.push_back(forOp);
  });
  std::reverse(band.begin(), band.end());
  AffineForOp innerMostForOp = band[band.size() - 1];

  // 5) Try to find reuse pattern
  //    TODO: support more reuse patterns
  bool canReuse = false;
  auto baseVar = *(requestedVars.begin());
  for (auto var : requestedVars) {
    if (std::find(requestedVars.begin(), requestedVars.end(), var + 1) !=
        requestedVars.end()) {
      canReuse = true;
    }
  }
  if (!canReuse) {
    reuseAtOp.emitError("Cannot find reuse pattern on axis ")
        << std::to_string(axis) << ". Only support stride 1 reuse pattern now";
    return failure();
  }

  // 6) Obtain indices and strides in load instructions
  SmallVector<SmallVector<AffineExpr>> allLoadAffineExpr;
  rootForOp.walk([&](AffineLoadOp loadOp) {
    if (loadOp.getOperand(0) == target) {
      auto var = loadOp.getAffineMap().getResult(axis);
      auto diff = var - baseVar;
      SmallVector<AffineExpr> singleLoadAffineExpr;
      if (diff.isa<AffineConstantExpr>()) {
        singleLoadAffineExpr.push_back(diff);
      } else {
        reuseAtOp.emitError("Cannot support non-constant stride");
        return;
      }
      for (unsigned int i = axis + 1; i < rank; ++i) {
        singleLoadAffineExpr.push_back(loadOp.getAffineMap().getResult(i));
      }
      allLoadAffineExpr.push_back(singleLoadAffineExpr);
    }
  });

  // 7) Create reuse buffer
  unsigned int numLoad = loadMap.size();
  int distance = (*(std::prev(allLoadAffineExpr.end())))[0]
                     .dyn_cast<AffineConstantExpr>()
                     .getValue();
  OpBuilder out_builder(rootForOp); // outside the stage
  mlir::Type elementType =
      target.getType().dyn_cast<MemRefType>().getElementType();
  SmallVector<int64_t> shape;
  shape.push_back(distance + 1);
  for (unsigned int i = axis + 1; i < rank; ++i)
    shape.push_back(arrayType.getShape()[i]);
  auto buf = out_builder.create<memref::AllocOp>(
      rootForOp.getLoc(), MemRefType::get(shape, elementType));
  unsigned int buf_rank = buf.getType().dyn_cast<MemRefType>().getRank();

  // 8) link the result SSA with the buffer
  reuseAtOp.getResult().replaceAllUsesWith(buf);

  // 9) Update loop bound & store index
  //    since some load/store will be created later, this step is done in
  //    advance
  SmallVector<AffineExpr> memAffineIndices;
  SmallVector<Operation *> opToRemove;
  // TODO: support non-constant bound
  band[axis].setConstantUpperBound(
      target.getType().dyn_cast<MemRefType>().getShape()[axis]);
  innerMostForOp.walk([&](AffineStoreOp op) {
    OpBuilder rewriter(op);
    memAffineIndices.clear();
    auto oldAffineMap = op.getAffineMap();
    for (unsigned int i = 0, e = oldAffineMap.getResults().size(); i < e; ++i) {
      AffineExpr idx;
      if (i == axis)
        // the iteration space now is related to the input tensor
        idx = oldAffineMap.getResult(i) - distance;
      else
        idx = oldAffineMap.getResult(i);
      memAffineIndices.push_back(idx);
    }
    auto affineMap = AffineMap::get(
        target.getType().dyn_cast<MemRefType>().getRank() /*rank*/, 0,
        memAffineIndices, rewriter.getContext());
    rewriter.create<AffineStoreOp>(
        op->getLoc(), op.getOperand(0) /*valueToStore*/,
        op.getOperand(1) /*memref*/, affineMap, op.indices());
    opToRemove.push_back(op);
  });

  // 10) Rewrite original memref to load from buffer
  innerMostForOp.walk([&](AffineLoadOp op) {
    OpBuilder rewriter(op);
    memAffineIndices.clear();
    auto idx = op.getAffineMap().getResult(axis) - baseVar;
    memAffineIndices.push_back(idx);
    for (unsigned int i = axis + 1; i < rank; ++i)
      memAffineIndices.push_back(op.getAffineMap().getResult(i));
    auto affineMap = AffineMap::get(buf_rank /*rank*/, 0, memAffineIndices,
                                    rewriter.getContext());
    // ValueRange operands{innerMostForOp.getInductionVar()};
    SmallVector<Value> operands;
    unsigned int size = band.size();
    for (unsigned int j = size - buf_rank; j < size; ++j)
      operands.push_back(band[j].getInductionVar());
    auto new_load =
        rewriter.create<AffineLoadOp>(op->getLoc(), buf, affineMap, operands);
    op->replaceAllUsesWith(new_load);
    opToRemove.push_back(op);
  });

  // 11) Create if structure
  //     only if the indices are inside the output tensor iteration space,
  //     results will be computed and written to output
  OpBuilder builder(&(*(innerMostForOp.getBody()->getOperations().begin())));
  auto loc = innerMostForOp.getBody()->getOperations().begin()->getLoc();
  // e.g. #set = affine_set<(d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
  //                                d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
  SmallVector<AffineExpr> constraints{builder.getAffineDimExpr(0) - distance};
  SmallVector<bool> eqFlags{false};
  auto ifCondSet = IntegerSet::get(
      1 /*dimCount*/, 0 /*symbolCount*/,
      constraints /*ArrayRef<AffineExpr> constraints*/, eqFlags);
  SmallVector<Value, 4> setOperands{band[axis].getInductionVar()};
  auto ifOp = builder.create<AffineIfOp>(loc, ifCondSet, setOperands,
                                         /*withElseRegion=*/false);
  auto &innerMostBody = innerMostForOp.getBody()->getOperations();
  auto &ifThenBody = ifOp.getThenBlock()->getOperations();
  ifThenBody.splice(ifThenBody.begin(), innerMostBody,
                    std::next(innerMostBody.begin()),
                    std::prev(innerMostBody.end()));

  // 12) shift buffer elements & load from memory to buffer
  loc = innerMostForOp.getBody()->getOperations().begin()->getLoc();
  for (std::size_t i = 0; i < numLoad; ++i) {
    // %tmp affine.load %buf[1]
    // affine.store %tmp, %buf[0]
    AffineLoadOp load;
    if (i < numLoad - 1) { // load from buffer
      auto affineMap =
          AffineMap::get(buf_rank /*rank*/, 0,
                         allLoadAffineExpr[i + 1] /*need to shift the element*/,
                         builder.getContext());
      SmallVector<Value> operands;
      unsigned int size = band.size();
      for (unsigned int j = size - buf_rank; j < size; ++j)
        operands.push_back(band[j].getInductionVar());
      load = builder.create<AffineLoadOp>(loc, buf, affineMap, operands);
    } else { // load from memory
      SmallVector<Value> memIndices;
      for (auto forOp : band)
        memIndices.push_back(forOp.getInductionVar());
      load = builder.create<AffineLoadOp>(loc, target, memIndices);
    }
    load->moveBefore(ifOp); // move inside if structure

    auto affineMap = AffineMap::get(buf_rank /*rank*/, 0, allLoadAffineExpr[i],
                                    builder.getContext());
    SmallVector<Value> operands;
    unsigned int size = band.size();
    for (unsigned int j = size - buf_rank; j < size; ++j)
      operands.push_back(band[j].getInductionVar());
    auto store =
        builder.create<AffineStoreOp>(loc, load, buf, affineMap, operands);
    store->moveBefore(ifOp);
  }

  // 13) Remove all the useless operations
  for (Operation *op : opToRemove) {
    op->erase();
  }

  return success();
}

LogicalResult HCLLoopTransformation::runBufferAt(FuncOp &f,
                                                 BufferAtOp &bufferAtOp) {
  // 1) Get the schedule
  auto target = bufferAtOp.target(); // return a Value type
  int axis = bufferAtOp.axis();
  const auto stage_name =
      dyn_cast<CreateStageHandleOp>(bufferAtOp.stage().getDefiningOp())
          .stage_name();

  // 2) Find the requested stage
  AffineForOp rootForOp;
  if (failed(getStage(f, rootForOp, stage_name))) {
    f.emitError("Cannot find Stage ") << stage_name.str();
    return failure();
  }

  // 3) Obtain non-reduction loops and reduction loops
  AffineLoopBand band;
  SmallVector<StringRef, 6> nameArr;
  // TODO: test if the requested loop has the target tensor
  bool isFound = findContiguousNestedLoops(rootForOp, band, nameArr);
  if (!isFound) {
    bufferAtOp.emitError("Cannot find nested loops for buffer_at");
    return failure();
  }
  SmallVector<AffineForOp, 6> nonReductionForOps;
  SmallVector<StringRef, 6> nonReductionNameArr;
  int firstReductionIdx = -1;
  for (std::size_t i = 0, e = band.size(); i != e; ++i) {
    if (!band[i]->hasAttr("reduction")) {
      nonReductionForOps.push_back(band[i]);
      nonReductionNameArr.push_back(getLoopName(band[i]));
    } else {
      if (firstReductionIdx == -1)
        firstReductionIdx = i;
    }
  }
  if (firstReductionIdx == -1)
    firstReductionIdx = band.size() - 1;
  // handle exception
  if (axis >= 0 && ((std::size_t)(axis + 1) >= band.size())) {
    bufferAtOp.emitError("Cannot buffer at the inner-most loop: axis=")
        << std::to_string(axis)
        << " inner-most axis=" << std::to_string(band.size() - 1);
    return failure();
  }
  if (axis >= 0 && axis >= firstReductionIdx) {
    bufferAtOp.emitError("Cannot buffer inside the reduction loops: axis=")
        << std::to_string(axis)
        << ", first reduction axis=" << std::to_string(firstReductionIdx);
    return failure();
  }

  // 4) Create write buffer
  // e.g.:
  // without reordering: (0, 1, 2r)
  //   buf_at 0: 1;(1,2r);1 insert at all[axis+1] but take non-red[axis+1]
  //   var buf_at 1: c;2r;c inner-most non-red buf_at 2: x cannot buffer
  //   at the inner-most
  // with reordering: (0, 1r, 2)
  //   buf_at 0: 2;(1r,2);2 non-red[axis+1]
  //   buf_at 1: x cannot buffer inside reduction loop
  //   buf_at 2: x
  if (axis == firstReductionIdx - 1 &&
      (std::size_t)firstReductionIdx ==
          nonReductionForOps.size()) { // inner-most non-reduction loop &&
                                       // no non-reduction loops inside
    OpBuilder builder(band[firstReductionIdx]);
    Location loc_front = band[firstReductionIdx].getLoc();
    mlir::Type elementType =
        target.getType().dyn_cast<MemRefType>().getElementType();
    SmallVector<Value, 4> memIndices;
    // a) Initialization
    // buffer only has one element
    auto buf = builder.create<memref::AllocOp>(
        loc_front, MemRefType::get({1}, elementType));
    auto zero = builder.create<ConstantOp>(
        loc_front, elementType, createZeroAttr(builder, elementType));
    // no need to create an explicit loop
    auto idx = builder.create<ConstantIndexOp>(loc_front, 0);
    memIndices.push_back(idx);
    builder.create<AffineStoreOp>(loc_front, zero, buf, memIndices);

    // link the result SSA with the buffer
    bufferAtOp.getResult().replaceAllUsesWith(buf);

    // b) Rewrite the original buffer
    // TODO: possible bug: replace uses before an untraversed op
    SmallVector<Operation *, 10> opToRemove;
    for (Operation &op : band[firstReductionIdx].getBody()->getOperations()) {
      memIndices.clear();
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (load.getOperand(0) != target)
          continue;
        OpBuilder mid_builder(&op);
        memIndices.push_back(idx);
        auto new_load =
            mid_builder.create<AffineLoadOp>(op.getLoc(), buf, memIndices);
        op.replaceAllUsesWith(new_load);
        opToRemove.push_back(&op);
      } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
        if (store.getOperand(1) != target)
          continue;
        OpBuilder mid_builder(&op);
        memIndices.push_back(idx);
        mid_builder.create<AffineStoreOp>(op.getLoc(), op.getOperand(0), buf,
                                          memIndices);
        opToRemove.push_back(&op);
      }
    }
    for (Operation *op : opToRemove) {
      op->erase();
    }

    // c) Write back
    //    no need to create an explicit loop
    memIndices.clear();
    memIndices.push_back(idx);
    auto load_from_buf =
        builder.create<AffineLoadOp>(loc_front, buf, memIndices);
    memIndices.clear();
    for (int i = 0; i < firstReductionIdx; ++i) {
      memIndices.push_back(band[i].getInductionVar());
    }
    builder.create<AffineStoreOp>(loc_front, load_from_buf, target, memIndices);

    // d) move the original loop in the middle
    band[firstReductionIdx]->moveBefore(load_from_buf);

  } else { // not the inner-most non-reduction axis
    OpBuilder builder(band[axis + 1]);
    Location loc_front = band[axis + 1].getLoc();
    SmallVector<int64_t> ubs;
    for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e; ++i) {
      ubs.push_back(nonReductionForOps[axis + 1].getConstantUpperBound());
    }
    // TODO: support more data types
    mlir::Type elementType =
        target.getType().dyn_cast<MemRefType>().getElementType();
    SmallVector<Value, 4> memIndices;
    // a) Initialization
    // a.1) Allocate buffer
    auto buf = builder.create<memref::AllocOp>(
        loc_front, MemRefType::get(ubs, elementType));
    auto zero = builder.create<ConstantOp>(
        loc_front, elementType, createZeroAttr(builder, elementType));

    // a.2) Create initialization loop
    //      need to create an explicit loop
    SmallVector<AffineForOp> initLoops;
    initLoops.push_back(builder.create<AffineForOp>(loc_front, 0, ubs[0]));
    AffineForOp forOp = initLoops[0];
    for (unsigned int i = axis + 2, e = nonReductionForOps.size(); i < e; ++i) {
      OpBuilder init_builder(&(*(forOp.getBody()->getOperations().begin())));
      forOp = init_builder.create<AffineForOp>(
          forOp.getBody()->getOperations().begin()->getLoc(), 0,
          ubs[i - axis - 1]);
      initLoops.push_back(forOp);
    }

    // a.3) Do the initialization
    OpBuilder init_builder(&(
        *(initLoops[initLoops.size() - 1].getBody()->getOperations().begin())));
    for (auto forOp : initLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    init_builder.create<AffineStoreOp>(initLoops[initLoops.size() - 1].getLoc(),
                                       zero, buf, memIndices);

    // b) Rewrite the original buffer
    SmallVector<Operation *, 10> opToRemove;
    band[axis + 1].walk([&](Operation *op) {
      memIndices.clear();
      if (auto load = dyn_cast<AffineLoadOp>(op)) {
        if (load.getOperand(0) != target)
          return;
        OpBuilder mid_builder(op);
        for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e;
             ++i) {
          memIndices.push_back(nonReductionForOps[i].getInductionVar());
        }
        auto new_load =
            mid_builder.create<AffineLoadOp>(op->getLoc(), buf, memIndices);
        op->replaceAllUsesWith(new_load);
        opToRemove.push_back(op);
      } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
        if (store.getOperand(1) != target)
          return;
        OpBuilder mid_builder(op);
        for (unsigned int i = axis + 1, e = nonReductionForOps.size(); i < e;
             ++i) {
          memIndices.push_back(nonReductionForOps[i].getInductionVar());
        }
        mid_builder.create<AffineStoreOp>(op->getLoc(), op->getOperand(0), buf,
                                          memIndices);
        opToRemove.push_back(op);
      }
    });
    for (Operation *op : opToRemove) {
      op->erase();
    }

    // c) Write back
    // c.1) Create write back loop
    Location loc_back =
        std::prev(band[axis + 1].getBody()->getOperations().end())->getLoc();
    SmallVector<AffineForOp> writeBackLoops;
    writeBackLoops.push_back(builder.create<AffineForOp>(loc_back, 0, ubs[0]));
    forOp = writeBackLoops[0];
    for (unsigned int i = axis + 2, e = nonReductionForOps.size(); i < e; ++i) {
      OpBuilder back_builder(&(*(forOp.getBody()->getOperations().begin())));
      forOp = back_builder.create<AffineForOp>(
          forOp.getBody()->getOperations().begin()->getLoc(), 0,
          ubs[i - axis - 1]);
      writeBackLoops.push_back(forOp);
    }

    // c.2) Load from intermediate results
    OpBuilder back_builder(&(*(writeBackLoops[writeBackLoops.size() - 1]
                                   .getBody()
                                   ->getOperations()
                                   .begin())));
    memIndices.clear();
    for (auto forOp : writeBackLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    auto load_from_buf = back_builder.create<AffineLoadOp>(
        writeBackLoops[writeBackLoops.size() - 1].getLoc(), buf, memIndices);

    // c.3) Store the results back to memory
    memIndices.clear();
    for (int i = 0; i < axis + 1; ++i) {
      memIndices.push_back(nonReductionForOps[i].getInductionVar());
    }
    for (auto forOp : writeBackLoops) {
      memIndices.push_back(forOp.getInductionVar());
    }
    back_builder.create<AffineStoreOp>(
        writeBackLoops[writeBackLoops.size() - 1].getLoc(), load_from_buf,
        target, memIndices);

    // d) Move the original loop between the two loops
    band[axis + 1]->moveBefore(writeBackLoops[0]);

    // e) Add names to loops
    SmallVector<std::string, 6> newNameArr;
    newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_init");
    newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_back");
    SmallVector<AffineForOp, 6> newLoops{initLoops[0], writeBackLoops[0]};
    setLoopNames(newLoops, newNameArr);

    // f) Automatic pipelining
    SmallVector<AffineForOp, 6> twoLoops{
        initLoops[initLoops.size() - 1],
        writeBackLoops[writeBackLoops.size() - 1]};
    SmallVector<int, 6> II{1, 1};
    setIntAttr(twoLoops, II, "pipeline_ii");
  }

  return success();
}

void HCLLoopTransformation::applyLoopTransformation(FuncOp &f) {
  SmallVector<Operation *, 10> opToRemove;
  // schedule should preverse orders, thus traverse one by one
  // the following shows the dispatching logic
  for (Operation &op : f.getOps()) {
    if (auto new_op = dyn_cast<SplitOp>(op)) {
      if (failed(runSplitting(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<TileOp>(op)) {
      if (failed(runTiling(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<ReorderOp>(op)) {
      if (failed(runReordering(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<UnrollOp>(op)) {
      if (failed(runUnrolling(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<PipelineOp>(op)) {
      if (failed(runPipelining(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<ParallelOp>(op)) {
      if (failed(runParallel(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<FuseOp>(op)) {
      if (failed(runFusing(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<ComputeAtOp>(op)) {
      if (failed(runComputeAt(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<PartitionOp>(op)) {
      if (failed(runPartition(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<ReuseAtOp>(op)) {
      if (failed(runReuseAt(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<BufferAtOp>(op)) {
      if (failed(runBufferAt(f, new_op)))
        return signalPassFailure();
      opToRemove.push_back(&op);
    }
  }
  // remove schedule operations (from back to front) & legacy loop handles
  std::reverse(opToRemove.begin(), opToRemove.end());
  std::set<Operation *> handleToRemove;
  for (Operation *op : opToRemove) {
    if (auto new_op = dyn_cast<SplitOp>(op)) {
      handleToRemove.insert(new_op.loop().getDefiningOp());
    } else if (auto new_op = dyn_cast<TileOp>(op)) {
      handleToRemove.insert(new_op.x_loop().getDefiningOp());
      handleToRemove.insert(new_op.y_loop().getDefiningOp());
    } else if (auto new_op = dyn_cast<FuseOp>(op)) {
      for (auto loop : new_op.loops()) {
        handleToRemove.insert(loop.getDefiningOp());
      }
    }
    op->erase();
  }
  for (Operation *op : handleToRemove) {
    op->erase();
  }
}

void HCLLoopTransformation::runOnFunction() {
  FuncOp f = getFunction();
  applyLoopTransformation(f);
}

namespace mlir {
namespace hcl {
// Register Loop Transformation Pass
void registerHCLLoopTransformationPass() {
  PassRegistration<HCLLoopTransformation>();
}

// Create A Loop Transformation Pass
std::unique_ptr<mlir::Pass> createHCLLoopTransformationPass() {
  return std::make_unique<HCLLoopTransformation>();
}

bool applyLoopTransformation(FuncOp &f) {
  HCLLoopTransformation pass;
  pass.applyLoopTransformation(f);
  return true;
}

} // namespace hcl
} // namespace mlir