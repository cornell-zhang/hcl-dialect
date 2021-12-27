#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>
#include <vector>
using namespace mlir;

namespace {

struct HCLLoopTransformation
    : public PassWrapper<HCLLoopTransformation, FunctionPass> {

  void runOnFunction() override;

  StringRef getArgument() const final { return "hcl-loop-transformation"; }
  StringRef getDescription() const final {
    return "Loop transformation in HeteroCL";
  }

  void runSplitting(FuncOp &f, hcl::SplitOp &splitOp);
  void runTiling(FuncOp &f, hcl::TileOp &tileOp);
  void runReordering(FuncOp &f, hcl::ReorderOp &reorderOp);
  void runUnrolling(FuncOp &f, hcl::UnrollOp &unrollOp);
  void runPipelining(FuncOp &f, hcl::PipelineOp &pipelineOp);
  void runParallel(FuncOp &f, hcl::ParallelOp &parallelOp);
  void runFusing(FuncOp &f, hcl::FuseOp &fuseOp);
  void runComputeAt(FuncOp &f, hcl::ComputeAtOp &computeAtOp);
  void runPartition(FuncOp &f, hcl::PartitionOp &partitionOp);
  void runBufferAt(FuncOp &f, hcl::BufferAtOp &bufferAtOp);
  // utils
  bool findContiguousNestedLoops(const AffineForOp &rootAffineForOp,
                                 SmallVector<AffineForOp, 6> &resForOps,
                                 SmallVector<StringRef, 6> &nameArr, int depth,
                                 bool countReductionLoops);
  bool addNamesToLoops(SmallVector<AffineForOp, 6> &forOps,
                       const SmallVector<std::string, 6> &nameArr);
  bool addIntAttrsToLoops(SmallVector<AffineForOp, 6> &forOps,
                          const SmallVector<int, 6> &attr_arr,
                          std::string attr_name);
};

} // namespace

bool HCLLoopTransformation::findContiguousNestedLoops(
    const AffineForOp &rootAffineForOp, SmallVector<AffineForOp, 6> &resForOps,
    SmallVector<StringRef, 6> &nameArr, int depth = -1,
    bool countReductionLoops = true) {
  // depth = -1 means traverses all the inner loops
  AffineForOp forOp = rootAffineForOp;
  unsigned int sizeNameArr = nameArr.size();
  if (sizeNameArr != 0)
    depth = sizeNameArr;
  else if (depth == -1)
    depth = 0x3f3f3f3f;
  resForOps.clear();
  for (int i = 0; i < depth; ++i) {
    if (!forOp) {
      if (depth != 0x3f3f3f3f)
        return false;
      else // reach the inner-most loop
        return true;
    }

    Attribute attr = forOp->getAttr("loop_name");
    const StringRef curr_loop = attr.cast<StringAttr>().getValue();
    if (sizeNameArr != 0 && curr_loop != nameArr[i])
      return false;

    if (forOp->hasAttr("reduction") == 1 && !countReductionLoops) {
      i--;
    } else {
      resForOps.push_back(forOp);
      if (sizeNameArr == 0)
        nameArr.push_back(curr_loop);
    }
    Block &body = forOp.region().front();
    // if (body.begin() != std::prev(body.end(), 2)) // perfectly nested
    //   break;

    forOp = dyn_cast<AffineForOp>(&body.front());
  }
  return true;
}

bool HCLLoopTransformation::addIntAttrsToLoops(
    SmallVector<AffineForOp, 6> &forOps, const SmallVector<int, 6> &attr_arr,
    const std::string attr_name) {
  assert(forOps.size() == attr_arr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr(
        attr_name,
        IntegerAttr::get(
            IntegerType::get(newForOp->getContext(), 32,
                             IntegerType::SignednessSemantics::Signless),
            attr_arr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

bool HCLLoopTransformation::addNamesToLoops(
    SmallVector<AffineForOp, 6> &forOps,
    const SmallVector<std::string, 6> &nameArr) {
  assert(forOps.size() == nameArr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr("loop_name", StringAttr::get(newForOp->getContext(),
                                                   nameArr[cnt_loop]));
    cnt_loop++;
  }
  return true;
}

/*
 *  Algorithm:
 *  1) Iterate schedule and get all SplitOp
 *  2) For each SplitOp, find corresponding loop
 *  3) Add names to new loops
 *  4) Remove the schedule operator
 */
void HCLLoopTransformation::runSplitting(FuncOp &f, hcl::SplitOp &splitOp) {
  SmallVector<AffineForOp, 6> tiledNest;
  // 1) get schedule
  unsigned int factor = splitOp.factor();
  const auto loop_name = splitOp.loop()
                             .getDefiningOp()
                             ->getAttr("loop_name")
                             .cast<StringAttr>()
                             .getValue();
  const auto stage_name =
      splitOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Find the requested stage,
  //    traverse all the nested loops,
  //    and split the requested loop
  SmallVector<std::string, 6> newNameArr;
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    // 2.1) Find stage
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      bool isOuterMost = false;
      bool isFound = false;
      SmallVector<AffineForOp, 6> forOps;
      SmallVector<unsigned, 6> tileSizes;
      // 2.2) Find loop
      rootForOp.walk([&](AffineForOp forOp) {
        if (!isFound &&
            loop_name ==
                forOp->getAttr("loop_name").cast<StringAttr>().getValue()) {
          isFound = true;
          forOps.push_back(forOp);
          tileSizes.push_back(factor);
          if (forOp->hasAttr("stage_name"))
            isOuterMost = true;
        }
      });
      // handle exception
      if (!isFound) {
        f.emitError("Cannot find the requested loop in Stage ")
            << stage_name.cast<StringAttr>().getValue().str();
        return signalPassFailure();
      }
      // 2.3) Split the loop
      if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
        return signalPassFailure();

      // 3) Add names to new loops
      newNameArr.push_back(loop_name.str() + ".outer");
      newNameArr.push_back(loop_name.str() + ".inner");
      addNamesToLoops(tiledNest, newNameArr);
      if (isOuterMost) {
        tiledNest[0]->setAttr("stage_name", stage_name);
      }
      break;
    }
  }
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto outer = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  auto inner = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  outer->setAttr("loop_name",
                 StringAttr::get(outer->getContext(), newNameArr[0]));
  inner->setAttr("loop_name",
                 StringAttr::get(inner->getContext(), newNameArr[1]));
  splitOp.getResult(0).replaceAllUsesWith(outer);
  splitOp.getResult(1).replaceAllUsesWith(inner);
}

void HCLLoopTransformation::runTiling(FuncOp &f, hcl::TileOp &tileOp) {
  // 1) get schedule
  unsigned int x_factor = tileOp.x_factor();
  unsigned int y_factor = tileOp.y_factor();
  const StringRef x_loop = tileOp.x_loop()
                               .getDefiningOp()
                               ->getAttr("loop_name")
                               .cast<StringAttr>()
                               .getValue();
  const StringRef y_loop = tileOp.y_loop()
                               .getDefiningOp()
                               ->getAttr("loop_name")
                               .cast<StringAttr>()
                               .getValue();
  const auto stage_name = tileOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Find the requested stage,
  //    traverse all the nested loops,
  //    and tile the requested loops
  SmallVector<std::string, 6> newNameArr;
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    // 2.1) Find stage
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      bool isOuterMost = false;
      bool isFound = false;
      SmallVector<AffineForOp, 6> forOps;
      SmallVector<unsigned, 6> tileSizes;
      tileSizes.push_back(x_factor);
      tileSizes.push_back(y_factor);
      SmallVector<StringRef, 6> nameArr;
      nameArr.push_back(x_loop);
      nameArr.push_back(y_loop);
      // 2.2) Find loops
      rootForOp.walk([&](AffineForOp forOp) {
        if (!isFound && findContiguousNestedLoops(forOp, forOps, nameArr))
          isFound = true;
        return;
      });
      // handle exception
      if (!isFound) {
        f.emitError("Cannot find contiguous nested loops starting from Loop ")
            << nameArr[0].str();
        return signalPassFailure();
      }
      if (forOps[0]->hasAttr("stage_name"))
        isOuterMost = true;
      // 2.3) Tile the loops
      SmallVector<AffineForOp, 6> tiledNest;
      if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
        return signalPassFailure();

      // 3) Add names to new loops
      newNameArr.push_back(x_loop.str() + ".outer");
      newNameArr.push_back(x_loop.str() + ".inner");
      newNameArr.push_back(y_loop.str() + ".outer");
      newNameArr.push_back(y_loop.str() + ".inner");
      addNamesToLoops(tiledNest, newNameArr);
      if (isOuterMost) {
        tiledNest[0]->setAttr("stage_name", stage_name);
      }
      break;
    }
  }
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto x_outer = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  auto x_inner = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  auto y_outer = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  auto y_inner = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  x_outer->setAttr("loop_name",
                   StringAttr::get(x_outer->getContext(), newNameArr[0]));
  x_inner->setAttr("loop_name",
                   StringAttr::get(x_inner->getContext(), newNameArr[1]));
  y_outer->setAttr("loop_name",
                   StringAttr::get(y_outer->getContext(), newNameArr[0]));
  y_inner->setAttr("loop_name",
                   StringAttr::get(y_inner->getContext(), newNameArr[1]));
  tileOp.getResult(0).replaceAllUsesWith(x_outer);
  tileOp.getResult(1).replaceAllUsesWith(x_inner);
  tileOp.getResult(2).replaceAllUsesWith(y_outer);
  tileOp.getResult(3).replaceAllUsesWith(y_inner);
}

void HCLLoopTransformation::runReordering(FuncOp &f,
                                          hcl::ReorderOp &reorderOp) {
  // 1) get schedule
  const auto stage_name =
      reorderOp.stage().getDefiningOp()->getAttr("stage_name");
  const auto loopsToBeReordered = reorderOp.loops(); // operand_range

  // 2) get all the loop names and id mapping
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    // 2.1) Find stage
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      SmallVector<AffineForOp, 6> forOps;
      SmallVector<unsigned, 6> permMap;
      std::map<std::string, unsigned> name2id;
      std::vector<std::string> origNameVec;
      unsigned int curr_depth = 0;
      rootForOp.walk([&](AffineForOp rootAffineForOp) { // from the inner most!
        std::string curr_loop_name = rootAffineForOp->getAttr("loop_name")
                                         .cast<StringAttr>()
                                         .getValue()
                                         .str();
        name2id[curr_loop_name] = curr_depth;
        origNameVec.push_back(curr_loop_name);
        curr_depth++;
      });
      std::reverse(origNameVec.begin(), origNameVec.end());
      for (auto name : origNameVec) { // need to reverse
        name2id[name] = curr_depth - 1 - name2id[name];
      }

      // 3) traverse all the input arguments that need to be reordered and
      // construct permMap possible inputs: a) # arguments = # loops:
      // (i,j,k)->(k,j,i) b) # arguments != # loops:
      //    input (k,i), but should be the same as a)
      // 3.1) map input arguments to the corresponding loop names
      std::vector<std::string> toBeReorderedNameVec;
      for (auto loop : loopsToBeReordered) {
        toBeReorderedNameVec.push_back(loop.getDefiningOp()
                                           ->getAttr("loop_name")
                                           .cast<StringAttr>()
                                           .getValue()
                                           .str());
      }
      // 3.2) traverse the original loop nests and create a new order for the
      // loops
      //     since the input arguments may not cover all the loops
      //     so this step is needed for creating permMap
      unsigned int cntInArgs = 0;
      for (auto name : origNameVec) {
        auto iter = std::find(toBeReorderedNameVec.begin(),
                              toBeReorderedNameVec.end(), name);
        if (iter != toBeReorderedNameVec.end()) { // name in the arguments
          permMap.push_back(name2id[toBeReorderedNameVec[cntInArgs++]]);
        } else { // not in
          permMap.push_back(name2id[name]);
        }
      }

      // 4) permute the loops
      // TODO: a) bug: cannot permute the outer-most loop
      //       b) imperfect loops
      SmallVector<AffineForOp, 6> nest;
      // Get the maximal perfect nest
      getPerfectlyNestedLoops(nest, rootForOp);
      // Permute if the nest's size is consistent with the specified
      // permutation
      if (nest.size() >= 2 && nest.size() == permMap.size()) {
        permuteLoops(nest, permMap);
      } else {
        f.emitError("Cannot permute the loops");
        return signalPassFailure();
      }
      break;
    }
  }
}

void HCLLoopTransformation::runUnrolling(FuncOp &f, hcl::UnrollOp &unrollOp) {
  // 1) get schedule
  unsigned int factor = unrollOp.factor();
  const auto loop_name = unrollOp.loop()
                             .getDefiningOp()
                             ->getAttr("loop_name")
                             .cast<StringAttr>()
                             .getValue();
  const auto stage_name =
      unrollOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Traverse all the nested loops and find the requested one
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      rootForOp.walk([&](AffineForOp forOp) {
        Attribute attr = forOp->getAttr("loop_name");
        if (loop_name == attr.cast<StringAttr>().getValue()) {
          forOp->setAttr(
              "unroll",
              IntegerAttr::get(
                  IntegerType::get(forOp->getContext(), 32,
                                   IntegerType::SignednessSemantics::Signless),
                  factor));
        }
      });
      break;
    }
  }
}

void HCLLoopTransformation::runParallel(FuncOp &f,
                                        hcl::ParallelOp &parallelOp) {
  // 1) get schedule
  const auto loop_name = parallelOp.loop()
                             .getDefiningOp()
                             ->getAttr("loop_name")
                             .cast<StringAttr>()
                             .getValue();
  const auto stage_name =
      parallelOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Traverse all the nested loops and find the requested one
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      rootForOp.walk([&](AffineForOp forOp) {
        Attribute attr = forOp->getAttr("loop_name");
        if (loop_name == attr.cast<StringAttr>().getValue()) {
          forOp->setAttr(
              "parallel",
              IntegerAttr::get(
                  IntegerType::get(forOp->getContext(), 32,
                                   IntegerType::SignednessSemantics::Signless),
                  1) // true
          );
        }
      });
      break;
    }
  }
}

void HCLLoopTransformation::runPipelining(FuncOp &f,
                                          hcl::PipelineOp &pipelineOp) {
  // 1) get schedule
  unsigned int ii = pipelineOp.ii();
  const auto loop_name = pipelineOp.loop()
                             .getDefiningOp()
                             ->getAttr("loop_name")
                             .cast<StringAttr>()
                             .getValue();
  const auto stage_name =
      pipelineOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Traverse all the nested loops and find the requested one
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      rootForOp.walk([&](AffineForOp forOp) {
        Attribute attr = forOp->getAttr("loop_name");
        if (loop_name == attr.cast<StringAttr>().getValue()) {
          forOp->setAttr(
              "pipeline_ii",
              IntegerAttr::get(
                  IntegerType::get(forOp->getContext(), 32,
                                   IntegerType::SignednessSemantics::Signless),
                  ii));
        }
      });
      break;
    }
  }
}

// modified from lib/Transforms/Utils/LoopUtils.cpp
void coalesceLoops(MutableArrayRef<AffineForOp> loops) {
  if (loops.size() < 2)
    return;

  AffineForOp innermost = loops.back();
  AffineForOp outermost = loops.front();
  AffineBound ub = outermost.getUpperBound();
  AffineMap origUbMap = ub.getMap();
  Location loc = outermost.getLoc();
  OpBuilder builder(outermost);
  for (AffineForOp loop : loops) {
    // We only work on normalized loops.
    if (loop.getStep() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0)
      return;
  }
  SmallVector<Value, 4> upperBoundSymbols;
  SmallVector<Value, 4> ubOperands(ub.getOperands().begin(),
                                   ub.getOperands().end());

  // 1. Store the upper bound of the outermost loop in a variable.
  Value prev;
  if (!llvm::hasSingleElement(origUbMap.getResults()))
    prev = builder.create<AffineMinOp>(loc, origUbMap, ubOperands);
  else
    prev = builder.create<AffineApplyOp>(loc, origUbMap, ubOperands);
  upperBoundSymbols.push_back(prev);

  // 2. Emit code computing the upper bound of the coalesced loop as product of
  // the number of iterations of all loops.
  for (AffineForOp loop : loops.drop_front()) {
    ub = loop.getUpperBound();
    origUbMap = ub.getMap();
    ubOperands = ub.getOperands();
    Value upperBound;
    // If upper bound map has more than one result, take their minimum.
    if (!llvm::hasSingleElement(origUbMap.getResults()))
      upperBound = builder.create<AffineMinOp>(loc, origUbMap, ubOperands);
    else
      upperBound = builder.create<AffineApplyOp>(loc, origUbMap, ubOperands);
    upperBoundSymbols.push_back(upperBound);
    SmallVector<Value, 4> operands;
    operands.push_back(prev);
    operands.push_back(upperBound);
    // Maintain running product of loop upper bounds.
    prev = builder.create<AffineApplyOp>(
        loc,
        AffineMap::get(/*numDims=*/1,
                       /*numSymbols=*/1,
                       builder.getAffineDimExpr(0) *
                           builder.getAffineSymbolExpr(0)),
        operands);
  }
  // Set upper bound of the coalesced loop.
  AffineMap newUbMap = AffineMap::get(
      /*numDims=*/0,
      /*numSymbols=*/1, builder.getAffineSymbolExpr(0), builder.getContext());
  outermost.setUpperBound(prev, newUbMap);

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
}

// Notice hcl.fuse (fuses nested loops) is different from affine.fuse,
// which fuses contiguous loops. This is actually the case of hcl.compute_at.
void HCLLoopTransformation::runFusing(FuncOp &f, hcl::FuseOp &fuseOp) {
  // 1) get schedule
  const auto loopsToBeFused = fuseOp.loops(); // operand_range
  unsigned int sizeOfFusedLoops = loopsToBeFused.size();
  const auto stage_name = fuseOp.stage().getDefiningOp()->getAttr("stage_name");
  SmallVector<StringRef, 6> nameArr;
  for (auto loop : loopsToBeFused) {
    nameArr.push_back(loop.getDefiningOp()
                          ->getAttr("loop_name")
                          .cast<StringAttr>()
                          .getValue());
  }

  // 2) Traverse all the nested loops and find the requested ones
  std::string new_name;
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      AffineForOp loopToBeDestroyed;
      SmallVector<AffineForOp, 6> forOps;
      bool isOuterMost = false;
      bool isFound = false;
      rootForOp.walk([&](AffineForOp forOp) {
        if (!isFound && findContiguousNestedLoops(forOp, forOps, nameArr))
          isFound = true;
        return;
      });
      // handle exception
      if (!isFound) {
        f.emitError("Cannot find contiguous nested loops starting from Loop ")
            << nameArr[0].str();
        return signalPassFailure();
      }
      if (forOps[0]->hasAttr("stage_name"))
        isOuterMost = true;

      // 3) construct new loop
      MutableArrayRef<AffineForOp> loops =
          llvm::makeMutableArrayRef(forOps.data(), sizeOfFusedLoops);
      coalesceLoops(loops);

      // 4) add name to the new loop
      for (auto forOp : forOps) {
        new_name +=
            forOp->getAttr("loop_name").cast<StringAttr>().getValue().str() +
            "_";
      }
      new_name += "fused";
      loops[0]->setAttr("loop_name",
                        StringAttr::get(loops[0]->getContext(), new_name));
      if (isOuterMost) {
        loops[0]->setAttr("stage_name", stage_name);
      }
      break;
    }
  }
  auto firstOp = *(f.getOps<AffineForOp>().begin());
  OpBuilder builder(firstOp);
  auto fused = builder.create<hcl::CreateLoopHandleOp>(
      firstOp->getLoc(), hcl::LoopHandleType::get(firstOp->getContext()));
  fused->setAttr("loop_name", StringAttr::get(fused->getContext(), new_name));
  fuseOp.getResult().replaceAllUsesWith(fused);
}

void HCLLoopTransformation::runComputeAt(FuncOp &f,
                                         hcl::ComputeAtOp &computeAtOp) {
  // 1) get schedule
  const auto loop1_name = computeAtOp.loop1()
                              .getDefiningOp()
                              ->getAttr("loop_name")
                              .cast<StringAttr>()
                              .getValue();
  const auto loop2_name = computeAtOp.loop2()
                              .getDefiningOp()
                              ->getAttr("loop_name")
                              .cast<StringAttr>()
                              .getValue();

  // 2) Traverse all the outer-most loops and find the requested one
  SmallVector<AffineForOp, 6> forOps;
  for (auto rootAffineForOp : f.getOps<AffineForOp>()) {
    Attribute attr = rootAffineForOp->getAttr("loop_name");
    if (loop1_name == attr.cast<StringAttr>().getValue() ||
        loop2_name == attr.cast<StringAttr>().getValue()) {
      forOps.push_back(rootAffineForOp);
      std::cout << "found" << std::endl;
    }
  }
  ComputationSliceState sliceUnion;
  for (int i = 3; i >= 1; --i) { // TODO: Change depth
    FusionResult result =
        canFuseLoops(forOps[0], forOps[1], i /*depth*/, &sliceUnion);
    if (result.value == FusionResult::Success) {
      fuseLoops(forOps[0], forOps[1], sliceUnion);
      forOps[0].erase();
      std::cout << std::to_string(i) << " yes" << std::endl;
      return;
    } else
      std::cout << std::to_string(i) << " no" << std::endl;
  }
}

// https://github.com/hanchenye/scalehls/blob/master/lib/Transforms/Directive/ArrayPartition.cpp
void HCLLoopTransformation::runPartition(FuncOp &f,
                                         hcl::PartitionOp &partitionOp) {
  auto memref = partitionOp.target(); // return a Value type
  auto kind = partitionOp.partition_kind();
  // TODO: Partition based on different dimensions
  unsigned int dim = partitionOp.dim();
  unsigned int factor = partitionOp.factor();

  if (!memref.getDefiningOp()) { // in func args
    for (auto arg : f.getArguments()) {
      if (memref == arg) { // found the corresponding array
        auto array = arg;
        auto builder = Builder(array.getContext());
        auto arrayType = array.getType().dyn_cast<MemRefType>();
        // Walk through each dimension of the current memory
        SmallVector<AffineExpr, 4> partitionIndices;
        SmallVector<AffineExpr, 4> addressIndices;

        for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
          if (kind == hcl::PartitionKindEnum::CyclicPartition) {
            partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
            addressIndices.push_back(
                builder.getAffineDimExpr(dim).floorDiv(factor));

          } else if (kind == hcl::PartitionKindEnum::BlockPartition) {
            auto blockFactor =
                (arrayType.getShape()[dim] + factor - 1) / factor;
            partitionIndices.push_back(
                builder.getAffineDimExpr(dim).floorDiv(blockFactor));
            addressIndices.push_back(builder.getAffineDimExpr(dim) %
                                     blockFactor);

          } else if (kind == hcl::PartitionKindEnum::CompletePartition) {
            partitionIndices.push_back(builder.getAffineConstantExpr(0));
            addressIndices.push_back(builder.getAffineDimExpr(dim));
          } else {
            f.emitError("No this partition kind");
          }
        }

        // Construct new layout map
        partitionIndices.append(addressIndices.begin(), addressIndices.end());
        auto layoutMap = AffineMap::get(arrayType.getRank(), 0,
                                        partitionIndices, builder.getContext());

        // Construct new array type
        auto newType =
            MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                            layoutMap, arrayType.getMemorySpace());

        // Set new type
        array.setType(newType);

        // update function signature
        auto resultTypes = f.front().getTerminator()->getOperandTypes();
        auto inputTypes = f.front().getArgumentTypes();
        f.setType(builder.getFunctionType(inputTypes, resultTypes));
        // llvm::raw_ostream &output = llvm::outs();
        // f.print(output);
      }
    }
  } else {
    // TODO: not in func args
    f.emitError("Has not implemented yet");
  }
}

void HCLLoopTransformation::runBufferAt(FuncOp &f,
                                        hcl::BufferAtOp &bufferAtOp) {
  // 1) get schedule
  auto target = bufferAtOp.target(); // return a Value type
  int axis = bufferAtOp.axis();
  const auto stage_name =
      bufferAtOp.stage().getDefiningOp()->getAttr("stage_name");

  // 2) Traverse all the nested loops and find the requested one
  bool isDone = false;
  for (auto rootForOp : f.getOps<AffineForOp>()) {
    if (stage_name == rootForOp->getAttr("stage_name").cast<StringAttr>()) {
      SmallVector<AffineForOp, 6> forOps;
      SmallVector<StringRef, 6> nameArr;
      // TODO: test if the requested loop has the target tensor
      if (isDone)
        break;
      bool isFound = findContiguousNestedLoops(rootForOp, forOps, nameArr);
      if (!isFound) {
        f.emitError("Cannot find nested loops for buffer_at");
        return;
      }
      SmallVector<AffineForOp, 6> nonReductionForOps;
      SmallVector<StringRef, 6> nonReductionNameArr;
      int firstReductionIdx = -1;
      for (std::size_t i = 0, e = forOps.size(); i != e; ++i) {
        if (!forOps[i]->hasAttr("reduction")) {
          nonReductionForOps.push_back(forOps[i]);
          nonReductionNameArr.push_back(
              forOps[i]->getAttr("loop_name").cast<StringAttr>().getValue());
        } else {
          if (firstReductionIdx == -1)
            firstReductionIdx = i;
        }
      }
      if (firstReductionIdx == -1)
        firstReductionIdx = forOps.size() - 1;
      if (axis >= 0 && ((std::size_t)(axis + 1) >= forOps.size())) {
        f.emitError("Cannot buffer at the inner-most loop: axis=")
            << std::to_string(axis)
            << " inner-most axis=" << std::to_string(forOps.size() - 1);
        return;
      }
      if (axis >= 0 && axis >= firstReductionIdx) {
        f.emitError("Cannot buffer inside the reduction loops: axis=")
            << std::to_string(axis)
            << ", first reduction axis=" << std::to_string(firstReductionIdx);
        return;
      }
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
        OpBuilder builder(forOps[firstReductionIdx]);
        Location loc_front = forOps[firstReductionIdx].getLoc();
        // TODO: support more data types
        mlir::Type elementType = builder.getF32Type();
        SmallVector<Value, 4> memIndices;
        // a) initialization
        // buffer only has one element
        auto buf = builder.create<memref::AllocOp>(
            loc_front, MemRefType::get({1}, elementType));
        auto zero = builder.create<ConstantOp>(
            loc_front, elementType, builder.getFloatAttr(elementType, 0.0));
        // no need to create an explicit loop
        auto idx = builder.create<ConstantIndexOp>(loc_front, 0);
        memIndices.push_back(idx);
        builder.create<AffineStoreOp>(loc_front, zero, buf, memIndices);

        // b) rewrite the original buffer
        // TODO: possible bug: replace uses before an untraversed op
        SmallVector<Operation *, 10> opToRemove;
        for (Operation &op :
             forOps[firstReductionIdx].getBody()->getOperations()) {
          if (auto load = dyn_cast<AffineLoadOp>(op)) {
            if (load.getOperand(0) != target) {
              continue;
            }
            // TODO: support more dimensions
            OpBuilder mid_builder(&op);
            memIndices.clear();
            memIndices.push_back(idx);
            auto new_load =
                mid_builder.create<AffineLoadOp>(op.getLoc(), buf, memIndices);
            op.replaceAllUsesWith(new_load);
            opToRemove.push_back(&op);
          } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
            if (store.getOperand(1) != target) {
              continue;
            }
            // TODO: support more dimensions
            OpBuilder mid_builder(&op);
            memIndices.clear();
            memIndices.push_back(idx);
            mid_builder.create<AffineStoreOp>(op.getLoc(), op.getOperand(0),
                                              buf, memIndices);
            opToRemove.push_back(&op);
          }
        }
        for (Operation *op : opToRemove) {
          op->erase();
        }

        // c) write back
        //    no need to create an explicit loop
        memIndices.clear();
        memIndices.push_back(idx);
        auto load_from_buf =
            builder.create<AffineLoadOp>(loc_front, buf, memIndices);
        memIndices.clear();
        for (int i = 0; i < firstReductionIdx; ++i) {
          memIndices.push_back(forOps[i].getInductionVar());
        }
        builder.create<AffineStoreOp>(loc_front, load_from_buf, target,
                                      memIndices);

        // d) move the original loop in the middle
        forOps[firstReductionIdx]->moveBefore(load_from_buf);

      } else { // not the inner-most non-reduction axis
        OpBuilder builder(forOps[axis + 1]);
        Location loc_front = forOps[axis + 1].getLoc();
        unsigned int ub = nonReductionForOps[axis + 1].getConstantUpperBound();
        // TODO: support more data types
        mlir::Type elementType = builder.getF32Type();
        SmallVector<Value, 4> memIndices;
        // a.1) allocate buffer
        auto buf = builder.create<memref::AllocOp>(
            loc_front, MemRefType::get({ub}, elementType));
        auto zero = builder.create<ConstantOp>(
            loc_front, elementType, builder.getFloatAttr(elementType, 0.0));

        // a.2) create initialization loop
        //      need to create an explicit loop
        AffineForOp initLoop = builder.create<AffineForOp>(loc_front, 0, ub);

        // a.3) do the initialization
        OpBuilder init_builder(
            &(*(initLoop.getBody()->getOperations().begin())));
        memIndices.push_back(initLoop.getInductionVar());
        init_builder.create<AffineStoreOp>(initLoop.getLoc(), zero, buf,
                                           memIndices);

        // b) rewrite the original buffer
        SmallVector<Operation *, 10> opToRemove;
        forOps[axis + 1].walk([&](Operation *op) {
          if (auto load = dyn_cast<AffineLoadOp>(op)) {
            if (load.getOperand(0) != target) {
              return;
            }
            // TODO: support more dimensions
            OpBuilder mid_builder(op);
            memIndices.clear();
            memIndices.push_back(
                nonReductionForOps[axis + 1].getInductionVar());
            auto new_load =
                mid_builder.create<AffineLoadOp>(op->getLoc(), buf, memIndices);
            op->replaceAllUsesWith(new_load);
            opToRemove.push_back(op);
          } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
            if (store.getOperand(1) != target) {
              return;
            }
            // TODO: support more dimensions
            OpBuilder mid_builder(op);
            memIndices.clear();
            memIndices.push_back(
                nonReductionForOps[axis + 1].getInductionVar());
            mid_builder.create<AffineStoreOp>(op->getLoc(), op->getOperand(0),
                                              buf, memIndices);
            opToRemove.push_back(op);
          }
        });
        for (Operation *op : opToRemove) {
          op->erase();
        }

        // c) write back
        Location loc_back =
            std::prev(forOps[axis + 1].getBody()->getOperations().end())
                ->getLoc();
        AffineForOp writeBackLoop =
            builder.create<AffineForOp>(loc_back, 0, ub);
        OpBuilder back_builder(
            &(*(writeBackLoop.getBody()->getOperations().begin())));
        memIndices.clear();
        memIndices.push_back(writeBackLoop.getInductionVar());
        auto load_from_buf = back_builder.create<AffineLoadOp>(
            writeBackLoop.getLoc(), buf, memIndices);

        memIndices.clear();
        for (int i = 0; i < axis + 1; ++i) {
          memIndices.push_back(nonReductionForOps[i].getInductionVar());
        }
        memIndices.push_back(writeBackLoop.getInductionVar());
        back_builder.create<AffineStoreOp>(writeBackLoop.getLoc(),
                                           load_from_buf, target, memIndices);

        // d) move the original loop between the two loops
        forOps[axis + 1]->moveBefore(writeBackLoop);
        // Add names to loops
        SmallVector<std::string, 6> newNameArr;
        newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_init");
        newNameArr.push_back(nonReductionNameArr[axis + 1].str() + "_back");
        SmallVector<AffineForOp, 6> newLoops{initLoop, writeBackLoop};
        addNamesToLoops(newLoops, newNameArr);
        // automatic pipelining
        SmallVector<AffineForOp, 6> twoLoops{initLoop, writeBackLoop};
        SmallVector<int, 6> II{1, 1};
        addIntAttrsToLoops(twoLoops, II, "pipeline_ii");
      }
      isDone = true;
      break;
    }
  }
}

void HCLLoopTransformation::runOnFunction() {
  FuncOp f = getFunction();
  SmallVector<Operation *, 10> opToRemove;
  // schedule should preverse orders, thus traverse one by one
  // the following shows the dispatching logic
  for (Operation &op : f.getOps()) {
    if (auto new_op = dyn_cast<hcl::SplitOp>(op)) {
      runSplitting(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::TileOp>(op)) {
      runTiling(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::ReorderOp>(op)) {
      runReordering(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::UnrollOp>(op)) {
      runUnrolling(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::PipelineOp>(op)) {
      runPipelining(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::ParallelOp>(op)) {
      runParallel(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::FuseOp>(op)) {
      runFusing(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::ComputeAtOp>(op)) {
      runComputeAt(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::PartitionOp>(op)) {
      runPartition(f, new_op);
      opToRemove.push_back(&op);
    } else if (auto new_op = dyn_cast<hcl::BufferAtOp>(op)) {
      runBufferAt(f, new_op);
      opToRemove.push_back(&op);
    }
  }
  // remove schedule operations (from back to front)
  std::reverse(opToRemove.begin(), opToRemove.end());
  std::set<Operation *> handleToRemove;
  for (Operation *op : opToRemove) {
    if (auto new_op = dyn_cast<hcl::SplitOp>(op)) {
      handleToRemove.insert(new_op.loop().getDefiningOp());
    } else if (auto new_op = dyn_cast<hcl::TileOp>(op)) {
      handleToRemove.insert(new_op.x_loop().getDefiningOp());
      handleToRemove.insert(new_op.y_loop().getDefiningOp());
    } else if (auto new_op = dyn_cast<hcl::FuseOp>(op)) {
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

namespace mlir {
namespace hcl {
// Register Loop Tiling Pass
void registerHCLLoopTransformationPass() {
  PassRegistration<HCLLoopTransformation>();
}

// Create A Loop Tiling Pass
std::unique_ptr<mlir::Pass> createHCLLoopTransformationPass() {
  return std::make_unique<HCLLoopTransformation>();
}
} // namespace hcl
} // namespace mlir