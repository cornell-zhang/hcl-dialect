#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
                                 SmallVector<StringRef, 6> &nameArr, int depth);
  bool addNamesToLoops(SmallVector<AffineForOp, 6> &forOps,
                       const SmallVector<std::string, 6> &nameArr);
};

} // namespace

bool HCLLoopTransformation::findContiguousNestedLoops(
    const AffineForOp &rootAffineForOp, SmallVector<AffineForOp, 6> &resForOps,
    SmallVector<StringRef, 6> &nameArr, int depth = -1) {
  // depth = -1 means traverses all the inner loops
  AffineForOp forOp = rootAffineForOp;
  Attribute attr = forOp->getAttr("loop_name");
  const StringRef curr_loop = attr.cast<StringAttr>().getValue();
  unsigned int sizeNameArr = nameArr.size();
  if (sizeNameArr != 0)
    depth = sizeNameArr;
  else if (depth == -1)
    depth = 0x3f3f3f3f;
  resForOps.clear();
  for (int i = 0; i < depth; ++i) {
    if (!forOp)
      return false;

    Attribute attr = forOp->getAttr("loop_name");
    const StringRef curr_loop = attr.cast<StringAttr>().getValue();
    if (sizeNameArr != 0 && curr_loop != nameArr[i])
      return false;

    resForOps.push_back(forOp);
    if (sizeNameArr == 0)
      nameArr.push_back(curr_loop);
    Block &body = forOp.region().front();
    // if (body.begin() != std::prev(body.end(), 2)) // perfectly nested
    //   break;

    forOp = dyn_cast<AffineForOp>(&body.front());
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
  const auto loop_name =
      splitOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

  // 2) Traverse all the nested loops and find the requested one
  //    and split the loop
  f.walk([&](AffineForOp forOp) {
    // mlir/IR/Operation.h
    Attribute attr = forOp->getAttr("loop_name");
    if (loop_name == attr.cast<StringAttr>().getValue()) {
      SmallVector<AffineForOp, 6> forOps;
      forOps.push_back(forOp);
      SmallVector<unsigned, 6> tileSizes;
      tileSizes.push_back(factor);
      if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
        return signalPassFailure();
    }
  });

  // 3) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(loop_name.str() + ".outer");
  newNameArr.push_back(loop_name.str() + ".inner");
  addNamesToLoops(tiledNest, newNameArr);
}

void HCLLoopTransformation::runTiling(FuncOp &f, hcl::TileOp &tileOp) {
  // 1) get schedule
  unsigned int x_factor = tileOp.x_factor();
  unsigned int y_factor = tileOp.y_factor();
  const StringRef x_loop =
      tileOp.x_loop().getType().cast<hcl::LoopHandleType>().getLoopName();
  const StringRef y_loop =
      tileOp.y_loop().getType().cast<hcl::LoopHandleType>().getLoopName();
  SmallVector<AffineForOp, 6> forOps;
  SmallVector<unsigned, 6> tileSizes;
  tileSizes.push_back(x_factor);
  tileSizes.push_back(y_factor);
  SmallVector<StringRef, 6> nameArr;
  nameArr.push_back(x_loop);
  nameArr.push_back(y_loop);

  // 2) Traverse all the nested loops and find the requested one
  //    and do tiling
  bool isFound = false;
  f.walk([&](AffineForOp forOp) {
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
  // try tiling
  SmallVector<AffineForOp, 6> tiledNest;
  if (failed(tilePerfectlyNested(forOps, tileSizes, &tiledNest)))
    return signalPassFailure();

  // 3) Add names to new loops
  SmallVector<std::string, 6> newNameArr;
  newNameArr.push_back(x_loop.str() + ".outer");
  newNameArr.push_back(x_loop.str() + ".inner");
  newNameArr.push_back(y_loop.str() + ".outer");
  newNameArr.push_back(y_loop.str() + ".inner");
  addNamesToLoops(tiledNest, newNameArr);
}

void HCLLoopTransformation::runReordering(FuncOp &f,
                                          hcl::ReorderOp &reorderOp) {
  // 1) get schedule
  const auto loopsToBeReordered = reorderOp.loops(); // operand_range

  // 2) get all the loop names and id mapping
  SmallVector<AffineForOp, 6> forOps;
  SmallVector<unsigned, 6> permMap;
  std::map<std::string, unsigned> name2id;
  std::vector<std::string> origNameVec;
  unsigned int curr_depth = 0;
  f.walk([&](AffineForOp rootAffineForOp) { // from the inner most!
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

  // 3) traverse all the input arguments that need to be reordered and construct
  // permMap possible inputs: a) # arguments = # loops: (i,j,k)->(k,j,i) b) #
  // arguments != # loops:
  //    input (k,i), but should be the same as a)
  // 3.1) map input arguments to the corresponding loop names
  std::vector<std::string> toBeReorderedNameVec;
  for (auto loop : loopsToBeReordered) {
    toBeReorderedNameVec.push_back(
        loop.getType().cast<hcl::LoopHandleType>().getLoopName().str());
  }
  // 3.2) traverse the original loop nests and create a new order for the loops
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
  // TODO: a) multiple stages
  //       b) bug: cannot permute the outer-most loop
  //       c) imperfect loops
  SmallVector<AffineForOp, 6> nest;
  for (auto forOp : f.getOps<AffineForOp>()) {
    // Get the maximal perfect nest
    getPerfectlyNestedLoops(nest, forOp);
    // Permute if the nest's size is consistent with the specified permutation
    if (nest.size() >= 2 && nest.size() == permMap.size()) {
      permuteLoops(nest, permMap);
    } else {
      f.emitError("Cannot permute the loops");
      return signalPassFailure();
    }
    break; // only the outer-most loop
  }
}

void HCLLoopTransformation::runUnrolling(FuncOp &f, hcl::UnrollOp &unrollOp) {
  // 1) get schedule
  unsigned int factor = unrollOp.factor();
  const auto loop_name =
      unrollOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

  // 2) Traverse all the nested loops and find the requested one
  f.walk([&](AffineForOp forOp) {
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
}

void HCLLoopTransformation::runParallel(FuncOp &f,
                                        hcl::ParallelOp &parallelOp) {
  // 1) get schedule
  const auto loop_name =
      parallelOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

  // 2) Traverse all the nested loops and find the requested one
  f.walk([&](AffineForOp forOp) {
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
}

void HCLLoopTransformation::runPipelining(FuncOp &f,
                                          hcl::PipelineOp &pipelineOp) {
  // 1) get schedule
  unsigned int ii = pipelineOp.ii();
  const auto loop_name =
      pipelineOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

  // 2) Traverse all the nested loops and find the requested one
  f.walk([&](AffineForOp forOp) {
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
  SmallVector<StringRef, 6> nameArr;
  for (auto loop : loopsToBeFused) {
    nameArr.push_back(loop.getType().cast<hcl::LoopHandleType>().getLoopName());
  }

  // 2) Traverse all the nested loops and find the requested ones
  AffineForOp loopToBeDestroyed;
  SmallVector<AffineForOp, 6> forOps;
  bool isFound = false;
  f.walk([&](AffineForOp forOp) {
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

  // 3) construct new loop
  MutableArrayRef<AffineForOp> loops =
      llvm::makeMutableArrayRef(forOps.data(), sizeOfFusedLoops);
  coalesceLoops(loops);

  // 4) add name to the new loop
  std::string new_name;
  for (auto forOp : forOps) {
    new_name +=
        forOp->getAttr("loop_name").cast<StringAttr>().getValue().str() + "_";
  }
  new_name += "fused";
  loops[0]->setAttr("loop_name",
                    StringAttr::get(loops[0]->getContext(), new_name));
}

void HCLLoopTransformation::runComputeAt(FuncOp &f,
                                         hcl::ComputeAtOp &computeAtOp) {
  // 1) get schedule
  const auto loop1_name =
      computeAtOp.loop1().getType().cast<hcl::LoopHandleType>().getLoopName();
  const auto loop2_name =
      computeAtOp.loop2().getType().cast<hcl::LoopHandleType>().getLoopName();

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
  auto memref = bufferAtOp.target(); // return a Value type
  int axis = bufferAtOp.axis();
  // 2) Traverse all the nested loops and find the requested one
  SmallVector<AffineForOp, 6> forOps;
  SmallVector<StringRef, 6> nameArr;
  bool isDone = false;
  for (auto forOp : f.getOps<AffineForOp>()) {
    if (isDone)
      break;
    findContiguousNestedLoops(forOp, forOps, nameArr, axis + 2);
    // TODO: Add operations and update loop bounds
    // a) initalization
    OpBuilder builder(forOps[axis + 1]);
    Location loc_front = forOps[axis + 1].getLoc();
    AffineForOp initLoop = builder.create<AffineForOp>(loc_front, 0, 1);
    // b) write back
    Location loc_back =
        std::prev(forOps[axis + 1].getBody()->getOperations().end())->getLoc();
    AffineForOp writeBackLoop = builder.create<AffineForOp>(loc_back, 0, 1);
    forOps[axis + 1]->moveBefore(writeBackLoop);
    // Add names to loops
    SmallVector<std::string, 6> newNameArr;
    newNameArr.push_back(nameArr[axis + 1].str() + "_init");
    newNameArr.push_back(nameArr[axis + 1].str());
    newNameArr.push_back(nameArr[axis + 1].str() + "_back");
    SmallVector<AffineForOp, 6> newLoops{initLoop, forOps[axis + 1],
                                         writeBackLoop};
    addNamesToLoops(newLoops, newNameArr);
    isDone = true;
    break;
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
  for (Operation *op : opToRemove) {
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