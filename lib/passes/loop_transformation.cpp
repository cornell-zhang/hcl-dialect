#include "hcl/HeteroCLDialect.h"
#include "hcl/HeteroCLOps.h"
#include "hcl/HeteroCLPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"

#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace mlir;

namespace {

struct HCLLoopTransformation : public PassWrapper<HCLLoopTransformation, FunctionPass> {

  void runOnFunction() override;
  
  StringRef getArgument() const final { return "hcl-loop-transformation"; }
  StringRef getDescription() const final {
    return "Loop transformation in HeteroCL";
  }

  void runSplitting();
  void runTiling();
  void runReordering();
  void runUnrolling();
  void runPipelining();
  void runParallel();
  void runFusing();
  void runComputeAt();
  // utils
  bool findContiguousNestedLoops(
      const AffineForOp& rootAffineForOp,
      const SmallVector<StringRef, 6>& nameArr,
      SmallVector<AffineForOp, 6>& resForOps);
  bool addNamesToLoops(
      SmallVector<AffineForOp, 6>& forOps,
      const SmallVector<std::string, 6>& nameArr);
};

} // namespace

bool HCLLoopTransformation::findContiguousNestedLoops(
      const AffineForOp& rootAffineForOp,
      const SmallVector<StringRef, 6>& nameArr,
      SmallVector<AffineForOp, 6>& resForOps) {
  AffineForOp forOp = rootAffineForOp;
  unsigned int depth = nameArr.size();
  resForOps.clear();
  for (unsigned i = 0; i < depth; ++i) {
    if (!forOp)
      return false;

    Attribute attr = forOp->getAttr("loop_name");
    const StringRef curr_loop = attr.cast<StringAttr>().getValue();
    if (curr_loop != nameArr[i])
      return false;

    resForOps.push_back(forOp);
    Block &body = forOp.region().front();
    // if (body.begin() != std::prev(body.end(), 2)) // perfectly nested
    //   break;

    forOp = dyn_cast<AffineForOp>(&body.front());
  }
  return true;
}

bool HCLLoopTransformation::addNamesToLoops(
      SmallVector<AffineForOp, 6>& forOps,
      const SmallVector<std::string, 6>& nameArr) {
  assert(forOps.size() == nameArr.size());
  unsigned cnt_loop = 0;
  for (AffineForOp newForOp : forOps) {
    newForOp->setAttr("loop_name", StringAttr::get(newForOp->getContext(), nameArr[cnt_loop]));
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
void HCLLoopTransformation::runSplitting() {
  FuncOp f = getFunction();
  SmallVector<AffineForOp, 6> tiledNest;
  for (hcl::SplitOp splitOp : f.getOps<hcl::SplitOp>()) {
    // 1) get schedule
    unsigned int factor = splitOp.factor();
    const auto loop_name = splitOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

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
    addNamesToLoops(tiledNest,newNameArr);
  }
  // 4) Remove the schedule operator
  // TODO: Fix bug
  // for (hcl::SplitOp splitOp : f.getOps<hcl::SplitOp>())
  //   splitOp.erase();
}

void HCLLoopTransformation::runTiling() {
  FuncOp f = getFunction();
  for (hcl::TileOp tileOp : f.getOps<hcl::TileOp>()) {
    // 1) get schedule
    unsigned int x_factor = tileOp.x_factor();
    unsigned int y_factor = tileOp.y_factor();
    const StringRef x_loop = tileOp.x_loop().getType().cast<hcl::LoopHandleType>().getLoopName();
    const StringRef y_loop = tileOp.y_loop().getType().cast<hcl::LoopHandleType>().getLoopName();
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
      if(!isFound && findContiguousNestedLoops(forOp,nameArr,forOps))
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
    addNamesToLoops(tiledNest,newNameArr);
  }
  // 4) Remove the schedule operator
  // TODO: may have !NodePtr->isKnownSentinel() bug
  // for (hcl::TileOp tileOp : f.getOps<hcl::TileOp>())
  //   tileOp.erase();
}

void HCLLoopTransformation::runReordering() {
  FuncOp f = getFunction();
  for (hcl::ReorderOp reorderOp : f.getOps<hcl::ReorderOp>()) {
    // 1) get schedule
    const auto loopsToBeReordered = reorderOp.loops(); // operand_range

    // 2) get all the loop names and id mapping
    SmallVector<AffineForOp, 6> forOps;
    SmallVector<unsigned, 6> permMap;
    std::map<std::string, unsigned> name2id;
    std::vector<std::string> origNameVec;
    unsigned int curr_depth = 0;
    f.walk([&](AffineForOp rootAffineForOp) { // from the inner most!
      std::string curr_loop_name = rootAffineForOp->getAttr("loop_name").cast<StringAttr>().getValue().str();
      name2id[curr_loop_name] = curr_depth;
      origNameVec.push_back(curr_loop_name);
      curr_depth++;
    });
    std::reverse(origNameVec.begin(),origNameVec.end());
    for (auto name : origNameVec) { // need to reverse
      name2id[name] = curr_depth - 1 - name2id[name];
    }

    // 3) traverse all the input arguments that need to be reordered and construct permMap
    // possible inputs:
    // a) # arguments = # loops: (i,j,k)->(k,j,i)
    // b) # arguments != # loops:
    //    input (k,i), but should be the same as a)
    // 3.1) map input arguments to the corresponding loop names
    std::vector<std::string> toBeReorderedNameVec;
    for (auto loop : loopsToBeReordered) {
      toBeReorderedNameVec.push_back(loop.getType().cast<hcl::LoopHandleType>().getLoopName().str());
    }
    // 3.2) traverse the original loop nests and create a new order for the loops
    //     since the input arguments may not cover all the loops
    //     so this step is needed for creating permMap
    unsigned int cntInArgs = 0;
    for (auto name : origNameVec) {
      auto iter = std::find(toBeReorderedNameVec.begin(), toBeReorderedNameVec.end(), name);
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
  // 5) TODO: Remove the schedule operator
}

void HCLLoopTransformation::runUnrolling() {
  FuncOp f = getFunction();
  for (hcl::UnrollOp unrollOp : f.getOps<hcl::UnrollOp>()) {
    // 1) get schedule
    unsigned int factor = unrollOp.factor();
    const auto loop_name = unrollOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

    // 2) Traverse all the nested loops and find the requested one
    f.walk([&](AffineForOp forOp) {
      Attribute attr = forOp->getAttr("loop_name");
      if (loop_name == attr.cast<StringAttr>().getValue()) {
        forOp->setAttr("unroll",
                      IntegerAttr::get(
                        IntegerType::get(
                          forOp->getContext(),
                          32,
                          IntegerType::SignednessSemantics::Signless),
                        factor)
                      );
      }
    });
  }
}

void HCLLoopTransformation::runParallel() {
  FuncOp f = getFunction();
  for (hcl::ParallelOp parallelOp : f.getOps<hcl::ParallelOp>()) {
    // 1) get schedule
    const auto loop_name = parallelOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

    // 2) Traverse all the nested loops and find the requested one
    f.walk([&](AffineForOp forOp) {
      Attribute attr = forOp->getAttr("loop_name");
      if (loop_name == attr.cast<StringAttr>().getValue()) {
        forOp->setAttr("parallel",
                      IntegerAttr::get(
                        IntegerType::get(
                          forOp->getContext(),
                          32,
                          IntegerType::SignednessSemantics::Signless),
                        1) // true
                      );
      }
    });
  }
}

void HCLLoopTransformation::runPipelining() {
  FuncOp f = getFunction();
  for (hcl::PipelineOp pipelineOp : f.getOps<hcl::PipelineOp>()) {
    // 1) get schedule
    unsigned int ii = pipelineOp.ii();
    const auto loop_name = pipelineOp.loop().getType().cast<hcl::LoopHandleType>().getLoopName();

    // 2) Traverse all the nested loops and find the requested one
    f.walk([&](AffineForOp forOp) {
      Attribute attr = forOp->getAttr("loop_name");
      if (loop_name == attr.cast<StringAttr>().getValue()) {
        forOp->setAttr("pipeline_ii",
                      IntegerAttr::get(
                        IntegerType::get(
                          forOp->getContext(),
                          32,
                          IntegerType::SignednessSemantics::Signless),
                        ii)
                      );
      }
    });
  }
}

// Notice hcl.fuse (fuses nested loops) is different from affine.fuse,
// which fuses contiguous loops. This is actually the case of hcl.compute_at.
void HCLLoopTransformation::runFusing() {
  FuncOp f = getFunction();
  for (hcl::FuseOp fuseOp : f.getOps<hcl::FuseOp>()) {
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
      if(!isFound && findContiguousNestedLoops(forOp,nameArr,forOps))
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
    // 3.1) create a fused loop
    Location loc = forOps[0].getLoc();
    OpBuilder builder(forOps[0].getOperation()); // create before forOps[0]
    AffineForOp fusedLoop = builder.create<AffineForOp>(loc, 0, 0);
    // 3.2) set upper/lower bounds and step
    // TODO: only support [0 to prod(factors) step 1] constant pattern now
    // OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    // OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    // fusedLoop.setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    // fusedLoop.setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    fusedLoop.setConstantLowerBound(0);
    unsigned int prod = 1;
    for (auto forOp : forOps) {
      prod *= forOp.getConstantUpperBound();
    }
    fusedLoop.setConstantUpperBound(prod);
    fusedLoop.setStep(1);
    Operation *topLoop = forOps[0].getOperation();
    auto &fusedLoopBody = fusedLoop.getBody()->getOperations();
    fusedLoopBody.splice(fusedLoop.getBody()->begin(),
                         topLoop->getBlock()->getOperations(),
                         topLoop);
    // 3.3) Put original loop body into the fused loop
    auto &forOpInnerMostBody = forOps[sizeOfFusedLoops - 1].getBody()->getOperations();
    // put forOpInnerMostBody from forOpInnerMostBody.begin() to std::prev(forOpInnerMostBody.end()) before fusedLoopBody.begin()
    forOpInnerMostBody.splice(fusedLoopBody.begin(), // fusedLoop.getBody()->begin()
                              forOpInnerMostBody,
                              forOpInnerMostBody.begin(),
                              std::prev(forOpInnerMostBody.end()));
    // 3.4) add name to the new loop
    std::string new_name;
    for (auto forOp : forOps) {
      new_name += forOp->getAttr("loop_name").cast<StringAttr>().getValue().str() + "_";
    }
    new_name += "fused";
    fusedLoop->setAttr("loop_name", StringAttr::get(fusedLoop->getContext(), new_name));
    
    // 4) TODO: remove the original loop (bug)
    for (int i = 0; i < forOps.size(); ++i){
      auto blockArg = forOps[i].getInductionVar();
      blockArg.replaceAllUsesWith(fusedLoop.getInductionVar());
    }
    forOps[0].erase();
    llvm::raw_ostream &output = llvm::outs();
    f.print(output);
  }
}

void HCLLoopTransformation::runComputeAt() {
  FuncOp f = getFunction();
  for (hcl::ComputeAtOp computeAtOp : f.getOps<hcl::ComputeAtOp>()) {
    // 1) get schedule
    const auto loop1_name = computeAtOp.loop1().getType().cast<hcl::LoopHandleType>().getLoopName();
    const auto loop2_name = computeAtOp.loop2().getType().cast<hcl::LoopHandleType>().getLoopName();

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
    for (int i = 3; i >= 1; --i){ // TODO: Change depth
      FusionResult result = canFuseLoops(forOps[0], forOps[1], i/*depth*/, &sliceUnion);
      if (result.value == FusionResult::Success) {
        fuseLoops(forOps[0], forOps[1], sliceUnion);
        forOps[0].erase();
        std::cout << std::to_string(i) << " yes" << std::endl;
        return;
      } else
        std::cout << std::to_string(i) << " no" << std::endl;
    }
  }
}

// https://github.com/llvm/llvm-project/blob/release/13.x/mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
void HCLLoopTransformation::runOnFunction()  {
  runSplitting();
  runTiling();
  runReordering();
  runUnrolling();
  runPipelining();
  runParallel();
  runFusing();
  runComputeAt();
}

namespace mlir {
namespace hcl{
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