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
      SmallVector<AffineForOp, 6>& resForOps,
      unsigned depth);
  bool addNamesToLoops(
      SmallVector<AffineForOp, 6>& forOps,
      const SmallVector<std::string, 6>& nameArr);
};

} // namespace

bool HCLLoopTransformation::findContiguousNestedLoops(
      const AffineForOp& rootAffineForOp,
      const SmallVector<StringRef, 6>& nameArr,
      SmallVector<AffineForOp, 6>& resForOps,
      unsigned depth) {
  AffineForOp forOp = rootAffineForOp;
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
      if(!isFound && findContiguousNestedLoops(forOp,nameArr,forOps,2))
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
    const auto loop1_name = reorderOp.loop1().getType().cast<hcl::LoopHandleType>().getLoopName().str();
    const auto loop2_name = reorderOp.loop2().getType().cast<hcl::LoopHandleType>().getLoopName().str();
    SmallVector<AffineForOp, 6> forOps;
    SmallVector<unsigned, 6> permMap;
    unsigned int curr_depth = 0;
    std::map<std::string, unsigned> loop_map;
    std::vector<std::string> name_vec;
    f.walk([&](AffineForOp rootAffineForOp) { // from the inner most!
      std::string curr_loop_name = rootAffineForOp->getAttr("loop_name").cast<StringAttr>().getValue().str();
      loop_map[curr_loop_name] = curr_depth;
      name_vec.push_back(curr_loop_name);
      curr_depth++;
    });
    std::reverse(name_vec.begin(),name_vec.end());
    for (auto name : name_vec) { // need to reverse
      loop_map[name] = curr_depth - 1 - loop_map[name];
    }
    // TODO: traverse all input arguments (loops)
    unsigned loop1_idx = loop_map[loop1_name];
    loop_map[loop1_name] = loop_map[loop2_name];
    loop_map[loop2_name] = loop1_idx;
    // get permMap
    for (auto name : name_vec) {
      unsigned idx = loop_map[name];
      std::cout << idx << std::endl;
      permMap.push_back(idx);
    }
    SmallVector<AffineForOp, 6> nest;
    for (auto forOp : f.getOps<AffineForOp>()) {
      std::cout << "t " << forOp->getAttr("loop_name").cast<StringAttr>().getValue().str() << std::endl;
      // Get the maximal perfect nest.
      getPerfectlyNestedLoops(nest, forOp);
      // Permute if the nest's size is consistent with the specified
      // permutation.
      if (nest.size() >= 2 && nest.size() == permMap.size()) {
        std::cout << "permute" << std::endl;
        permuteLoops(nest, permMap);
      } else {
        // TODO: raise error
      }
      break; // only the outer-most loop
    }
  }
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
    const auto loop1_name = fuseOp.loop1().getType().cast<hcl::LoopHandleType>().getLoopName();
    const auto loop2_name = fuseOp.loop2().getType().cast<hcl::LoopHandleType>().getLoopName();

    // 2) Traverse all the nested loops and find the requested one
    AffineForOp loop_to_be_destroyed;
    f.walk([&](AffineForOp rootAffineForOp) {
      SmallVector<AffineForOp, 6> forOps;
      AffineForOp rootForOp = rootAffineForOp;
      Attribute attr = rootAffineForOp->getAttr("loop_name");
      if (loop1_name == attr.cast<StringAttr>().getValue()) {
        for (unsigned i = 0; i < 2; ++i) {
          if (i == 1) {
            Attribute attr_y = rootForOp->getAttr("loop_name");
            const StringRef curr_y_loop = attr_y.cast<StringAttr>().getValue();
            if (curr_y_loop != loop2_name)
              break;
          }
          forOps.push_back(rootForOp);
          Block &body = rootForOp.region().front();
          if (body.begin() != std::prev(body.end(), 2))
            break;

          rootForOp = dyn_cast<AffineForOp>(&body.front());
          if (!rootForOp)
            break;
        }
        // LoopUtils.cpp interchangeLoops
        auto &forOpBBody = forOps[1].getBody()->getOperations();
        Location loc = rootAffineForOp.getLoc();
        OpBuilder b(rootAffineForOp.getOperation());
        AffineForOp fusedLoop = b.create<AffineForOp>(loc, 0, 0);
        // OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
        // OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
        // fusedLoop.setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
        // fusedLoop.setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
        fusedLoop.setConstantLowerBound(0);
        fusedLoop.setConstantUpperBound(forOps[0].getConstantUpperBound() * forOps[1].getConstantUpperBound());
        fusedLoop.setStep(1);
        auto &fusedLoopBody = fusedLoop.getBody()->getOperations();
        // put forOpBBody from forOpBBody.begin() to std::prev(forOpBBody.end()) before fusedLoopBody.begin()
        forOpBBody.splice(fusedLoopBody.begin(), forOpBBody, forOpBBody.begin(),
                    std::prev(forOpBBody.end()));
        // Add names to new loop
        std::string new_name = loop1_name.str() + "_" + loop2_name.str() + "_fused";
        fusedLoop->setAttr("loop_name", StringAttr::get(fusedLoop->getContext(), new_name));
        // Replace original IVs with intra-tile loop IVs.
        // forOps[0].replaceAllUsesWith(fusedLoop.getInductionVar());
        // forOps[1].replaceAllUsesWith(fusedLoop.getInductionVar());
        // TODO: remove the original loop (bug)
        loop_to_be_destroyed = rootAffineForOp;
      }
    });
    // loop_to_be_destroyed.erase();
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