//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The HCL-MLIR Authors.
//
// Modified from the ScaleHLS project
//
//===----------------------------------------------------------------------===//

#include "hcl/Support/Utils.h"

using namespace mlir;
using namespace hcl;

//===----------------------------------------------------------------------===//
// HLSCpp attribute utils
//===----------------------------------------------------------------------===//

/// Parse loop directives.
Attribute hcl::getLoopDirective(Operation *op, std::string name) {
  return op->getAttr(name);
}

StringRef hcl::getLoopName(AffineForOp &forOp) {
  return forOp->getAttr("loop_name").cast<StringAttr>().getValue();
}

void hcl::setLoopName(AffineForOp &forOp, std::string loop_name) {
  forOp->setAttr("loop_name", StringAttr::get(forOp->getContext(), loop_name));
}

void hcl::setStageName(AffineForOp &forOp, StringRef stage_name) {
  forOp->setAttr("stage_name",
                 StringAttr::get(forOp->getContext(), stage_name));
}

std::vector<std::string> hcl::split_names(const std::string &arg_names) {
  std::stringstream ss(arg_names);
  std::vector<std::string> args;
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    args.push_back(substr);
  }
  return args;
}

/// Parse other attributes.
SmallVector<int64_t, 8> hcl::getIntArrayAttrValue(Operation *op,
                                                  StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = attr.dyn_cast<IntegerAttr>())
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

bool hcl::setIntAttr(SmallVector<AffineForOp, 6> &forOps,
                     const SmallVector<int, 6> &attr_arr,
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

bool hcl::setLoopNames(SmallVector<AffineForOp, 6> &forOps,
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

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

LogicalResult hcl::getStage(FuncOp &func, AffineForOp &forOp,
                            StringRef stage_name) {
  for (auto rootForOp : func.getOps<AffineForOp>()) {
    if (stage_name ==
        rootForOp->getAttr("stage_name").cast<StringAttr>().getValue()) {
      forOp = rootForOp;
      return success();
    }
  }
  return failure();
}

static unsigned getChildLoopNum(Operation *op);

int hcl::getLoop(AffineForOp &forOp, StringRef loop_name) {
  // return the axis id
  auto currentLoop = forOp;
  int cnt = -1;
  while (true) {
    cnt++;
    if (getLoopName(currentLoop) == loop_name) {
      forOp = currentLoop;
      return cnt;
    }

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return -1;
}

bool hcl::findContiguousNestedLoops(const AffineForOp &rootAffineForOp,
                                    SmallVector<AffineForOp, 6> &resForOps,
                                    SmallVector<StringRef, 6> &nameArr,
                                    int depth, bool countReductionLoops) {
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

/// Collect all load and store operations in the block and return them in "map".
// void hcl::getMemAccessesMap(Block &block, MemAccessesMap &map) {
//   for (auto &op : block) {
//     if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
//       map[MemRefAccess(&op).memref].push_back(&op);

//     else if (op.getNumRegions()) {
//       // Recursively collect memory access operations in each block.
//       for (auto &region : op.getRegions())
//         for (auto &block : region)
//           getMemAccessesMap(block, map);
//     }
//   }
// }

// Check if the lhsOp and rhsOp are in the same block. If so, return their
// ancestors that are located at the same block. Note that in this check,
// AffineIfOp is transparent.
Optional<std::pair<Operation *, Operation *>>
hcl::checkSameLevel(Operation *lhsOp, Operation *rhsOp) {
  // If lhsOp and rhsOp are already at the same level, return true.
  if (lhsOp->getBlock() == rhsOp->getBlock())
    return std::pair<Operation *, Operation *>(lhsOp, rhsOp);

  // Helper to get all surrounding AffineIfOps.
  auto getSurroundIfs =
      ([&](Operation *op, SmallVector<Operation *, 4> &nests) {
        nests.push_back(op);
        auto currentOp = op;
        while (true) {
          if (auto parentOp = currentOp->getParentOfType<AffineIfOp>()) {
            nests.push_back(parentOp);
            currentOp = parentOp;
          } else
            break;
        }
      });

  SmallVector<Operation *, 4> lhsNests;
  SmallVector<Operation *, 4> rhsNests;

  getSurroundIfs(lhsOp, lhsNests);
  getSurroundIfs(rhsOp, rhsNests);

  // If any parent of lhsOp and any parent of rhsOp are at the same level,
  // return true.
  for (auto lhs : lhsNests)
    for (auto rhs : rhsNests)
      if (lhs->getBlock() == rhs->getBlock())
        return std::pair<Operation *, Operation *>(lhs, rhs);

  return Optional<std::pair<Operation *, Operation *>>();
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned hcl::getCommonSurroundingLoops(Operation *A, Operation *B,
                                        AffineLoopBand *band) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getLoopIVs(*A, &loopsA);
  getLoopIVs(*B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
    if (band != nullptr)
      band->push_back(loopsB[i]);
  }
  return numCommonLoops;
}

/// Calculate the upper and lower bound of "bound" if possible.
Optional<std::pair<int64_t, int64_t>>
hcl::getBoundOfAffineBound(AffineBound bound) {
  auto boundMap = bound.getMap();
  if (boundMap.isSingleConstant()) {
    auto constBound = boundMap.getSingleConstantResult();
    return std::pair<int64_t, int64_t>(constBound, constBound);
  }

  // For now, we can only handle one result affine bound.
  if (boundMap.getNumResults() != 1)
    return Optional<std::pair<int64_t, int64_t>>();

  auto context = boundMap.getContext();
  SmallVector<int64_t, 4> lbs;
  SmallVector<int64_t, 4> ubs;
  for (auto operand : bound.getOperands()) {
    // Only if the affine bound operands are induction variable, the calculation
    // is possible.
    if (!isForInductionVar(operand))
      return Optional<std::pair<int64_t, int64_t>>();

    // Only if the owner for op of the induction variable has constant bound,
    // the calculation is possible.
    auto ifOp = getForInductionVarOwner(operand);
    if (!ifOp.hasConstantBounds())
      return Optional<std::pair<int64_t, int64_t>>();

    auto lb = ifOp.getConstantLowerBound();
    auto ub = ifOp.getConstantUpperBound();
    auto step = ifOp.getStep();

    lbs.push_back(lb);
    ubs.push_back(ub - 1 - (ub - 1 - lb) % step);
  }

  // TODO: maybe a more efficient algorithm.
  auto operandNum = bound.getNumOperands();
  SmallVector<int64_t, 16> results;
  for (unsigned i = 0, e = pow(2, operandNum); i < e; ++i) {
    SmallVector<AffineExpr, 4> replacements;
    for (unsigned pos = 0; pos < operandNum; ++pos) {
      if (i >> pos % 2 == 0)
        replacements.push_back(getAffineConstantExpr(lbs[pos], context));
      else
        replacements.push_back(getAffineConstantExpr(ubs[pos], context));
    }
    auto newExpr =
        bound.getMap().getResult(0).replaceDimsAndSymbols(replacements, {});

    if (auto constExpr = newExpr.dyn_cast<AffineConstantExpr>())
      results.push_back(constExpr.getValue());
    else
      return Optional<std::pair<int64_t, int64_t>>();
  }

  auto minmax = std::minmax_element(results.begin(), results.end());
  return std::pair<int64_t, int64_t>(*minmax.first, *minmax.second);
}

/// Return the layout map of "memrefType".
AffineMap hcl::getLayoutMap(MemRefType memrefType) {
  // Check whether the memref has layout map.
  auto memrefMaps = memrefType.getLayout();
  if (memrefMaps.getAffineMap().isIdentity())
    return (AffineMap) nullptr;

  return memrefMaps.getAffineMap();
}

bool hcl::isFullyPartitioned(MemRefType memrefType, int axis) {
  if (memrefType.getRank() == 0)
    return true;

  bool fullyPartitioned = false;
  if (auto layoutMap = getLayoutMap(memrefType)) {
    SmallVector<int64_t, 8> factors;
    getPartitionFactors(memrefType, &factors);

    // Case 1: Use floordiv & mod
    auto shapes = memrefType.getShape();
    if (axis == -1) // all the dimensions
      fullyPartitioned =
          factors == SmallVector<int64_t, 8>(shapes.begin(), shapes.end());
    else
      fullyPartitioned = factors[axis] == shapes[axis];

    // Case 2: Partition index is an identity function
    if (axis == -1) {
      bool flag = true;
      for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
        auto expr = layoutMap.getResult(dim);
        if (!expr.isa<AffineDimExpr>()) {
          flag = false;
          break;
        }
      }
      fullyPartitioned |= flag;
    } else {
      auto expr = layoutMap.getResult(axis);
      fullyPartitioned |= expr.isa<AffineDimExpr>();
    }
  }

  return fullyPartitioned;
}

// Calculate partition factors through analyzing the "memrefType" and return
// them in "factors". Meanwhile, the overall partition number is calculated and
// returned as well.
int64_t hcl::getPartitionFactors(MemRefType memrefType,
                                 SmallVector<int64_t, 8> *factors) {
  auto shape = memrefType.getShape();
  auto layoutMap = getLayoutMap(memrefType);
  int64_t accumFactor = 1;

  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    int64_t factor = 1;

    if (layoutMap) {
      auto expr = layoutMap.getResult(dim);

      if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>())
        if (auto rhsExpr = binaryExpr.getRHS().dyn_cast<AffineConstantExpr>()) {
          if (expr.getKind() == AffineExprKind::Mod)
            factor = rhsExpr.getValue();
          else if (expr.getKind() == AffineExprKind::FloorDiv)
            factor = (shape[dim] + rhsExpr.getValue() - 1) / rhsExpr.getValue();
        }
    }

    accumFactor *= factor;
    if (factors != nullptr)
      factors->push_back(factor);
  }

  return accumFactor;
}

/// This is method for finding the number of child loops which immediatedly
/// contained by the input operation.
static unsigned getChildLoopNum(Operation *op) {
  unsigned childNum = 0;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        if (isa<AffineForOp>(op))
          ++childNum;

  return childNum;
}

/// Get the whole loop band given the innermost loop and return it in "band".
static void getLoopBandFromInnermost(AffineForOp forOp, AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
}

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp hcl::getLoopBandFromOutermost(AffineForOp forOp,
                                          AffineLoopBand &band) {
  band.clear();
  auto currentLoop = forOp;
  while (true) {
    band.push_back(currentLoop);

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return band.back();
}

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void hcl::getLoopBands(Block &block, AffineLoopBands &bands,
                       bool allowHavingChilds) {
  bands.clear();
  block.walk([&](AffineForOp loop) {
    auto childNum = getChildLoopNum(loop);

    if (childNum == 0 || (childNum > 1 && allowHavingChilds)) {
      AffineLoopBand band;
      getLoopBandFromInnermost(loop, band);
      bands.push_back(band);
    }
  });
}

void hcl::getArrays(Block &block, SmallVectorImpl<Value> &arrays,
                    bool allowArguments) {
  // Collect argument arrays.
  if (allowArguments)
    for (auto arg : block.getArguments()) {
      if (arg.getType().isa<MemRefType>())
        arrays.push_back(arg);
    }

  // Collect local arrays.
  for (auto &op : block.getOperations()) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op))
      arrays.push_back(op.getResult(0));
  }
}

Optional<unsigned> hcl::getAverageTripCount(AffineForOp forOp) {
  if (auto optionalTripCount = getConstantTripCount(forOp))
    return optionalTripCount.getValue();
  else {
    // TODO: A temporary approach to estimate the trip count. For now, we take
    // the average of the upper bound and lower bound of trip count as the
    // estimated trip count.
    auto lowerBound = getBoundOfAffineBound(forOp.getLowerBound());
    auto upperBound = getBoundOfAffineBound(forOp.getUpperBound());

    if (lowerBound && upperBound) {
      auto lowerTripCount =
          upperBound.getValue().second - lowerBound.getValue().first;
      auto upperTripCount =
          upperBound.getValue().first - lowerBound.getValue().second;
      return (lowerTripCount + upperTripCount + 1) / 2;
    } else
      return Optional<unsigned>();
  }
}

bool hcl::checkDependence(Operation *A, Operation *B) {
  return true;
  // TODO: Fix the following
  //   AffineLoopBand commonLoops;
  //   unsigned numCommonLoops = getCommonSurroundingLoops(A, B, &commonLoops);

  //   // Traverse each loop level to find dependencies.
  //   for (unsigned depth = numCommonLoops; depth > 0; depth--) {
  //     // Skip all parallel loop level.
  //     if (auto loopAttr = getLoopDirective(commonLoops[depth - 1]))
  //       if (loopAttr.getParallel())
  //         continue;

  //     FlatAffineValueConstraints depConstrs;
  //     DependenceResult result = checkMemrefAccessDependence(
  //         MemRefAccess(A), MemRefAccess(B), depth, &depConstrs,
  //         /*dependenceComponents=*/nullptr);
  //     if (hasDependence(result))
  //       return true;
  //   }

  //   return false;
}

static bool gatherLoadOpsAndStoreOps(AffineForOp forOp,
                                     SmallVectorImpl<Operation *> &loadOps,
                                     SmallVectorImpl<Operation *> &storeOps) {
  bool hasIfOp = false;
  forOp.walk([&](Operation *op) {
    if (auto load = dyn_cast<AffineReadOpInterface>(op))
      loadOps.push_back(op);
    else if (auto load = dyn_cast<memref::LoadOp>(op))
      loadOps.push_back(op);
    else if (auto store = dyn_cast<AffineWriteOpInterface>(op))
      storeOps.push_back(op);
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      storeOps.push_back(op);
    else if (isa<AffineIfOp>(op))
      hasIfOp = true;
  });
  return !hasIfOp;
}

bool hcl::analyzeDependency(const AffineForOp &forOpA,
                            const AffineForOp &forOpB,
                            SmallVectorImpl<Dependency> &dependency) {
  SmallVector<Operation *, 4> readOpsA;
  SmallVector<Operation *, 4> writeOpsA;
  SmallVector<Operation *, 4> readOpsB;
  SmallVector<Operation *, 4> writeOpsB;

  if (!gatherLoadOpsAndStoreOps(forOpA, readOpsA, writeOpsA)) {
    return false;
  }

  if (!gatherLoadOpsAndStoreOps(forOpB, readOpsB, writeOpsB)) {
    return false;
  }

  DenseSet<Value> OpAReadMemRefs;
  DenseSet<Value> OpAWriteMemRefs;
  DenseSet<Value> OpBReadMemRefs;
  DenseSet<Value> OpBWriteMemRefs;

  for (Operation *op : readOpsA) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpAReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsA) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpAWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Operation *op : readOpsB) {
    if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpBReadMemRefs.insert(loadOp.getMemRef());
    }
  }

  for (Operation *op : writeOpsB) {
    if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      OpBWriteMemRefs.insert(storeOp.getMemRef());
    }
  }

  for (Value memref : OpBReadMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::RAR);
  }

  for (Value memref : OpBWriteMemRefs) {
    if (OpAWriteMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAW);
    else if (OpAReadMemRefs.count(memref) > 0)
      dependency.push_back(Dependency::WAR);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// PtrLikeMemRefAccess Struct Definition
//===----------------------------------------------------------------------===//

PtrLikeMemRefAccess::PtrLikeMemRefAccess(Operation *loadOrStoreOpInst) {
  Operation *opInst = nullptr;
  SmallVector<Value, 4> indices;

  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();

    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();

    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }

  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);

  accessMap.reset(map, operands);
}

bool PtrLikeMemRefAccess::operator==(const PtrLikeMemRefAccess &rhs) const {
  if (memref != rhs.memref || impl != rhs.impl)
    return false;

  if (impl == rhs.impl && impl && rhs.impl)
    return true;

  AffineValueMap diff;
  AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

// Returns the index of 'op' in its block.
inline static unsigned getBlockIndex(Operation &op) {
  unsigned index = 0;
  for (auto &opX : *op.getBlock()) {
    if (&op == &opX)
      break;
    ++index;
  }
  return index;
}

// Returns a string representation of 'sliceUnion'.
std::string hcl::getSliceStr(const mlir::ComputationSliceState &sliceUnion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  // Slice insertion point format [loop-depth, operation-block-index]
  unsigned ipd = mlir::getNestingDepth(&*sliceUnion.insertPoint);
  unsigned ipb = getBlockIndex(*sliceUnion.insertPoint);
  os << "insert point: (" << std::to_string(ipd) << ", " << std::to_string(ipb)
     << ")";
  assert(sliceUnion.lbs.size() == sliceUnion.ubs.size());
  os << " loop bounds: ";
  for (unsigned k = 0, e = sliceUnion.lbs.size(); k < e; ++k) {
    os << '[';
    sliceUnion.lbs[k].print(os);
    os << ", ";
    sliceUnion.ubs[k].print(os);
    os << "] ";
  }
  return os.str();
}

Value hcl::castInteger(OpBuilder builder, Location loc, Value input,
                       Type srcType, Type tgtType, bool is_signed) {
  int oldWidth = srcType.cast<IntegerType>().getWidth();
  int newWidth = tgtType.cast<IntegerType>().getWidth();
  Value casted;
  if (newWidth < oldWidth) {
    // trunc
    casted = builder.create<arith::TruncIOp>(loc, tgtType, input);
  } else if (newWidth > oldWidth) {
    // extend
    if (is_signed) {
      casted = builder.create<arith::ExtSIOp>(loc, tgtType, input);
    } else {
      casted = builder.create<arith::ExtUIOp>(loc, tgtType, input);
    }
  } else {
    casted = input;
  }
  return casted;
}

/* CastIntMemRef
 * Allocate a new Int MemRef of target width and build a
 * AffineForOp loop nest to load, cast, store the elements
 * from oldMemRef to newMemRef.
 */
Value hcl::castIntMemRef(OpBuilder &builder, Location loc,
                         const Value &oldMemRef, size_t newWidth,
                         bool unsign, bool replace,
                         const Value &dstMemRef) {
  // If newWidth == oldWidth, no need to cast.
  if (newWidth == oldMemRef.getType().cast<MemRefType>().getElementType()
                       .cast<IntegerType>()
                       .getWidth()) {
    return oldMemRef;
  }
  // first, alloc new memref
  MemRefType oldMemRefType = oldMemRef.getType().cast<MemRefType>();
  Type newElementType = builder.getIntegerType(newWidth);
  MemRefType newMemRefType =
      oldMemRefType.clone(newElementType).cast<MemRefType>();
  Value newMemRef;
  if (!dstMemRef) {
    newMemRef = builder.create<memref::AllocOp>(loc, newMemRefType);
  }
  // replace all uses
  if (replace)
    oldMemRef.replaceAllUsesWith(newMemRef);
  // build loop nest
  SmallVector<int64_t, 4> lbs(oldMemRefType.getRank(), 0);
  SmallVector<int64_t, 4> steps(oldMemRefType.getRank(), 1);
  size_t oldWidth =
      oldMemRefType.getElementType().cast<IntegerType>().getWidth();
  buildAffineLoopNest(
      builder, loc, lbs, oldMemRefType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value v = nestedBuilder.create<AffineLoadOp>(loc, oldMemRef, ivs);
        Value casted;
        if (newWidth < oldWidth) {
          // trunc
          casted =
              nestedBuilder.create<arith::TruncIOp>(loc, newElementType, v);
        } else if (newWidth > oldWidth) {
          // extend
          if (unsign) {
            casted =
                nestedBuilder.create<arith::ExtUIOp>(loc, newElementType, v);
          } else {
            casted =
                nestedBuilder.create<arith::ExtSIOp>(loc, newElementType, v);
          }
        } else {
          casted = v; // no cast happened
        }
        if (dstMemRef) {
          nestedBuilder.create<AffineStoreOp>(loc, casted, dstMemRef, ivs);
        } else {
          nestedBuilder.create<AffineStoreOp>(loc, casted, newMemRef, ivs);
        }
      });
  return newMemRef;
}
