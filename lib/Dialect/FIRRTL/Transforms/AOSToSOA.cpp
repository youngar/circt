//===- AOSToSOA.cpp - Array of Structs to Struct of Arrays -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AOSToSOA pass, which takes any structure embedded in
// an array, and converts it to an array embedded within a struct.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "firrtl-aos-to-soa"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

namespace {
class LiftBundlesVisitor
    : public FIRRTLVisitor<LiftBundlesVisitor, LogicalResult> {
public:
  explicit LiftBundlesVisitor(MLIRContext *);

  LogicalResult visit(FModuleOp);

  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitStmt;

  LogicalResult visitUnhandledOp(Operation *op);

  LogicalResult visitInvalidOp(Operation *) { return failure(); }

  LogicalResult visitDecl(InstanceOp);
  LogicalResult visitDecl(MemOp);
  LogicalResult visitDecl(NodeOp);
  LogicalResult visitDecl(RegOp);
  LogicalResult visitDecl(RegResetOp);
  LogicalResult visitDecl(WireOp);

  template <typename OpTy>
  void handleConnect(OpTy op);
  LogicalResult visitStmt(ConnectOp);
  LogicalResult visitStmt(StrictConnectOp);

  LogicalResult visitExpr(AggregateConstantOp);
  LogicalResult visitExpr(VectorCreateOp);
  LogicalResult visitExpr(SubindexOp);
  LogicalResult visitExpr(SubfieldOp);

private:
  /// Type Conversion
  Type convertType(Type);
  FIRRTLBaseType convertType(FIRRTLBaseType);
  FIRRTLBaseType convertType(FIRRTLBaseType, SmallVector<unsigned> &);

  /// Aggregate Constant Conversion Helpers
  Attribute convertAggregate(Type, ArrayRef<Value>);
  Attribute convertVectorOfBundle();

  Attribute convertConstant(Type type, Attribute fields);
  Attribute convertVectorConstant(FVectorType type, ArrayAttr fields);
  Attribute convertBundleConstant(BundleType type, ArrayAttr fields);
  Attribute convertBundleInVectorConstant(BundleType elementType,
                                          ArrayRef<Attribute> elements);

  /// Path Rewriting
  std::pair<SmallVector<Value>, bool> buildPath(ImplicitLocOpBuilder &builder,
                                                Value oldValue, Value newValue,
                                                unsigned fieldID);

  /// Operand Fixup
  std::pair<SmallVector<Value>, bool> fixOperand(ImplicitLocOpBuilder &, Value);
  SmallVector<Value> explode(Value);
  void explode(ImplicitLocOpBuilder &, Value, SmallVector<Value> &);

  /// fix a ground type operand.
  Value fixAtomicOperand(ImplicitLocOpBuilder &, Value);

  /// Read-only / RHS Operand Fixup. Value MUST be passive.
  Value fixROperand(ImplicitLocOpBuilder &, Value);
  Value fixROperand(ImplicitLocOpBuilder &, FieldRef);

  Value sinkVecDimIntoOperands(ImplicitLocOpBuilder &, FIRRTLBaseType,
                               const SmallVectorImpl<Value> &);

  MLIRContext *context;
  SmallVector<Operation *> toDelete;

  /// A mapping from old values to their fixed up values.
  /// If a value is unchanged, it will be mapped to itself.
  /// If a value is present in the map, and does not map to itself, then
  /// it must be deleted.
  DenseMap<Value, Value> valueMap;

  /// A cache mapping uncoverted types to their soa-converted equivalents.
  DenseMap<FIRRTLBaseType, FIRRTLBaseType> typeMap;

  /// pull an access op from the cache if available, create the op if needed.
  Value getSubfield(Value value, unsigned index);
  Value getSubindex(Value value, unsigned index);
  Value getSubaccess(Value value, Value index);

  /// A cache of generated subfield/index/access operations
  DenseMap<std::pair<Value, unsigned>, Value> subfieldCache;
  DenseMap<std::pair<Value, unsigned>, Value> subindexCache;
  DenseMap<std::pair<Value, Value>, Value> subaccessCache;
};

} // end anonymous namespace

LiftBundlesVisitor::LiftBundlesVisitor(MLIRContext *context)
    : context(context) {}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type LiftBundlesVisitor::convertType(Type type) {
  if (auto firrtlType = type.dyn_cast<FIRRTLBaseType>())
    return convertType(firrtlType);
  return type;
}

FIRRTLBaseType LiftBundlesVisitor::convertType(FIRRTLBaseType type) {
  auto cached = typeMap.lookup(type);
  if (cached)
    return cached;

  SmallVector<unsigned> dimensions;
  auto converted = convertType(type, dimensions);

  typeMap.insert({type, converted});
  return converted;
}

// NOLINTNEXTLINE(misc-no-recursion)
FIRRTLBaseType
LiftBundlesVisitor::convertType(FIRRTLBaseType type,
                                SmallVector<unsigned> &dimensions) {
  // Vector Types
  if (auto vectorType = type.dyn_cast<FVectorType>(); vectorType) {
    dimensions.push_back(vectorType.getNumElements());
    auto converted = convertType(vectorType.getElementType(), dimensions);
    dimensions.pop_back();
    return converted;
  }

  // Bundle Types
  if (auto bundleType = type.dyn_cast<BundleType>(); bundleType) {
    SmallVector<BundleType::BundleElement> elements;
    for (auto element : bundleType.getElements()) {
      elements.push_back(BundleType::BundleElement(
          element.name, element.isFlip, convertType(element.type, dimensions)));
    }

    return BundleType::get(context, elements);
  }

  // Ground Types
  for (auto size : llvm::reverse(dimensions))
    type = FVectorType::get(type, size);
  return type;
}

//===----------------------------------------------------------------------===//
// Operand Fixup
//===----------------------------------------------------------------------===//

namespace {
struct VectorAccess {
  enum Kind { subindex, subaccess };

  VectorAccess(unsigned index) : index(index), kind(Kind::subindex) {}

  VectorAccess(Value value) : value(value), kind(Kind::subaccess) {}

  union {
    unsigned index;
    Value value;
  };
  Kind kind;
};
} // namespace

Value LiftBundlesVisitor::getSubfield(Value value, unsigned index) {
  Value result = subfieldCache[{value, index}];
  if (!result) {
    OpBuilder builder(context);
    builder.setInsertionPointAfterValue(value);
    // llvm::errs() << "    getSubfield: cache miss value=" << value
    //              << " index=" << index << "\n";
    result =
        builder.create<SubfieldOp>(value.getLoc(), value, index).getResult();
    subfieldCache[{value, index}] = result;
  }
  // llvm::errs() << "    getSubfield: " << result << "\n";
  return result;
}

Value LiftBundlesVisitor::getSubindex(Value value, unsigned index) {
  auto &result = subindexCache[{value, index}];
  if (!result) {
    OpBuilder builder(context);
    builder.setInsertionPointAfterValue(value);
    // llvm::errs() << "    getSubindex: cache miss value=" << value
    //              << " index=" << index << "\n";
    result =
        builder.create<SubindexOp>(value.getLoc(), value, index).getResult();
  }
  // llvm::errs() << "    getSubindex: " << result << "\n";
  return result;
}

Value LiftBundlesVisitor::getSubaccess(Value value, Value index) {
  auto &result = subaccessCache[{value, index}];
  if (!result) {
    OpBuilder builder(context);
    builder.setInsertionPointAfterValue(value);
    // llvm::errs() << "    getSubaccess: cache miss value=" << value
    //              << " index=" << index << "\n";
    result =
        builder.create<SubaccessOp>(value.getLoc(), value, index).getResult();
  }
  // llvm::errs() << "    getSubaccess: " << result << "\n";
  return result;
}

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::fixOperand(ImplicitLocOpBuilder &builder, Value value) {
  SmallVector<unsigned> bundleAccesses;
  SmallVector<VectorAccess> vectorAccesses;

  // Walk back through the subaccess ops to the canonical storage location.
  // Collect the path according to the type of access, splitting bundle accesses
  // ops from vector accesses ops.
  while (value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      break;
    if (auto subfieldOp = dyn_cast<SubfieldOp>(op)) {
      value = subfieldOp.getInput();
      bundleAccesses.push_back(subfieldOp.getFieldIndex());
      continue;
    }
    if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
      value = subindexOp.getInput();
      vectorAccesses.push_back(subindexOp.getIndex());
      continue;
    }
    if (auto subaccessOp = dyn_cast<SubaccessOp>(op)) {
      value = subaccessOp.getInput();
      vectorAccesses.push_back(subaccessOp.getIndex());
      continue;
    }
    break;
  }

  // Value now points at the original canonical storage location.
  // find it's converted equivalent.
  // llvm::errs() << "old value = " << value << "\n";

  value = valueMap[value];
  assert(value && "canonical storage location must have been converted");

  // llvm::errs() << "new value = " << value << "\n";

  // Replay the subaccess operations, in the converted order
  // (bundle accesses first, then vector accesses).
  for (auto fieldIndex : llvm::reverse(bundleAccesses))
    value = getSubfield(value, fieldIndex);

  /// If the current value is a bundle, but we have vector accesses to replay,
  // we must explode the bundle and apply the vector accesses to each leaf of
  // the bundle.

  SmallVector<Value> values;
  bool exploded = false;

  if (value.getType().isa<BundleType>() && !vectorAccesses.empty()) {
    explode(builder, value, values);
    exploded = true;
  } else {
    values.push_back(value);
    exploded = false;
  }

  /// Finally, replay the vector access operations.
  for (auto &value : values) {
    for (auto access : llvm::reverse(vectorAccesses)) {
      if (access.kind == VectorAccess::subindex) {
        value = getSubindex(value, access.index);
        continue;
      }
      if (access.kind == VectorAccess::subaccess) {
        value = getSubaccess(value, access.value);
        continue;
      }
    }
  }

  // llvm::errs() << "fixOperand: values:\n";
  // for (auto value : values)
  //   llvm::errs() << "    " << value << "\n";

  return {values, exploded};
}

void LiftBundlesVisitor::explode(ImplicitLocOpBuilder &builder, Value value,
                                 SmallVector<Value> &output) {
  auto type = value.getType();
  if (auto bundleType = type.dyn_cast<BundleType>()) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      auto field = getSubfield(value, i);
      explode(builder, field, output);
    }
  } else {
    output.push_back(value);
  }
}

SmallVector<Value> LiftBundlesVisitor::explode(Value value) {
  ImplicitLocOpBuilder builder(value.getLoc(), context);
  builder.setInsertionPointAfterValue(value);
  auto output = SmallVector<Value>();
  explode(builder, value, output);
  return output;
}

Value LiftBundlesVisitor::fixAtomicOperand(ImplicitLocOpBuilder &builder,
                                           Value value) {
  assert(value.getType().cast<FIRRTLBaseType>().isGround());
  auto [values, exploded] = fixOperand(builder, value);
  assert(!exploded);
  assert(values.size() == 1);
  return values[0];
}

//===----------------------------------------------------------------------===//
// Read-Only / RHS Operand Fixup
//===----------------------------------------------------------------------===//

Value LiftBundlesVisitor::fixROperand(ImplicitLocOpBuilder &builder,
                                      Value operand) {
  // llvm::errs() << "fixROperand: val=" << operand << "\n";

  auto [values, exploded] = fixOperand(builder, operand);
  if (!exploded) {
    // llvm::errs() << "fixROperand not exploded\n";
    return values.front();
  }
  // llvm::errs() << "fixROperand exploded\n";

  auto newType = convertType(operand.getType());
  return builder.create<BundleCreateOp>(operand.getLoc(), newType, values);
}

//===----------------------------------------------------------------------===//
// Base Case -- Any Regular Operation
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitUnhandledOp(Operation *op) {
  // llvm::errs() << "visitUnhandledOp: " << *op << "\n";

  ImplicitLocOpBuilder builder(op->getLoc(), op);
  bool changed = false;

  // Typical operations read from passive operands, only.
  // We can materialze any passive operand into a single value, potentially with
  // fresh intermediate bundle create ops in between.
  SmallVector<Value> newOperands;
  for (auto oldOperand : op->getOperands()) {
    auto newOperand = fixROperand(builder, oldOperand);
    changed |= (oldOperand != newOperand);
    newOperands.push_back(newOperand);
    // llvm::errs() << "visitUnhandledOp: old operand: " << oldOperand << "\n";
    // llvm::errs() << "visitUnhandledOp: new operand: " << newOperand << "\n";
  }

  /// We can rewrite the type of any result, but if any result type changes,
  /// then the operation will be cloned.
  SmallVector<Type> newTypes;
  for (auto oldResult : op->getResults()) {
    auto oldType = oldResult.getType();
    auto newType = convertType(oldType);
    changed |= oldType != newType;
    newTypes.push_back(newType);
  }

  if (changed) {
    auto *newOp = builder.clone(*op);
    newOp->setOperands(newOperands);
    for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
      auto newResult = newOp->getResult(i);
      newResult.setType(newTypes[i]);
      valueMap[op->getResult(i)] = newResult;
    }
    toDelete.push_back(op);
    op = newOp;
  } else {
    // As a safety precaution, all unchanged "canonical storage locations" must
    // be mapped to themselves.
    for (auto result : op->getResults())
      valueMap[result] = result;
  }

  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks())
      for (auto &op : block)
        if (failed(dispatchVisitor(&op)))
          return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitDecl(InstanceOp op) {
  llvm::errs() << "InstanceOp\n";

  auto changed = false;
  auto oldTypes = op->getResultTypes();
  SmallVector<Type> newTypes;
  for (auto oldType : oldTypes) {
    auto newType = convertType(oldType);
    if (oldType != newType)
      changed = true;
    newTypes.push_back(newType);
  }

  if (!changed) {
    for (auto result : op.getResults())
      valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);
  auto newOp = builder.create<InstanceOp>(
      op.getLoc(), newTypes, op.getModuleNameAttr(), op.getNameAttr(),
      op.getNameKindAttr(), op.getPortDirectionsAttr(), op.getPortNamesAttr(),
      op.getAnnotationsAttr(), op.getPortAnnotationsAttr(),
      op.getLowerToBindAttr(), op.getInnerSymAttr());

  for (size_t i = 0, e = op.getNumResults(); i < e; ++i)
    valueMap[op.getResult(i)] = newOp.getResult(i);

  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(MemOp op) {
  // llvm::errs() << "MemOp\n";

  auto changed = false;
  auto oldTypes = op->getResultTypes();
  SmallVector<Type> newTypes;
  for (auto oldType : oldTypes) {
    auto newType = convertType(oldType);
    if (oldType != newType)
      changed = true;
    newTypes.push_back(newType);
  }

  if (!changed) {
    for (auto result : op.getResults())
      valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);
  auto newOp = builder.create<MemOp>(
      op.getLoc(), newTypes, op.getReadLatencyAttr(), op.getWriteLatencyAttr(),
      op.getDepthAttr(), op.getRuwAttr(), op.getPortNamesAttr(),
      op.getNameAttr(), op.getNameKindAttr(), op.getAnnotationsAttr(),
      op.getPortAnnotationsAttr(), op.getInnerSymAttr(), op.getGroupIDAttr(),
      op.getInitAttr());

  for (size_t i = 0, e = op.getNumResults(); i < e; ++i)
    valueMap[op.getResult(i)] = newOp.getResult(i);

  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(NodeOp op) {
  // llvm::errs() << "NodeOp\n";

  auto changed = false;
  ImplicitLocOpBuilder builder(op.getLoc(), op);

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldInput = op.getInput();
  auto newInput = fixROperand(builder, oldInput);
  if (oldInput != newInput)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  auto newOp = builder.create<NodeOp>(
      op.getLoc(), newInput, op.getNameAttr(), op.getNameKindAttr(),
      op.getAnnotationsAttr(), op.getInnerSymAttr());

  valueMap[op.getResult()] = newOp.getResult();
  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(RegOp op) {
  bool changed = false;
  ImplicitLocOpBuilder builder(op.getLoc(), op);

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldClockVal = op.getClockVal();
  auto newClockVal = fixAtomicOperand(builder, oldClockVal);
  if (oldClockVal != newClockVal)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  auto newOp = builder.create<RegOp>(
      op.getLoc(), newType, newClockVal, op.getNameAttr(), op.getNameKindAttr(),
      op.getAnnotationsAttr(), op.getInnerSymAttr());

  toDelete.push_back(op);
  valueMap.insert({op.getResult(), newOp.getResult()});
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(RegResetOp op) {
  bool changed = false;
  ImplicitLocOpBuilder builder(op.getLoc(), op);

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldClockVal = op.getClockVal();
  auto newClockVal = fixAtomicOperand(builder, oldClockVal);
  if (oldClockVal != newClockVal)
    changed = true;

  auto oldResetSignal = op.getResetSignal();
  auto newResetSignal = fixAtomicOperand(builder, oldResetSignal);
  if (oldResetSignal != newResetSignal)
    changed = true;

  auto oldResetValue = op.getResetValue();
  auto newResetValue = fixROperand(builder, oldResetValue);
  if (oldResetValue != newResetValue)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  auto newOp = builder.create<RegResetOp>(
      op.getLoc(), newType, newClockVal, newResetSignal, newResetValue,
      op.getNameAttr(), op.getNameKindAttr(), op.getAnnotationsAttr(),
      op.getInnerSymAttr());

  toDelete.push_back(op);
  valueMap.insert({op.getResult(), newOp.getResult()});
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(WireOp op) {
  // llvm::errs() << "WireOp\n";
  auto changed = false;

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);
  auto newOp = builder.create<WireOp>(
      op.getLoc(), newType, op.getNameAttr(), op.getNameKindAttr(),
      op.getAnnotationsAttr(), op.getInnerSymAttr());

  valueMap[op.getResult()] = newOp.getResult();
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

template <typename OpTy>
void LiftBundlesVisitor::handleConnect(OpTy op) {
  // llvm::errs() << "handle connect\n";
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  llvm::errs() << op << "\n";

  auto oldLhs = op.getDest();
  auto oldLhsType = cast<FIRRTLBaseType>(oldLhs.getType());
  llvm::errs() << "oldLhsType = " << oldLhsType << "\n";

  auto [newLhs, lhsExploded] = fixOperand(builder, oldLhs);
  llvm::errs() << "exploded = " << lhsExploded << "\n";

  // Happy-path: The LHS did not explode, and is passive (so, not writing to the
  // RHS). We can guarantee that the rhs will resolve to a single unexploded
  // value.
  if (!lhsExploded && oldLhsType.isPassive()) {
    llvm::errs() << "!!!!!!should not be here\n";
    auto newRhs = fixROperand(builder, op.getSrc());
    builder.create<OpTy>(op.getLoc(), newLhs[0], newRhs);
    toDelete.push_back(op);
    return;
  }

  auto [newRhs, rhsExploded] = fixOperand(builder, op.getSrc());
  if (!rhsExploded && lhsExploded)
    newRhs = explode(newRhs[0]);

  if (!lhsExploded && rhsExploded)
    newLhs = explode(newLhs[0]);

  // llvm::errs() << "lhsSize = " << newLhs.size() << " rhsSize=" <<
  // newRhs.size()
  //              << "\n";

  assert(newLhs.size() == newRhs.size() &&
         "Something went wrong exploding the elements");

  // Emit connections between all leaf elements.
  auto newLhsType = convertType(oldLhsType);
  const auto *newLhsIt = newLhs.begin();
  const auto *newRhsIt = newRhs.begin();
  std::function<void(Type)> explodeConnect = [&](Type type) {
    if (auto bundleType = dyn_cast<BundleType>(type)) {
      for (auto &e : bundleType) {
        if (e.isFlip)
          std::swap(newLhsIt, newRhsIt);
        explodeConnect(e.type);
      }
    } else if (auto vectorType = type.dyn_cast<FVectorType>()) {
      explodeConnect(vectorType.getElementType());
    } else {

      auto newLHS = *newLhsIt++;
      auto newRHS = *newRhsIt++;
      // llvm::errs() << "lhs=" << newLHS << "\n";
      // llvm::errs() << "rhs=" << newRHS << "\n";
      builder.create<OpTy>(newLHS, newRHS);
    }
  };
  explodeConnect(newLhsType);

  toDelete.push_back(op);
}

LogicalResult LiftBundlesVisitor::visitStmt(ConnectOp op) {
  // llvm::errs() << "ConnectOp\n";
  handleConnect(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitStmt(StrictConnectOp op) {
  // llvm::errs() << "StrictConnectOp\n";
  handleConnect(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Conversion
//===----------------------------------------------------------------------===//

Attribute
LiftBundlesVisitor::convertBundleInVectorConstant(BundleType type,
                                                  ArrayRef<Attribute> fields) {
  auto numBundleFields = type.getNumElements();
  SmallVector<SmallVector<Attribute>> newBundleFields;
  newBundleFields.resize(numBundleFields);
  for (auto bundle : fields) {
    auto subfields = bundle.cast<ArrayAttr>();
    for (size_t i = 0; i < numBundleFields; ++i) {
      newBundleFields[i].push_back(subfields[i]);
    }
  }

  SmallVector<Attribute> newFieldAttrs;
  for (auto &newBundleField : newBundleFields) {
    newFieldAttrs.push_back(ArrayAttr::get(context, newBundleField));
  }
  return ArrayAttr::get(context, newFieldAttrs);
}

Attribute LiftBundlesVisitor::convertVectorConstant(FVectorType oldType,
                                                    ArrayAttr oldElements) {
  auto oldElementType = oldType.getElementType();
  auto newElementType = convertType(oldElementType);

  if (oldElementType == newElementType)
    if (auto bundleElementType = oldElementType.dyn_cast<BundleType>())
      return convertBundleInVectorConstant(bundleElementType,
                                           oldElements.getValue());

  SmallVector<Attribute> newElements;
  for (auto oldElement : oldElements) {
    newElements.push_back(convertConstant(oldElementType, oldElement));
  }

  auto bundleType = newElementType.cast<BundleType>();
  return convertBundleInVectorConstant(bundleType, newElements);
}

Attribute LiftBundlesVisitor::convertBundleConstant(BundleType type,
                                                    ArrayAttr fields) {
  SmallVector<Attribute> converted;
  auto elements = type.getElements();
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    converted.push_back(convertConstant(elements[i].type, fields[i]));
  }
  return ArrayAttr::get(context, converted);
}

Attribute LiftBundlesVisitor::convertConstant(Type type, Attribute value) {
  if (auto bundleType = type.dyn_cast<BundleType>())
    return convertBundleConstant(bundleType, value.cast<ArrayAttr>());

  if (auto vectorType = type.dyn_cast<FVectorType>())
    return convertVectorConstant(vectorType, value.cast<ArrayAttr>());

  return value;
}

LogicalResult LiftBundlesVisitor::visitExpr(AggregateConstantOp op) {
  auto oldValue = op.getResult();

  auto oldType = oldValue.getType();
  auto newType = convertType(oldType);
  if (oldType == newType) {
    valueMap[oldValue] = oldValue;
    return success();
  }

  auto fields = convertConstant(oldType, op.getFields()).cast<ArrayAttr>();

  OpBuilder builder(op);
  auto newOp =
      builder.create<AggregateConstantOp>(op.getLoc(), newType, fields);

  valueMap[oldValue] = newOp.getResult();
  toDelete.push_back(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Aggregate Create Ops
//===----------------------------------------------------------------------===//

Value LiftBundlesVisitor::sinkVecDimIntoOperands(
    ImplicitLocOpBuilder &builder, FIRRTLBaseType type,
    const SmallVectorImpl<Value> &values) {
  auto length = values.size();
  if (auto bundleType = type.dyn_cast<BundleType>()) {
    SmallVector<Value> newFields;
    SmallVector<BundleType::BundleElement> newElements;
    for (auto &[i, elt] : llvm::enumerate(bundleType)) {
      SmallVector<Value> subValues;
      for (auto v : values)
        subValues.push_back(builder.create<SubfieldOp>(v, i));
      auto newField = sinkVecDimIntoOperands(builder, elt.type, subValues);
      newFields.push_back(newField);
      newElements.emplace_back(elt.name, /*isFlip=*/false,
                               newField.getType().cast<FIRRTLBaseType>());
    }
    auto newType = BundleType::get(builder.getContext(), newElements);
    auto newBundle = builder.create<BundleCreateOp>(newType, newFields);
    return newBundle;
  }
  auto newType = FVectorType::get(type, length);
  return builder.create<VectorCreateOp>(newType, values);
}

LogicalResult LiftBundlesVisitor::visitExpr(VectorCreateOp op) {
  // llvm::errs() << "VectorCreateOp\n";

  ImplicitLocOpBuilder builder(op.getLoc(), op);

  auto oldType = op.getType();
  auto newType = convertType(oldType);

  if (oldType == newType) {
    auto changed = false;
    SmallVector<Value> newFields;
    for (auto oldField : op.getFields()) {
      auto newField = fixROperand(builder, oldField);
      // llvm::errs() << "new field : " << newField << "\n";
      if (oldField != newField)
        changed = true;
      newFields.push_back(newField);
    }

    if (!changed) {
      auto result = op.getResult();
      valueMap[result] = result;
      return success();
    }

    auto newOp =
        builder.create<VectorCreateOp>(op.getLoc(), newType, newFields);
    valueMap[op.getResult()] = newOp.getResult();
    toDelete.push_back(op);
    return success();
  }

  // OK, We are in for some pain!

  SmallVector<Value> convertedOldFields;
  for (auto oldField : op.getFields()) {
    auto convertedField = fixROperand(builder, oldField);
    // llvm::errs() << "new field : " << convertedField << "\n";
    convertedOldFields.push_back(convertedField);
  }

  auto value = sinkVecDimIntoOperands(
      builder, convertType(oldType.getElementType()), convertedOldFields);
  valueMap[op.getResult()] = value;
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pathing Ops
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitExpr(SubindexOp op) {
  // llvm::errs() << "SubindexOp\n";
  // auto rootValue = getFieldRefFromValue(op).getValue();
  // if (valueMap.count(rootValue))
  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(SubfieldOp op) {
  // llvm::errs() << "SubfieldOp\n";
  // auto rootValue = getFieldRefFromValue(op).getValue();
  // if (valueMap.count(rootValue))
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor Entrypoint
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visit(FModuleOp op) {
  BitVector portsToErase(op.getNumPorts() * 2);
  {
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    auto ports = op.getPorts();
    auto count = 0;
    for (auto &[index, port] : llvm::enumerate(ports)) {
      auto oldType = port.type;
      auto newType = convertType(oldType);
      if (newType == oldType)
        continue;

      auto newPort = port;
      newPort.type = newType;

      portsToErase[count + index] = true;
      newPorts.push_back({index + 1, newPort});
      ++count;
    }
    op.insertPorts(newPorts);
  }
  
  llvm::errs() << op.getNameAttr() << "\n";

  auto *body = op.getBodyBlock();
  for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
    if (portsToErase[i]) {
      auto oldArg = body->getArgument(i);
      auto newArg = body->getArgument(i + 1);
      valueMap[oldArg] = newArg;
    } else {
      auto oldArg = body->getArgument(i);
      valueMap[oldArg] = oldArg;
    }
  }

  // auto numPorts = op.getNumPorts();
  // SmallVector<Attribute> newPortTypes;
  // newPortTypes.resize(numPorts);
  // for (size_t i = 0; i < numPorts; ++i) {
  // if (failed(fixPort(op, i)))
  //   return failure();
  // }

  // op->setAttr("portTypes", ArrayAttr::get(op.getContext(), newPortTypes));

  // auto body = op.getBodyBlock();
  // for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
  //   auto argument = body->getArgument(i);
  //   argument.setType(newPortTypes[i].dyn_cast<TypeAttr>().getValue());
  // }

  // for (auto it = body->begin(), e = body->end(); it != e; ++it) {

  for (auto &op : *body) {
    if (failed(dispatchVisitor(&op)))
      return failure();
  }

  while (!toDelete.empty())
    toDelete.pop_back_val()->erase();
  op.erasePorts(portsToErase);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class AOSToSOAPass : public AOSToSOABase<AOSToSOAPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void AOSToSOAPass::runOnOperation() {
  llvm::errs() << "===========================\n"
               << "START\n"
               << "---------------------------\n";

  std::vector<FModuleOp> modules;
  llvm::append_range(modules, getOperation().getBody().getOps<FModuleOp>());
  auto result =
      failableParallelForEach(&getContext(), modules, [&](FModuleOp module) {
        LiftBundlesVisitor visitor(&getContext());
        return visitor.visit(module);
      });

  if (result.failed())
    signalPassFailure();
  // markAllAnalysesPreserved();
  // // auto changed = convertFModuleOp(getOperation());
  // auto changed = false;
  // if (!changed) {
  //   markAllAnalysesPreserved();
  // }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAOSToSOAPass() {
  return std::make_unique<AOSToSOAPass>();
}
