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

  LogicalResult visitUnhandledOp(Operation *op) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      assert(operand.getType() == convertType(operand.getType()) &&
             "This is probably a bogus assert");
      op->setOperand(index, fixROperand(operand));
    }

    for (auto result : op->getResults()) {
      assert(result.getType() == convertType(result.getType()) &&
             "The op should be cloned if any result type changes");
      valueMap[result] = result;
    }

    return success();
  }

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
  LogicalResult visitExpr(BundleCreateOp);
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
  std::pair<SmallVector<Value>, bool> buildPath(Value oldValue, Value newValue,
                                                unsigned fieldID);

  /// Operand Fixup
  std::pair<SmallVector<Value>, bool> fixOperand(FieldRef);
  std::pair<SmallVector<Value>, bool> fixOperand(Value);
  SmallVector<Value> explode(Value);
  void explode(OpBuilder, Value, SmallVector<Value> &);
  Value fixAtomicOperand(Value value);

  /// Read-only / RHS Operand Fixup
  Value fixROperand(Value);
  Value fixROperand(FieldRef);

  Value sinkVecDimIntoOperands(ImplicitLocOpBuilder &builder,
                               FIRRTLBaseType type,
                               const SmallVectorImpl<Value> &values);

  /// Utility
  bool remapped(Value value) { return valueMap.lookup(value) != value; }

  MLIRContext *context;
  DenseSet<Operation *> toDelete;

  /// A mapping from old values to their fixed up values.
  /// If a value is unchanged, it will be mapped to itself.
  /// If a value is present in the map, and does not map to itself, then
  /// it must be deleted.
  DenseMap<Value, Value> valueMap;
  // DenseMap<FieldRef, SmallVector<Value, 1>> fields;
  // DenseMap<FieldRef, SmallVector<FieldRef>> mappings;
  DenseMap<FIRRTLBaseType, FIRRTLBaseType> typeMap;
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
// Path Rewriting
//===----------------------------------------------------------------------===//

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::buildPath(Value oldValue, Value newValue,
                              unsigned fieldID) {

  llvm::errs() << "buildPath\n";
  llvm::errs() << "  old=" << oldValue << "\n";
  llvm::errs() << "  new=" << newValue << "\n";
  llvm::errs() << "  fld=" << fieldID << "\n";

  auto loc = oldValue.getLoc();
  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(newValue);

  SmallVector<unsigned> subfieldIndices;
  SmallVector<unsigned> subindexIndices;
  // bool inVector = false;
  // bool inVectorOfBundle = false;

  auto oldType = oldValue.getType();
  // auto newType = newValue.getType();

  // llvm::errs() << getFieldName(FieldRef(oldValue, fieldID)) << "\n";

  auto type = oldType;
  while (fieldID != 0) {
    if (auto bundle = type.dyn_cast<BundleType>()) {
      auto index = bundle.getIndexForFieldID(fieldID);
      fieldID -= bundle.getFieldID(index);
      subfieldIndices.push_back(index);
      // inVectorOfBundle = inVector;
      type = bundle.getElementType(index);
    } else {
      auto vector = type.cast<FVectorType>();
      auto index = vector.getIndexForFieldID(fieldID);
      fieldID -= vector.getFieldID(index);
      subindexIndices.push_back(index);
      // inVector = true;
      type = vector.getElementType();
    }
  }

  auto value = newValue;
  for (auto index : subfieldIndices) {
    auto op = builder.create<SubfieldOp>(loc, value, index);
    // op->dump();
    value = op.getResult();
  }

  SmallVector<Value> values;
  bool exploded = false;

  if (value.getType().isa<BundleType>() && 0 < subindexIndices.size()) {
    assert(0 < subindexIndices.size());
    assert(value.getType().isa<BundleType>());
    explode(builder, value, values);
    exploded = true;
  } else {
    values.push_back(value);
    exploded = false;
  }

  for (size_t i = 0, e = values.size(); i < e; ++i) {
    auto value = values[i];
    for (auto index : subindexIndices) {
      auto op = builder.create<SubindexOp>(loc, value, index);
      value = op.getResult();
    }
    values[i] = value;
  }

  return {values, exploded};
}

//===----------------------------------------------------------------------===//
// Operand Fixup
//===----------------------------------------------------------------------===//

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::fixOperand(FieldRef ref) {
  llvm::errs() << "fixOperand ref=" << ref.getFieldID()
               << " val=" << ref.getValue() << "\n";
  auto value = ref.getValue();
  auto converted = valueMap.lookup(value);
  assert(converted);
  return buildPath(value, converted, ref.getFieldID());
}

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::fixOperand(Value value) {
  return fixOperand(getFieldRefFromValue(value));
}

void LiftBundlesVisitor::explode(OpBuilder builder, Value value,
                                 SmallVector<Value> &output) {
  auto type = value.getType();
  if (auto bundleType = type.dyn_cast<BundleType>()) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      auto op = builder.create<SubfieldOp>(value.getLoc(), value, i);
      explode(builder, op, output);
    }
  } else {
    output.push_back(value);
  }
}

SmallVector<Value> LiftBundlesVisitor::explode(Value value) {
  auto builder = OpBuilder(context);
  builder.setInsertionPointAfterValue(value);
  auto output = SmallVector<Value>();
  explode(builder, value, output);
  return output;
}

Value LiftBundlesVisitor::fixAtomicOperand(Value value) {
  auto [values, exploded] = fixOperand(value);
  assert(!exploded);
  assert(values.size() == 1);
  return values[0];
}

//===----------------------------------------------------------------------===//
// Read-Only / RHS Operand Fixup
//===----------------------------------------------------------------------===//

Value LiftBundlesVisitor::fixROperand(Value operand) {
  auto ref = getFieldRefFromValue(operand);

  llvm::errs() << "fixROperand ref=" << ref.getFieldID()
               << " val=" << ref.getValue() << "\n";

  auto [values, exploded] = fixOperand(ref);
  if (!exploded) {
    llvm::errs() << "fixROperand not exploded\n";
    return values.front();
  }
  llvm::errs() << "fixROperand exploded\n";

  auto newType = convertType(operand.getType());

  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(operand);
  return builder.create<BundleCreateOp>(operand.getLoc(), newType, values);
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

  toDelete.insert(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(MemOp op) {
  llvm::errs() << "MemOp\n";

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

  toDelete.insert(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(NodeOp op) {
  llvm::errs() << "NodeOp\n";

  auto changed = false;

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldInput = op.getInput();
  auto newInput = fixROperand(oldInput);
  if (oldInput != newInput)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);
  auto newOp = builder.create<NodeOp>(
      op.getLoc(), newInput, op.getNameAttr(), op.getNameKindAttr(),
      op.getAnnotationsAttr(), op.getInnerSymAttr());

  valueMap[op.getResult()] = newOp.getResult();
  toDelete.insert(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(RegOp op) {
  bool changed = false;

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldClockVal = op.getClockVal();
  auto newClockVal = fixAtomicOperand(oldClockVal);
  if (oldClockVal != newClockVal)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);
  auto newOp = builder.create<RegOp>(
      op.getLoc(), newType, newClockVal, op.getNameAttr(), op.getNameKindAttr(),
      op.getAnnotationsAttr(), op.getInnerSymAttr());

  toDelete.insert(op);
  valueMap.insert({op.getResult(), newOp.getResult()});
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(RegResetOp op) {
  bool changed = false;

  auto oldType = op.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    changed = true;

  auto oldClockVal = op.getClockVal();
  auto newClockVal = fixAtomicOperand(oldClockVal);
  if (oldClockVal != newClockVal)
    changed = true;

  auto oldResetSignal = op.getResetSignal();
  auto newResetSignal = fixAtomicOperand(oldResetSignal);
  if (oldResetSignal != newResetSignal)
    changed = true;

  auto oldResetValue = op.getResetValue();
  auto newResetValue = fixROperand(oldResetValue);
  if (oldResetValue != newResetValue)
    changed = true;

  if (!changed) {
    auto result = op.getResult();
    valueMap[result] = result;
    return success();
  }

  OpBuilder builder(op);

  auto newOp = builder.create<RegResetOp>(
      op.getLoc(), newType, newClockVal, newResetSignal, newResetValue,
      op.getNameAttr(), op.getNameKindAttr(), op.getAnnotationsAttr(),
      op.getInnerSymAttr());

  toDelete.insert(op);
  valueMap.insert({op.getResult(), newOp.getResult()});
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(WireOp op) {
  llvm::errs() << "WireOp\n";
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
  toDelete.insert(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

template <typename OpTy>
void LiftBundlesVisitor::handleConnect(OpTy op) {
  llvm::errs() << "handle connect\n";
  auto [lhs, lhsExploded] = fixOperand(op.getDest());

  // If the LHS did not explode, then we can construct the RHS as a read-only
  // operand.
  if (!lhsExploded) {
    auto rhs = fixROperand(op.getSrc());
    OpBuilder(op).create<OpTy>(op.getLoc(), lhs[0], rhs);
    toDelete.insert(op);
    return;
  }

  // We must force the RHS to explode.
  auto [rhs, rhsExploded] = fixOperand(op.getSrc());
  if (!rhsExploded)
    rhs = explode(rhs[0]);

  assert(lhs.size() == rhs.size() &&
         "Something went wrong exploding the elements");

  // Emit connections between all leaf elements.
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto type = convertType(op.getDest().getType());
  const auto *lit = lhs.begin();
  const auto *rit = rhs.begin();
  std::function<void(Type)> explodeConnect = [&](Type type) {
    if (auto bundleType = type.dyn_cast<BundleType>()) {
      for (auto &e : bundleType) {
        if (e.isFlip)
          std::swap(lit, rit);
        explodeConnect(e.type);
      }
    } else if (auto vectorType = type.dyn_cast<FVectorType>()) {
      explodeConnect(vectorType.getElementType());
    } else {
      builder.create<OpTy>(*lit++, *rit++);
    }
  };
  explodeConnect(type);

  toDelete.insert(op);
}

LogicalResult LiftBundlesVisitor::visitStmt(ConnectOp op) {
  llvm::errs() << "ConnectOp\n";
  handleConnect(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitStmt(StrictConnectOp op) {
  llvm::errs() << "StrictConnectOp\n";
  handleConnect(op);
  return success();
}

// LogicalResult LiftBundlesVisitor::visitStmt(WhenOp op) {
//   auto oldCondition = op.getCondition();
//   auto newCondition = fixROperand(op.)

//     for (auto &op : *body) {
//     auto result = dispatchVisitor(&op);
//     if (result.failed())
//       return result;
//     }
// }

//===----------------------------------------------------------------------===//
// Constant Conversion
//===----------------------------------------------------------------------===//

Attribute
LiftBundlesVisitor::convertBundleInVectorConstant(BundleType type,
                                                  ArrayRef<Attribute> fields) {

  // llvm::errs() << "convertBundleInVectorConstant\n";
  // llvm::errs() << "type=";
  // type.dump();
  // llvm::errs() << "\n";
  // llvm::errs() << "fields={";
  // for (auto f : fields) {
  //   llvm::errs() << " ";
  //   f.dump();
  // }
  // llvm::errs() << " ]\n";

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
    valueMap.insert({oldValue, oldValue});
    return success();
  }

  auto fields = convertConstant(oldType, op.getFields()).cast<ArrayAttr>();

  OpBuilder builder(op);
  auto newOp =
      builder.create<AggregateConstantOp>(op.getLoc(), newType, fields);

  valueMap[op.getResult()] = newOp.getResult();
  toDelete.insert(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Aggregate Create Ops
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitExpr(BundleCreateOp op) {
  // Tentatively mark as deleted. BundleCreateOps are converted or preserved
  // on demand.
  toDelete.insert(op);
  return success();
}

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
  llvm::errs() << "VectorCreateOp\n";

  auto oldType = op.getType();
  auto newType = convertType(oldType);

  if (oldType == newType) {
    auto changed = false;
    SmallVector<Value> newFields;
    for (auto oldField : op.getFields()) {
      auto newField = fixROperand(oldField);
      llvm::errs() << "new field : " << newField << "\n";
      if (oldField != newField)
        changed = true;
      newFields.push_back(newField);
    }

    if (!changed) {
      auto result = op.getResult();
      valueMap[result] = result;
      return success();
    }

    OpBuilder builder(op);
    auto newOp =
        builder.create<VectorCreateOp>(op.getLoc(), newType, newFields);
    valueMap[op.getResult()] = newOp.getResult();
    toDelete.insert(op);
    return success();
  }

  // OK, We are in for some pain!

  SmallVector<Value> convertedOldFields;
  for (auto oldField : op.getFields()) {
    auto convertedField = fixROperand(oldField);
    llvm::errs() << "new field : " << convertedField << "\n";
    convertedOldFields.push_back(convertedField);
  }

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto value = sinkVecDimIntoOperands(
      builder, convertType(oldType.getElementType()), convertedOldFields);
  valueMap[op.getResult()] = value;
  toDelete.insert(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pathing Ops
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitExpr(SubindexOp op) {
  llvm::errs() << "SubindexOp\n";
  // auto rootValue = getFieldRefFromValue(op).getValue();
  // if (valueMap.count(rootValue))
  toDelete.insert(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(SubfieldOp op) {
  llvm::errs() << "SubfieldOp\n";
  // auto rootValue = getFieldRefFromValue(op).getValue();
  // if (valueMap.count(rootValue))
  toDelete.insert(op);
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
      if (newType == oldType) {
        // valueMap.insert({port, port});
        continue;
      }

      auto newPort = port;
      newPort.type = newType;

      portsToErase[count + index] = true;
      newPorts.push_back({index + 1, newPort});
      ++count;
    }
    op.insertPorts(newPorts);
  }

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

  auto result = body->walk([&](Operation *op) {
    if (failed(dispatchVisitor(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  for (auto *op : toDelete) {
    op->dropAllUses();
    op->erase();
  }
  op.erasePorts(portsToErase);

  if (result.wasInterrupted())
    return failure();
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

  // auto visitor = LiftBundlesVisitor(&getContext());
  LiftBundlesVisitor visitor(&getContext());
  auto result = visitor.visit(getOperation());

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
