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
// #include "llvm/ADT/MapVector.h"
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

  LogicalResult visitUnhandledOp(Operation *) { return failure(); }
  LogicalResult visitInvalidOp(Operation *) { return failure(); }

  LogicalResult visitDecl(InstanceOp);
  LogicalResult visitDecl(WireOp);
  LogicalResult visitStmt(ConnectOp);
  LogicalResult visitStmt(StrictConnectOp);
  LogicalResult visitExpr(ConstantOp);
  LogicalResult visitExpr(AggregateConstantOp);
  LogicalResult visitExpr(BundleCreateOp);
  LogicalResult visitExpr(VectorCreateOp);
  LogicalResult visitExpr(SubindexOp);
  LogicalResult visitExpr(SubfieldOp);

  LogicalResult visitOperand(Value);
  LogicalResult visitOperand(OpResult);
  LogicalResult visitOperand(BlockArgument);

private:
  Type convertType(Type);
  FIRRTLBaseType convertType(FIRRTLBaseType);
  FIRRTLBaseType convertType(FIRRTLBaseType, SmallVector<unsigned>);

  Attribute convertAggregate(Type type, ArrayRef<Value> values);
  Attribute convertVectorOfBundle();

  Attribute convertConstant(Type type, Attribute fields);
  Attribute convertVectorConstant(FVectorType type, ArrayAttr fields);
  Attribute convertBundleConstant(BundleType type, ArrayAttr fields);
  Attribute convertBundleInVectorConstant(BundleType elementType,
                                          ArrayRef<Attribute> elements);

  std::pair<SmallVector<Value>, bool> buildPath(Value oldValue, Value newValue,
                                                unsigned fieldID);

  std::pair<SmallVector<Value>, bool> fixFieldRef(FieldRef fieldRef);

  std::pair<SmallVector<Value>, bool> fixOperand(Value value);

  SmallVector<Value> explode(Value value);
  void explode(OpBuilder builder, Value value, SmallVector<Value> &output);

  MLIRContext *context;
  SmallVector<Operation *> toDelete;

  /// A mapping from old values to their fixed up values.
  /// If a value is unchanged, it will be mapped to itself.
  /// If a value is present in the map, and does not map to itself, then
  /// it must be deleted.
  DenseMap<Value, Value> valueMap;
  // DenseMap<FieldRef, SmallVector<Value, 1>> fields;
  // DenseMap<FieldRef, SmallVector<FieldRef>> mappings;
};
} // end anonymous namespace

LiftBundlesVisitor::LiftBundlesVisitor(MLIRContext *context)
    : context(context) {}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type LiftBundlesVisitor::convertType(Type type) {
  if (auto firrtlType = type.dyn_cast<FIRRTLBaseType>()) {
    return convertType(firrtlType);
  }
  return type;
}

FIRRTLBaseType LiftBundlesVisitor::convertType(FIRRTLBaseType type) {
  SmallVector<unsigned> dimensions;
  return convertType(type, dimensions);
}

FIRRTLBaseType
LiftBundlesVisitor::convertType(FIRRTLBaseType type,
                                SmallVector<unsigned> dimensions) {
  // Vector Types
  if (auto vectorType = type.dyn_cast<FVectorType>(); vectorType) {
    dimensions.push_back(vectorType.getNumElements());
    return convertType(vectorType.getElementType(), dimensions);
    dimensions.pop_back();
  }

  // Bundle Types
  if (auto bundleType = type.dyn_cast<BundleType>(); bundleType) {
    SmallVector<BundleType::BundleElement> elements;
    for (auto element : bundleType.getElements()) {
      elements.push_back(BundleType::BundleElement(
          element.name, element.isFlip, convertType(element.type, dimensions)));
    }

    return BundleType::get(elements, context);
  }

  // Ground Types
  for (auto size : llvm::reverse(dimensions)) {
    type = FVectorType::get(type, size);
  }
  return type;
}

//===----------------------------------------------------------------------===//
// Path Rewriting
//===----------------------------------------------------------------------===//

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::buildPath(Value oldValue, Value newValue,
                              unsigned fieldID) {

  auto loc = oldValue.getLoc();
  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(newValue);

  SmallVector<unsigned> subfieldIndices;
  SmallVector<unsigned> subindexIndices;
  bool inVector = false;
  bool inVectorOfBundle = false;

  auto oldType = oldValue.getType();
  auto newType = newValue.getType();

  // llvm::errs() << getFieldName(FieldRef(oldValue, fieldID)) << "\n";

  auto type = oldType;
  while (fieldID != 0) {
    if (auto bundle = type.dyn_cast<BundleType>()) {
      auto index = bundle.getIndexForFieldID(fieldID);
      fieldID -= bundle.getFieldID(index);
      subfieldIndices.push_back(index);
      inVectorOfBundle = inVector;
      type = bundle.getElementType(index);
    } else {
      auto vector = type.cast<FVectorType>();
      auto index = vector.getIndexForFieldID(fieldID);
      fieldID -= vector.getFieldID(index);
      subindexIndices.push_back(index);
      inVector = true;
      type = vector.getElementType();
    }
  }

  // llvm::errs() << "printing subfield operators\n";
  // for (auto index : subfieldIndices) {
  //   llvm::errs() << index << "\n";
  // }

  // llvm::errs() << "printing subindex operators\n";
  // for (auto index : subindexIndices) {
  //   llvm::errs() << index << "\n";
  // }

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

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::fixFieldRef(FieldRef fieldRef) {
  auto oldValue = fieldRef.getValue();
  auto newValue = valueMap.lookup(oldValue);
  if (!newValue)
    return {{oldValue}, false};

  return buildPath(oldValue, newValue, fieldRef.getFieldID());
}

//===----------------------------------------------------------------------===//
// Operand Fixup
//===----------------------------------------------------------------------===//

std::pair<SmallVector<Value>, bool>
LiftBundlesVisitor::fixOperand(Value value) {
  return fixFieldRef(getFieldRefFromValue(value));
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

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitDecl(InstanceOp op) {
  llvm::errs() << "InstanceOp\n";

  // for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
  //   auto result = op.getResult(i);
  //   auto type = TypeConverterX::convert(context, result.getType());
  //   result.setType(type);
  // }
  return success();
}

LogicalResult LiftBundlesVisitor::visitDecl(WireOp op) {
  // TODO: Rewrite the wire's type.
  llvm::errs() << "WireOp\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitStmt(ConnectOp op) {
  llvm::errs() << "ConnectOp\n";
  auto [lhs, lhsExploded] = fixOperand(op.getDest());
  auto [rhs, rhsExploded] = fixOperand(op.getSrc());

  if (rhsExploded && !lhsExploded)
    lhs = explode(lhs[0]);

  if (lhsExploded && !rhsExploded)
    rhs = explode(rhs[0]);

  if (lhs.size() != rhs.size())
    assert(false && "Something went wrong exploding the elements");

  auto builder = OpBuilder(context);
  auto loc = op.getLoc();
  builder.setInsertionPoint(op);

  for (size_t i = 0, e = lhs.size(); i < e; ++i) {
    builder.create<ConnectOp>(loc, lhs[i], rhs[i]);
  }

  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitStmt(StrictConnectOp op) {
  llvm::errs() << "StrictConnectOp\n";
  auto [lhs, lhsExploded] = fixOperand(op.getDest());
  auto [rhs, rhsExploded] = fixOperand(op.getSrc());

  if (rhsExploded && !lhsExploded)
    lhs = explode(lhs[0]);

  if (lhsExploded && !rhsExploded)
    rhs = explode(rhs[0]);

  if (lhs.size() != rhs.size())
    assert(false && "Something went wrong exploding the elements");

  auto builder = OpBuilder(context);
  auto loc = op.getLoc();
  builder.setInsertionPoint(op);

  for (size_t i = 0, e = lhs.size(); i < e; ++i) {
    builder.create<StrictConnectOp>(loc, lhs[i], rhs[i]);
  }

  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Conversion
//===----------------------------------------------------------------------===//

Attribute LiftBundlesVisitor::convertBundleInVectorConstant(
    BundleType type, ArrayRef<Attribute> fields) {
  
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
      return convertBundleInVectorConstant(bundleElementType, oldElements.getValue());

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

LogicalResult LiftBundlesVisitor::visitExpr(ConstantOp op) {
  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(AggregateConstantOp op) {
  auto oldType = op.getType();
  auto newType = convertType(oldType);

  if (oldType == newType)
    return success();

  auto fields = convertConstant(oldType, op.getFields()).cast<ArrayAttr>();

  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(op);
  auto newOp = builder.create<AggregateConstantOp>(op.getLoc(), newType, fields);
  valueMap[op.getResult()] = newOp.getResult();
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Aggregate Create Ops
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitExpr(BundleCreateOp op) {
  auto result = op.getResult();
  auto oldType = result.getType();
  auto newType = convertType(oldType);
  if (oldType != newType)
    result.setType(newType);
  
  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(VectorCreateOp op) {
  auto oldResult = op.getResult();
  auto oldType = oldResult.getType();
  auto newType = convertType(oldType);

  if (oldType == newType) {
    if (auto bundleElementType = oldType.getElementType().dyn_cast<BundleType>()) {
      convertVectorOfBundle()
    }
  }
    return success();

  auto bundleType = oldType.getElementType().cast<BundleType>();

  OpBuilder builder(context);
  builder.setInsertionPointAfter(op);
  builder.create<BundleCreateOp>(
  auto valueMap[oldResult] = newOp.getResult();
  return success();
}

//===----------------------------------------------------------------------===//
// Pathing Ops
//===----------------------------------------------------------------------===//

LogicalResult LiftBundlesVisitor::visitExpr(SubindexOp op) {
  llvm::errs() << "SubindexOp\n";
  auto rootValue = getFieldRefFromValue(op).getValue();
  if (valueMap.count(rootValue))
    toDelete.push_back(op);

  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(SubfieldOp op) {
  llvm::errs() << "SubfieldOp\n";
  auto rootValue = getFieldRefFromValue(op).getValue();
  if (valueMap.count(rootValue))
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

  auto body = op.getBodyBlock();
  for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
    if (portsToErase[i]) {
      auto oldArg = body->getArgument(i);
      auto newArg = body->getArgument(i + 1);
      valueMap[oldArg] = newArg;
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
    auto result = dispatchVisitor(&op);
    if (result.failed())
      return result;
  }

  for (auto op : toDelete) {
    op->dropAllUses();
    op->erase();
  }
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
