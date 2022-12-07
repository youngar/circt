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
// #include "llvm/ADT/"
#define DEBUG_TYPE "firrtl-aos-to-soa"

using namespace circt;
using namespace firrtl;

// namespace {
// class TypeConverterX {
// public:
//   static FIRRTLBaseType convert(MLIRContext *context, FIRRTLBaseType type);
//   static TypeAttr convert(MLIRContext *context, TypeAttr attr);
//   static Type convert(MLIRContext *context, Type type);

// private:
//   class State {
//   public:
//     std::vector<unsigned> dimensions;
//   };

//   class Stash {
//   public:
//     Stash(State &state)
//         : state(state), dimensionality(state.dimensions.size()) {}

//     ~Stash() { state.dimensions.resize(dimensionality); }

//   private:
//     State &state;
//     unsigned dimensionality;
//   };

//   static FIRRTLBaseType convert(MLIRContext *context, State &state,
//                                 FIRRTLBaseType type);
// };
// } // namespace

// TypeAttr TypeConverterX::convert(MLIRContext *context, TypeAttr attr) {
//   return TypeAttr::get(convert(context, attr.getValue()));
// }

// FIRRTLBaseType TypeConverterX::convert(MLIRContext *context,
//                                        FIRRTLBaseType type) {
//   State state;
//   return convert(context, state, type);
// }

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
  LogicalResult visitExpr(ConstantOp);
  LogicalResult visitExpr(SubindexOp);
  LogicalResult visitExpr(SubfieldOp);

  LogicalResult visitOperand(Value);
  LogicalResult visitOperand(OpResult);
  LogicalResult visitOperand(BlockArgument);

  Type convertType(Type);
  FIRRTLBaseType convertType(FIRRTLBaseType);
  FIRRTLBaseType convertType(FIRRTLBaseType, SmallVector<unsigned>);

  SmallVector<Value> buildPath(Value oldValue, Value newValue,
                               unsigned fieldID);

  SmallVector<Value> fixFieldRef(FieldRef fieldRef);

  SmallVector<Value> fixOperand(Value value);

private:
  // Type conver tType(Type);
  // Attribute convertTypeAttr(Attribute);

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
  llvm::errs() << "WireOp\n";
  return success();
}

SmallVector<Value> LiftBundlesVisitor::buildPath(Value oldValue, Value newValue,
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

  auto type = oldType;
  while (fieldID != 0) {
    if (auto bundle = type.dyn_cast<BundleType>()) {
      auto index = bundle.getIndexForFieldID(fieldID);
      fieldID -= bundle.getFieldID(index);
      subfieldIndices.push_back(index);
      inVectorOfBundle = inVector;
    } else {
      auto vector = type.cast<FVectorType>();
      auto index = vector.getIndexForFieldID(fieldID);
      fieldID -= vector.getFieldID(index);
      subindexIndices.push_back(index);
      inVector = true;
    }
  }

  auto value = newValue;
  for (auto index : subfieldIndices) {
    auto op = builder.create<SubfieldOp>(loc, value, index);
    value = op.getResult();
  }

  SmallVector<Value> values;
  std::function<void(Value)> explode = [&](Value value) {
    if (auto bundleType = value.getType().dyn_cast<BundleType>()) {
      auto values = SmallVector<Value>();
      for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
        auto fieldValue = builder.create<SubfieldOp>(loc, value, i);
        explode(fieldValue);
      }
    } else {
      for (auto index : subindexIndices) {
        auto op = builder.create<SubindexOp>(loc, value, index);
        value = op.getResult();
      }
      values.push_back(value);
    }
  };

  if (subindexIndices.size() != 0)
    explode(value);
  else
    values.push_back(value);

  return values;
}

SmallVector<Value> LiftBundlesVisitor::fixFieldRef(FieldRef fieldRef) {
  auto oldValue = fieldRef.getValue();
  auto newValue = valueMap.lookup(oldValue);
  if (!newValue)
    return {oldValue};

  return buildPath(oldValue, newValue, fieldRef.getFieldID());
}

SmallVector<Value> LiftBundlesVisitor::fixOperand(Value value) {
  return fixFieldRef(getFieldRefFromValue(value));
}

LogicalResult LiftBundlesVisitor::visitStmt(ConnectOp op) {
  llvm::errs() << "ConnectOp\n";
  auto operands = op.getOperands();
  // auto remapped = false;
  // for (auto operand : operands) {
  //   if (map.count(operand)) {
  //     remapped = true;
  //     break;
  //   }
  // }

  auto dst = getFieldRefFromValue(op.getDest());
  auto dstType = dst.getValue().getType();
  auto src = getFieldRefFromValue(op.getSrc());
  auto srcType = dst.getValue().getType();

  auto count = 0;
  for (auto operand : op.getOperands()) {
    auto values = fixOperand(operand);
    if (values.size() != 1)
      assert(false && "cannot handle exploded operands");
    
    auto value = values[0];
    op.setOperand(count, value);
    count++;
  }

  toDelete.push_back(op);
  return success();
}

LogicalResult LiftBundlesVisitor::visitExpr(ConstantOp op) {
  llvm::errs() << "ConstantOp\n";
  return success();
}

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
// Operand Updating
//===----------------------------------------------------------------------===//

// LogicalResult LiftBundlesVisitor::visitOperand(Value operand) {
//   if (auto result = operand.dyn_cast<OpResult>()) {
//     return visitOperand(result);
//   }

//   if (auto argument = operand.dyn_cast<BlockArgument>()) {
//     return visitOperand(argument);
//   }

//   return failure();
// }

// namespace {
// class AccessPathFixup {
// public:
//   static SmallVector<Value> fixup(Value);

// private:
//   struct State {
//     std::vector<unsigned> indices;
//     std::vector<unsigned> fields;
//   };

//   static SmallVector<Value> fixup(Value, State &);

// };
// }

// // if the root value was mutated, then the path will
// // have to be updated.
// SmallVector<Value> AccessPathFixup::fixup(value) {
//   // we need to know if this specific operation
//   // indexed into a vector of bundles
//   // if yes, we need to convert the
//   auto op = value.getDefiningOp();
//   //while
//   if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
//   }
// }

// // SmallVector<Value> AccessPathFixup::fixup(Value value, State state) {
// // }

// LogicalResult fixOperand(OpResult operand) {
//   auto value = operand.getValue();
//   auto op = operand.getValue().getDefiningOp();
//   if (auto indexOp = op.dyn_cast<SubindexOp>()) {
//     auto result = visitOperand(subindexOp);
//   }
// }

// LogicalResult fixOperand(Value value) {
//   return fixOperand(value.getDefiningOp());
// }

// struct Index {
//   static Index asConst(unsigned index)
//     : kind(Kind::Const), asConstIndex(index) {}

//   enum class Kind { Const, Value, };

//   Kind kind;
//   union {
//     Value      asValueIndex;
//     unsigned   asConstIndex;
//   };
// };

// // returns a value to
// std::pair<Value, bool> fixOperand(Operation *operand) {
//   // if the given operation was remapped,
//   // we have to build a path to the new value.
//   // When
//   auto inVector = false;
//   auto inVectorOfBundle = false;

//   SmallVector<Index> vectorAccesses;
//   SmallVector<unsigned> bundleAccesses;

//   auto it = operand;
//   while (true) {
//     if (auto op = dyn_cast<SubindexOp>(it)) {
//       inVector = true;
//       vectorAccesses.push_back({ Index::Kind::Subindex, op.getIndex());
//       it = op.getInput().getDefiningOp();
//       continue;
//     }

//     if (auto op = dyn_cast<SubaccessOp>(it)) {
//       auto inVector = true;
//       continue;
//     }

//     if (auto op = dyn_cast<SubfieldOp>(it)) {
//       inVectorOfBundle = inVector;
//       it = op.getInput().getDefiningOp();
//       continue;
//     }

//     // Identified the root object we are accessing.
//     if (map.count(it)) {
//       build_path()
//     }
//     break;
//   }

//   if (!inVectorOfBundle)
//     return { value, false };

//   // if the original operand accessed a record through a vector, we will
//   have to correct the operand.

//   auto result = it.getResult();

//   auto value =
//   return { value, true };
// }

// LogicalResult visitOperand(BlockArgument operand) { return success(); }

// LogicalResult fixResult(Operation *op) {
//   if (auto subindexOp = op->dyn_cast<SubindexOp>()) {
//     return fixPath(subindexOp);
//   }

//   return failure();
// }

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
// Access-path Rewriting
//===----------------------------------------------------------------------===//

// SmallVector<Value, 1> LiftBundlesVisitor::fixPath() {
//   if ()
// }

// // Assume the given path is required, rebuild the path, returning 1 or
// more
// // values representing the fields
// SmallVector<Value, 1> LiftBundlesVisitor::fixPath(SubindexOp op) {
//   // Trace the access path to the end. If the defining value was not
//   remapped,
//   // then we don't need to do any work.

//   // if the element type is a vector, we are okay.
//   auto input = op.getInput();
//   auto type  = input.getType().cast<FVectorType>();
// }

//===----------------------------------------------------------------------===//
// Module Op Conversion
//===----------------------------------------------------------------------===//

// bool convertPort() { auto type = op.getType(); }

// bool convertPortsofFModuleOp(FModuleOp moduleOp) {
//   auto changed = false;
//   auto ports = op.getPorts();
//   auto i = size_t(0);
//   auto e = ports.size();
//   for (; i < e; ++i) {
//     changed = convertPort(ports, index) || changed;
//   }

//   return changed;
// }

// bool convertFModuleOp(FModuleOp op) {
//   bool changed = false;
//   changed = convertFModulePorts() || changed;
//   changed = convertFModuleBody() || changed;
//   return changed;
// }

// } // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class AOSToSOAPass : public AOSToSOABase<AOSToSOAPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void AOSToSOAPass::runOnOperation() {
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
