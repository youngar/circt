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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "firrtl-aos-to-soa"

using namespace circt;
using namespace firrtl;

// namespace {

// struct TypeAccumulation {};

// bool convertFVectorType(FVectorType vectorType) {
//   auto elementType = vectorType.getElementType();
//   if (auto bundleElementType = dyn_cast<BundleType>(elementType)) {
//     return false; // convertBundleType(bundleElementType);
//   }
//   return false;
// }

// bool convertType(Type type) {
//   if (auto vectorType = dyn_cast<FVectorType>(type))
//     return convertFVectorType(vectorType);
//   return false;
// }

// bool convertPort(port) {

// }

// bool convertPort() {}

// bool convertUsers(Value value) {
//   for (auto op : value.use_begin()) {
//   }
// }

//===----------------------------------------------------------------------===//
// Type Rewriter
//===----------------------------------------------------------------------===//

namespace {
class TypeRewriteTable {
public:
  TypeRewriteTable();

  Type rewrite(Type);

private:
  // True if the given type is a vector of struct.
  bool needsRewrite(Type) const;

  Type buildRewrite(Type) const;
};
}

TypeRewriteTable::TypeRewriteTable() {}

Type TypeRewriteTable::rewrite(Type t) {
  // if the type doesn't require any rewrite, just return it.
  if (!needsRewrite(t)) {
    return nullptr;
  }

  return t;
}

bool TypeRewriteTable::needsRewrite(Type t) const {
  // auto vt = t.dyn_cast<VectorType>();
  // if (vt == nullptr) {
  //   return false;
  // }

  // auto et = vt.getElementType();
  // while (et.isa<VectorType>()) {
  //   if (et.isa<BundleType>())
  //     return true;
  //   if (!et.isa<VectorType>())
  //     return false;

  //   auto vt = et.dyn_cast<VectorType>();
  //   et = vt.getElementType();
  // }

  return false;
}

namespace {
class TypeConverterX {
public:
  static FIRRTLBaseType convert(MLIRContext *context, FIRRTLBaseType type);
  static TypeAttr convert(MLIRContext *context, TypeAttr attr);
  static Type convert(MLIRContext *context, Type type);

private:
  class State {
  public:
    std::vector<unsigned> dimensions;
  };

  class Stash {
  public:
    Stash(State &state) : state(state), dimensionality(state.dimensions.size()) {}

    ~Stash() {
      state.dimensions.resize(dimensionality);
    }

  private:
    State &state;
    unsigned dimensionality;
  };

  static FIRRTLBaseType convert(MLIRContext *context, State &state, FIRRTLBaseType type);
};
}

TypeAttr TypeConverterX::convert(MLIRContext *context, TypeAttr attr) {
  return TypeAttr::get(convert(context, attr.getValue()));
}

Type TypeConverterX::convert(MLIRContext *context, Type type) {
  if (auto ft = type.dyn_cast<FIRRTLBaseType>()) {
    return convert(context, ft).dyn_cast<Type>();
  }
  return type;
}

FIRRTLBaseType TypeConverterX::convert(MLIRContext *context, FIRRTLBaseType type) {
  State state;
  return convert(context, state, type);
}

FIRRTLBaseType TypeConverterX::convert(MLIRContext *context, State &state, FIRRTLBaseType type) {

  // Vector Types

  if (auto vectorType = type.dyn_cast<FVectorType>(); vectorType) {
    state.dimensions.push_back(vectorType.getNumElements());
    return convert(context, state, vectorType.getElementType());
  }

  // Bundle Types

  if (auto bundleType = type.dyn_cast<BundleType>(); bundleType) {
    SmallVector<BundleType::BundleElement> elements;
    for (auto element : bundleType.getElements()) {
      Stash guard(state);
      elements.push_back(BundleType::BundleElement(element.name, element.isFlip, convert(context, state, element.type)));
    }

    return BundleType::get(elements, context);
  }

  // Ground Types

  for (auto size : llvm::reverse(state.dimensions)) {
    type = FVectorType::get(type, size);
  }
  return type;
}

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

namespace {
class LiftBundlesVisitor
    : public FIRRTLVisitor<LiftBundlesVisitor, LogicalResult> {
public:
  explicit LiftBundlesVisitor(MLIRContext *context);

  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<LiftBundlesVisitor, LogicalResult>::visitStmt;

  LogicalResult visit(FModuleOp op);

private:
  Type convertType(Type type);

  Attribute convertTypeAttr(Attribute attr);

  DenseMap<FieldRef, SmallVector<FieldRef>> mappings;
  MLIRContext *context;
  TypeRewriteTable typeRewriteTable;
};
} // end anonymous namespace

LiftBundlesVisitor::LiftBundlesVisitor(MLIRContext *context)
    : mappings(), context(context), typeRewriteTable() {}

// LogicalResult LiftBundlesVisitor::visit(SubfieldOp op) { return success(); }

// LogicalResult LiftBundlesVisitor::visit(SubindexOp op) { return success(); }

// LogicalResult LiftBundlesVisitor::visitGeneric(Operation *op) {
//   // for (auto operand & op->getOperands()) {
//   //   convertO
//   // }
//   return success();
// }

// LogicalResult LiftBundlesVisitor::visit(SubaccessOp op) {
//   return success();
// }

// LogicalResult LiftBundlesVisitor::visitConnect(ConnectOp op) {
//   // auto helper = [*](SmallVector outerTypes) {
//   //   auto aType = a.getType();
//   //   auto bType = b.getType();
//   //   if (a.type == vector) {
//   //     outerTypes.push_back
//   //   }
//   // }

//   return success();
// }

LogicalResult LiftBundlesVisitor::visit(FModuleOp op) {
  auto numPorts = op.getNumPorts();

  // SmallVector<Direction> newDirections;
  // SmallVector<Attribute> newNames, newTypes, newAnnos, newSyms;
  SmallVector<Attribute> newPortTypes;
  newPortTypes.resize(numPorts);

  // newDirections.reserve(numPorts);
  // newNames.reserve(numPorts);
  // newTypes.reserve(numPorts);
  // newAnnos.reserve(numPorts);
  // newSyms.reserve(numPorts);

  // SmallVector<TypeAttr> newPortTypeAttrs;
  // newPortTypeAttrs.resize(e);

  for (size_t i = 0; i < numPorts; ++i) {
    newPortTypes[i] = TypeConverterX::convert(context, op.getPortTypeAttr(i)).cast<Attribute>();
  }

  // op->setAttr("portDirections",
  //             direction::packAttribute(op.getContext(), newDirections));
  // op->setAttr("portNames", ArrayAttr::get(op.getContext(), newNames));
  op->setAttr("portTypes", ArrayAttr::get(op.getContext(), newPortTypes));
  // op->setAttr("portAnnotations", ArrayAttr::get(op.getContext(), newAnnos));
  // op.setPortSymbols(newSyms);

  auto body = op.getBodyBlock();
  for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
    auto argument = body->getArgument(i);
    argument.setType(newPortTypes[i].dyn_cast<TypeAttr>().getValue());
  }
  return success();
}

Type LiftBundlesVisitor::convertType(Type type) {
  return typeRewriteTable.rewrite(type);
}


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
  visitor.visit(getOperation());
  markAllAnalysesPreserved();
  // auto changed = convertFModuleOp(getOperation());
  auto changed = false;
  if (!changed) {
    markAllAnalysesPreserved();
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAOSToSOAPass() {
  return std::make_unique<AOSToSOAPass>();
}
