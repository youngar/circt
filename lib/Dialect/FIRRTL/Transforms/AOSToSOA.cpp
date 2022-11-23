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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace firrtl;

namespace {

struct TypeAccumulation {

};

bool convertFVectorType(FVectorType vectorType) {
  auto elementType = vectorType.getElementType();
  if (auto bundleElementType = dyn_cast<BundleType>(elementType)) {
    return false; // convertBundleType(bundleElementType);
  }
  return false;
}

bool convertType(Type type) {
  if (auto vectorType = dyn_cast<FVectorType>(type))
    return convertFVectorType(vectorType);
  return false;
}

// bool convertPort(port) {

// }

bool convertFModulePorts(FModuleOp moduleOp) {
  return false;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class AOSToSOAPass : public AOSToSOABase<AOSToSOAPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void AOSToSOAPass::runOnOperation() {
  markAllAnalysesPreserved();
  // ModuleVisitor visitor;
  // if (!visitor.run(getOperation()))
  //   markAllAnalysesPreserved();
  // if (failed(visitor.checkInitialization()))
  //   signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAOSToSOAPass() {
  return std::make_unique<AOSToSOAPass>();
}
