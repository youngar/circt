//===- AddSeqMemPorts.cpp - Add extra ports to memories ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the AddSeqMemPorts pass.  This pass will add extra ports
// to memory modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Parallel.h"

using namespace circt;
using namespace firrtl;

namespace {
struct AddSeqMemPortsPass : public AddSeqMemPortsBase<AddSeqMemPortsPass> {
  void runOnOperation() override;
  LogicalResult processAddPortAnno(Location loc, Annotation anno);
  LogicalResult processFileAnno(Location loc, StringRef metadataDir,
                                Annotation anno);
  LogicalResult processAnnos(CircuitOp circuit);

  hw::OutputFileAttr portFile;
};
} // end anonymous namespace

// void AddSeqMemPortsPass::runOnTestharness(FModuleOp module) {
//   if (module == dut) {
//     // Memories are lowered differently once we are in the DUT.
//     if (AnnotationSet(module).hasAnnotation(
//             "sifive.enterprise.firrtl.MarkDUTAnnotation")) {
//     }
//   }
// }

// module.getBody()->walk([&](Operation *op) {
//   if (auto instance = dyn_cast<InstanceOp>(op)) {
//     auto child = instanceGraph->getReferencedModule(instance);
//     if (child != dut) {
//       return runOnTestharness(module);
//     }

//     // If the instance is the DUT, we will have to handle
//     SmallVector<Value> newPorts;

//   } else if (auto mem = dyn_cast<MemOp>(op)) {
//     // Memories in the testharness are lowered without duplication.
//     SmallVector<Value> newPorts;
//     lowerMemory(mem, /*isTestHarness=*/true, newPorts);
//     assert(newPorts.size() == 0 && "should not have added ports");
//   }
// });

// // This is a list of ports we will need to fish upward through this module.
// SmallVector<PortInfo> newPorts;
// }

LogicalResult AddSeqMemPortsPass::processAddPortAnno(Location loc,
                                                     Annotation anno) {
  auto name = anno.getMember<StringAttr>("name");
  if (!name)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'name' of string type");

  auto input = anno.getMember<BoolAttr>("input");
  if (!input)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'input' of boolean type");
  auto direction = input.getValue() ? Direction::In : Direction::Out;

  auto width = anno.getMember<IntegerAttr>("width");
  if (!width)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'width' of integer type");
  auto type = UIntType::get(&getContext(), width.getInt());

  // userPorts.emplace_back(name, type, direction);
  return success();
}

LogicalResult AddSeqMemPortsPass::processFileAnno(Location loc,
                                                  StringRef metadataDir,
                                                  Annotation anno) {
  if (portFile)
    return emitError(
        loc, "circuit has two AddSeqMemPortsFileAnnotation annotations");

  auto filename = anno.getMember<StringAttr>("filename");
  if (!filename)
    return emitError(loc,
                     "AddSeqMemPortsFileAnnotation requires field 'filename' "
                     "of string type");

  portFile = hw::OutputFileAttr::getFromDirectoryAndFilename(
      &getContext(), metadataDir, filename.getValue(),
      /*excludeFromFilelist=*/true);
  return success();
}

LogicalResult AddSeqMemPortsPass::processAnnos(CircuitOp circuit) {
  auto loc = circuit.getLoc();

  // Find the metadata directory.
  auto dirAnno = AnnotationSet(circuit).getAnnotation(
      "sifive.enterprise.firrtl.MetadataDirAnnotation");
  StringRef metadataDir = "metadata";
  if (dirAnno) {
    auto dir = dirAnno.getMember<StringAttr>("dirname");
    if (!dir)
      return emitError(loc, "MetadataDirAnnotation requires field 'dirname' of "
                            "string type");
    metadataDir = dir.getValue();
  }

  // Remove the annotations we care about.
  bool error = false;
  AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
    if (error)
      return false;
    if (anno.isClass("sifive.enterprise.firrtl.AddSeqMemPortAnnotation")) {
      error = failed(processAddPortAnno(loc, anno));
      return true;
    }
    if (anno.isClass("AddSeqMemPortsFileAnnotation")) {
      error = failed(processFileAnno(loc, metadataDir, anno));
      return true;
    }
    return false;
  });
  return failure(error);
}

// InstanceGraphNode *AddSeqMemPortsPass::findDUT() {
//   // Find the DUT module.
//   for (auto &op : *getOperation().getBody()) {
//     if (auto module = dyn_cast<FModuleLike>(op))
//       if (AnnotationSet::removeAnnotations(module, dutAnnoClass)) {
//         dut = module;
//         break;
//       }
//   }
//   if (!dut)
//     dut = instanceGraph->getTopLevelModule();
// }

// void addMemoryPorts(FMemModule mem) { memCount[mem] = 1; }

void AddSeqMemPortsPass::runOnOperation() {
  auto circuit = getOperation();

  // Reset the global port number.
  unsigned userPortNumber = 0;

  if (failed(processAnnos(circuit)))
    return signalPassFailure();

  // We do an up-front search for the DUT.  This is to handle the case
  // where there is no module explicitly marked as
  // auto *dutNode = findDUT();

  // Add ports to all memories under the DUT.
  // DenseMap<Operation *> memCount;
  // for (auto *node : llvm::post_order(dutNode)) {
  //   auto op = *node->getModule();
  //   if (auto mem = dyn_cast<FMemModuleOp>(op)) {

  //   } else if (auto module = dyn_cast<FModuleOp>(op)) {
  //   }
  // }
  //

  // instanceGraph = &getAnalysis<InstanceGraph>(getOperation());

  // runOnModule(instanceGraph->getTopLevelModule());
  // memories.clear();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAddSeqMemPortsPass() {
  return std::make_unique<AddSeqMemPortsPass>();
}
