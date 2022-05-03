//===- LowerMemory.cpp - Lower Memories -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMemories pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Parallel.h"
#include <set>

using namespace circt;
using namespace firrtl;

static const char dutAnnoClass[] = "sifive.enterprise.firrtl.MarkDUTAnnotation";

namespace {
struct LowerMemoryPass : public LowerMemoryBase<LowerMemoryPass> {

  void emitMemoryModule(MemOp op, const FirMemory &summary);
  FirMemory getOrCreateMemModule(MemOp op, bool shouldDedup);
  void emitMemoryInstance(MemOp op, const FirMemory &summary);
  void lowerMemory(MemOp mem, bool shouldDedup);
  void runOnModule(FModuleOp module, bool shouldDedup);
  void runOnOperation() override;

  std::set<FirMemory> memories;
};
} // end anonymous namespace

void LowerMemoryPass::emitMemoryModule(MemOp op, const FirMemory &mem) {
  // Insert the memory module at the bottom of the circuit.
  auto *context = &getContext();
  auto b = OpBuilder::atBlockEnd(getOperation().getBody());

  SmallVector<PortInfo> ports;
  // We don't need a single bit mask, it can be combined with enable. Create
  // an unmasked memory if maskBits = 1.
  FIRRTLType u1Type = UIntType::get(context, 1);

  auto makePortCommon = [&](StringRef prefix, size_t idx, FIRRTLType addrType) {
    ports.push_back({b.getStringAttr(prefix + Twine(idx) + "_addr"), addrType,
                     Direction::In});
    ports.push_back(
        {b.getStringAttr(prefix + Twine(idx) + "_en"), u1Type, Direction::In});
    ports.push_back(
        {b.getStringAttr(prefix + Twine(idx) + "_clk"), u1Type, Direction::In});
  };

  FIRRTLType dataType = UIntType::get(context, mem.dataWidth);

  FIRRTLType maskType = UIntType::get(&getContext(), mem.maskBits);
  FIRRTLType addrType =
      UIntType::get(&getContext(), std::max(1U, llvm::Log2_64_Ceil(mem.depth)));

  for (size_t i = 0, e = mem.numReadPorts; i != e; ++i) {
    makePortCommon("R", i, addrType);
    ports.push_back(
        {b.getStringAttr("R" + Twine(i) + "_data"), dataType, Direction::Out});
  }
  for (size_t i = 0, e = mem.numReadWritePorts; i != e; ++i) {
    makePortCommon("RW", i, addrType);
    ports.push_back(
        {b.getStringAttr("RW" + Twine(i) + "_wmode"), u1Type, Direction::In});
    ports.push_back(
        {b.getStringAttr("RW" + Twine(i) + "_wdata"), dataType, Direction::In});
    ports.push_back({b.getStringAttr("RW" + Twine(i) + "_rdata"), dataType,
                     Direction::Out});
    // Ignore mask port, if maskBits =1
    if (mem.isMasked)
      ports.push_back({b.getStringAttr("RW" + Twine(i) + "_wmask"), maskType,
                       Direction::In});
  }

  for (size_t i = 0, e = mem.numWritePorts; i != e; ++i) {
    makePortCommon("W", i, addrType);
    ports.push_back(
        {b.getStringAttr("W" + Twine(i) + "_data"), dataType, Direction::In});
    // Ignore mask port, if maskBits =1
    if (mem.isMasked)
      ports.push_back(
          {b.getStringAttr("W" + Twine(i) + "_mask"), maskType, Direction::In});
  }

  // Mask granularity is the number of data bits that each mask bit can
  // guard. By default it is equal to the data bitwidth.
  auto maskGran = mem.isMasked ? mem.dataWidth / mem.maskBits : mem.dataWidth;

  // Make the global module for the memory
  auto moduleName = StringAttr::get(context, op.name() + "_ext");
  b.create<FMemModuleOp>(mem.loc, moduleName, ports, mem.numReadPorts,
                         mem.numWritePorts, mem.numReadWritePorts, dataType,
                         mem.maskBits, mem.readLatency, mem.writeLatency,
                         mem.depth);
}

FirMemory LowerMemoryPass::getOrCreateMemModule(MemOp op, bool shouldDedup) {
  // Try to find a matching memory blackbox that we already created.  If
  // shouldDedup is true, we will just generate a new memory module.
  auto summary = op.getSummary();
  if (shouldDedup) {
    auto it = memories.find(summary);
    if (it != memories.end())
      return *it;
  }
  // Create a new module for this memory.
  emitMemoryModule(op, summary);

  // Record the memory module.  We don't want to use this module for other
  // memories, then we don't add it to the table.
  if (shouldDedup)
    memories.insert(summary);

  return summary;
}

static SmallVector<SubfieldOp> getAllFieldAccesses(Value structValue,
                                                   StringRef field) {
  SmallVector<SubfieldOp> accesses;
  for (auto op : structValue.getUsers()) {
    assert(isa<SubfieldOp>(op));
    auto fieldAccess = cast<SubfieldOp>(op);
    auto elemIndex =
        fieldAccess.input().getType().cast<BundleType>().getElementIndex(field);
    if (elemIndex.hasValue() &&
        fieldAccess.fieldIndex() == elemIndex.getValue()) {
      accesses.push_back(fieldAccess);
    }
  }
  return accesses;
}

void LowerMemoryPass::emitMemoryInstance(MemOp op, const FirMemory &summary) {
  OpBuilder builder(op);
  auto *context = &getContext();
  auto memName = op.name();
  if (memName.empty())
    memName = "mem";

  // Process each port in turn.
  SmallVector<Type, 8> portTypes;
  SmallVector<Direction> portDirections;
  SmallVector<Attribute> portNames;
  DenseMap<Operation *, size_t> returnHolder;

  // The result values of the memory are not necessarily in the same order as
  // the memory module that we're lowering to.  We need to lower the read
  // ports before the read/write ports, before the write ports.
  for (unsigned memportKindIdx = 0; memportKindIdx != 3; ++memportKindIdx) {
    MemOp::PortKind memportKind = MemOp::PortKind::Read;
    auto portLabel = "R";
    switch (memportKindIdx) {
    default:
      assert(0 && "invalid idx");
      break; // Silence warning
    case 1:
      memportKind = MemOp::PortKind::ReadWrite;
      portLabel = "RW";
      break;
    case 2:
      memportKind = MemOp::PortKind::Write;
      portLabel = "W";
      break;
    }

    // This is set to the count of the kind of memport we're emitting, for
    // label names.
    unsigned portNumber = 0;

    // Get an unsigned type with the specified width.
    auto getType = [&](size_t width) { return UIntType::get(context, width); };
    // Get the size of an address type big enough to index a memory range.
    auto getAddressType = [&](size_t depth) {
      return getType(llvm::Log2_64_Ceil(depth));
    };

    // Memories return multiple structs, one for each port, which means we
    // have two layers of type to split apart.
    for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
      // Process all of one kind before the next.
      if (memportKind != op.getPortKind(i))
        continue;

      auto addPort = [&](Direction direction, StringRef field, Type portType) {
        // Map subfields of the memory port to module ports.
        auto accesses = getAllFieldAccesses(op.getResult(i), field);
        for (auto a : accesses)
          returnHolder[a] = portTypes.size() - 1;
        // Record the new port information.
        portTypes.push_back(portType);
        portDirections.push_back(direction);
        portNames.push_back(
            builder.getStringAttr(portLabel + Twine(portNumber) + "_" + field));
      };

      if (memportKind == MemOp::PortKind::Read) {
        addPort(Direction::In, "addr", getAddressType(summary.depth));
        addPort(Direction::In, "en", getType(1));
        addPort(Direction::In, "clk", getType(1));
        addPort(Direction::Out, "data", op.getDataType());
      } else if (memportKind == MemOp::PortKind::ReadWrite) {
        addPort(Direction::In, "addr", getAddressType(summary.depth));
        addPort(Direction::In, "en", getType(1));
        addPort(Direction::In, "clk", getType(1));
        addPort(Direction::In, "wmode", getType(1));
        addPort(Direction::In, "wdata", op.getDataType());
        addPort(Direction::Out, "rdata", op.getDataType());
        // Ignore mask port, if maskBits =1
        if (summary.isMasked)
          addPort(Direction::In, "wmask", getType(summary.maskBits));
      } else {
        addPort(Direction::In, "addr", getAddressType(summary.depth));
        addPort(Direction::In, "en", getType(1));
        addPort(Direction::In, "clk", getType(1));
        addPort(Direction::In, "data", op.getDataType());
        // Ignore mask port, if maskBits == 1
        if (summary.isMasked)
          addPort(Direction::In, "mask", getType(summary.maskBits));
      }

      ++portNumber;
    }
  }

  // Create the instance to replace the memop. The instance name matches the
  // name of the memory module.
  auto inst = builder.create<InstanceOp>(
      op.getLoc(), portTypes, summary.getFirMemoryName(),
      summary.getFirMemoryName(), portDirections, portNames, op.annotations(),
      op.portAnnotations(), /*lowerToBind=*/false, op.inner_symAttr());

  // Update all users of the result of read ports
  for (auto [subfield, result] : returnHolder)
    subfield->getResult(0).replaceAllUsesWith(inst.getResult(result));
  op->erase();
}

void LowerMemoryPass::lowerMemory(MemOp mem, bool shouldDedup) {
  auto summary = getOrCreateMemModule(mem, shouldDedup);
  emitMemoryInstance(mem, summary);
}

void LowerMemoryPass::runOnModule(FModuleOp module, bool shouldDedup) {
  module.getBody()->walk([&](MemOp op) { lowerMemory(op, shouldDedup); });
}

void LowerMemoryPass::runOnOperation() {
  auto circuit = getOperation();
  auto *body = circuit.getBody();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  // The design under test module.
  FModuleOp dutMod;

  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto *node = instanceGraph.lookup(&(*it));
    llvm::for_each(llvm::depth_first(node), [&](hw::InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }

  // We iterate the circuit from top-to-bottom to make sure that we get
  // consistent memory names.
  for (auto module : body->getOps<FModuleOp>())
    runOnModule(module, /*shouldDedup=*/dutModuleSet.contains(module));

  memories.clear();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerMemoryPass() {
  return std::make_unique<LowerMemoryPass>();
}