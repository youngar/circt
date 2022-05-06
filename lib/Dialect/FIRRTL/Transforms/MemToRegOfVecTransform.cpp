//===- MemToRegOfVecTransform.cpp - MemToRegOfVecTransform Pass
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MemToRegOfVecTransform pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mem-to-regOfVec"

using namespace circt;
using namespace firrtl;

namespace {
struct MemToRegOfVecTransformPass
    : public MemToRegOfVecTransformBase<MemToRegOfVecTransformPass> {
  MemToRegOfVecTransformPass(bool replSeqMem, bool ignoreReadEnable)
      : replSeqMem(replSeqMem), ignoreReadEnable(ignoreReadEnable){};
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs()
               << "\n Running MemToRegOfVecTransformPass on module:"
               << getOperation().getName());
    getOperation().getBody()->walk([&](MemOp memOp) {
      LLVM_DEBUG(llvm::dbgs() << "\n Memory op:" << memOp);

      auto firMem = memOp.getSummary();
      // Ignore if the memory is candidate for macro replacement.
      // The requirements for macro replacement:
      // 1. read latency and write latency of one.
      // 2. only one readwrite port or write port.
      // 3. zero or one read port.
      // 4. undefined read-under-write behavior.
      if (replSeqMem &&
          ((firMem.readLatency == 1 && firMem.writeLatency == 1) &&
           (firMem.numWritePorts + firMem.numReadWritePorts == 1) &&
           (firMem.numReadPorts <= 1) && firMem.dataWidth > 0))
        return;

      generateMemory(memOp, firMem);
      memOp.erase();
    });
  }

  Value addPipelineStages(ImplicitLocOpBuilder &b, size_t stages, Value clock,
                          Value pipeInput, StringRef name, Value gate = {}) {
    if (!stages)
      return pipeInput;

    while (stages--) {
      auto reg = b.create<RegOp>(pipeInput.getType(), clock,
                                 moduleNamespace.newName(name));
      if (gate) {
        b.create<WhenOp>(gate, /*withElseRegion*/ false,
                         [&]() { b.create<StrictConnectOp>(reg, pipeInput); });
      } else
        b.create<StrictConnectOp>(reg, pipeInput);

      pipeInput = reg;
    }

    return pipeInput;
  }

  Value getClock(ImplicitLocOpBuilder &builder, Value bundle) {
    return builder.create<SubfieldOp>(bundle, "clk");
  }

  Value getAddr(ImplicitLocOpBuilder &builder, Value bundle) {
    return builder.create<SubfieldOp>(bundle, "addr");
  }

  Value getWmode(ImplicitLocOpBuilder &builder, Value bundle) {
    return builder.create<SubfieldOp>(bundle, "wmode");
  }

  Value getEnable(ImplicitLocOpBuilder &builder, Value bundle) {
    return builder.create<SubfieldOp>(bundle, "en");
  }

  Value getMask(ImplicitLocOpBuilder &builder, Value bundle) {
    auto bType = bundle.getType().cast<FIRRTLType>().cast<BundleType>();
    if (bType.getElement("mask").hasValue())
      return builder.create<SubfieldOp>(bundle, "mask");
    return builder.create<SubfieldOp>(bundle, "wmask");
  }

  Value getData(ImplicitLocOpBuilder &builder, Value bundle,
                bool getWdata = false) {
    auto bType = bundle.getType().cast<FIRRTLType>().cast<BundleType>();
    if (bType.getElement("data").hasValue())
      return builder.create<SubfieldOp>(bundle, "data");
    if (bType.getElement("rdata").hasValue() && !getWdata)
      return builder.create<SubfieldOp>(bundle, "rdata");
    return builder.create<SubfieldOp>(bundle, "wdata");
  }

  void generateRead(FirMemory firMem, Value clock, Value addr, Value enable,
                    Value data, Value regOfVec, ImplicitLocOpBuilder &builder) {
    if (ignoreReadEnable) {
      // If read enable is ignored, then guard the address update with read
      // enable.
      for (size_t j = 0, e = firMem.readLatency; j != e; ++j) {
        auto enLast = enable;
        if (j < e - 1)
          enable = addPipelineStages(builder, 1, clock, enable, "en");
        addr = addPipelineStages(builder, 1, clock, addr, "addr", enLast);
      }
    } else {
      // Add pipeline stages to respect the read latency. One register for each
      // latency cycle.
      enable =
          addPipelineStages(builder, firMem.readLatency, clock, enable, "en");
      addr =
          addPipelineStages(builder, firMem.readLatency, clock, addr, "addr");
    }

    // Read the register[address] into a temporary.
    Value rdata = builder.create<SubaccessOp>(regOfVec, addr);
    if (!ignoreReadEnable) {
      // Initialize read data out with invalid.
      builder.create<StrictConnectOp>(
          data, builder.create<InvalidValueOp>(data.getType()));
      // If enable is true, then connect the data read from memory register.
      builder.create<WhenOp>(enable, /*withElseRegion*/ false, [&]() {
        builder.create<StrictConnectOp>(data, rdata);
      });
    } else {
      // Ignore read enable signal.
      builder.create<StrictConnectOp>(data, rdata);
    }
  }

  void generateWrite(FirMemory firMem, Value clock, Value addr, Value enable,
                     Value maskBits, Value wdataIn, Value regOfVec,
                     ImplicitLocOpBuilder &builder) {

    auto numStages = firMem.writeLatency - 1;
    // Add pipeline stages to respect the write latency. Intermediate registers
    // for each stage.
    addr = addPipelineStages(builder, numStages, clock, addr, "addr");
    enable = addPipelineStages(builder, numStages, clock, enable, "en");
    wdataIn = addPipelineStages(builder, numStages, clock, wdataIn, "wdata");
    maskBits = addPipelineStages(builder, numStages, clock, maskBits, "wmask");
    // Create the register access.
    auto rdata = builder.create<SubaccessOp>(regOfVec, addr);

    // The tuple for the access to individual fields of an aggregate data type.
    // Tuple::<register, data, mask>
    // The logic:
    // if (mask)
    //  register = data
    SmallVector<std::tuple<Value, Value, Value>, 8> loweredRegDataMaskFields;

    // Write to each aggregate data field is guarded by the corresponding mask
    // field. This means we have to generate read and write access for each
    // individual field of the aggregate type.
    // There are two options to handle this,
    // 1. FlattenMemory: cast the aggregate data into a UInt and generate
    // appropriate mask logic.
    // 2. Create access for each individual field of the aggregate type.
    // Here we implement the option 2 using getFields.
    // getFields, creates an access to each individual field of the data and
    // mask, and the corresponding field into the register.  It populates
    // the loweredRegDataMaskFields vector.
    // This is similar to what happens in LowerTypes.
    //
    if (!getFields(rdata, wdataIn, maskBits, loweredRegDataMaskFields,
                   builder)) {
      wdataIn.getDefiningOp()->emitOpError(
          "Cannot convert memory to bank of registers");
      return;
    }
    // If enable:
    builder.create<WhenOp>(enable, /*withElseRegion*/ false, [&]() {
      // For each data field. Only one field if not aggregate.
      for (auto regDataMask : loweredRegDataMaskFields) {
        auto regField = std::get<0>(regDataMask);
        auto dataField = std::get<1>(regDataMask);
        auto maskField = std::get<2>(regDataMask);
        // If mask, then update the register field.
        builder.create<WhenOp>(maskField, /*withElseRegion*/ false, [&]() {
          builder.create<StrictConnectOp>(regField, dataField);
        });
      }
    });
  }

  void generateReadWrite(FirMemory firMem, Value clock, Value addr,
                         Value enable, Value maskBits, Value wdataIn,
                         Value rdataOut, Value wmode, Value regOfVec,
                         ImplicitLocOpBuilder &builder) {

    // Add pipeline stages to respect the write latency. Intermediate registers
    // for each stage. Number of pipeline stages, max of read/write latency.
    auto numStages = std::max(firMem.readLatency, firMem.writeLatency) - 1;
    addr = addPipelineStages(builder, numStages, clock, addr, "addr");
    enable = addPipelineStages(builder, numStages, clock, enable, "en");
    wdataIn = addPipelineStages(builder, numStages, clock, wdataIn, "wdata");
    maskBits = addPipelineStages(builder, numStages, clock, maskBits, "wmask");

    // Read the register[address] into a temporary.
    Value rdata = builder.create<SubaccessOp>(regOfVec, addr);

    SmallVector<std::tuple<Value, Value, Value>, 8> loweredRegDataMaskFields;
    if (!getFields(rdata, wdataIn, maskBits, loweredRegDataMaskFields,
                   builder)) {
      wdataIn.getDefiningOp()->emitOpError(
          "Cannot convert memory to bank of registers");
      return;
    }
    // Initialize read data out with invalid.
    builder.create<StrictConnectOp>(
        rdataOut, builder.create<InvalidValueOp>(rdataOut.getType()));
    // If enable:
    builder.create<WhenOp>(enable, /*withElseRegion*/ false, [&]() {
      // If write mode:
      builder.create<WhenOp>(
          wmode, true,
          // Write block:
          [&]() {
            // For each data field. Only one field if not aggregate.
            for (auto regDataMask : loweredRegDataMaskFields) {
              auto regField = std::get<0>(regDataMask);
              auto dataField = std::get<1>(regDataMask);
              auto maskField = std::get<2>(regDataMask);
              // If mask true, then set the field.
              builder.create<WhenOp>(
                  maskField, /*withElseRegion*/ false, [&]() {
                    builder.create<StrictConnectOp>(regField, dataField);
                  });
            }
          },
          // Read block:
          [&]() { builder.create<StrictConnectOp>(rdataOut, rdata); });
    });
  }

  // Generate individual field accesses for an aggregate type. Return false if
  // it fails. Which can happen if invalid fields are present of the mask and
  // input types donot match. The assumption is that, \p reg and \p input have
  // exactly the same type. And \p mask has the same bundle fields, but each
  // field is of type UInt<1> So, populate the \p results with each field
  // access. For example, the first entry should be access to first field of \p
  // reg, first field of \p input and first field of \p mask.
  bool getFields(Value reg, Value input, Value mask,
                 SmallVectorImpl<std::tuple<Value, Value, Value>> &results,
                 ImplicitLocOpBuilder &builder) {

    // Check if the number of fields of mask and input type match.
    auto isValidMask = [&](FIRRTLType inType, FIRRTLType maskType) -> bool {
      if (auto bundle = inType.dyn_cast<BundleType>()) {
        if (auto mBundle = maskType.dyn_cast<BundleType>())
          return mBundle.getNumElements() == bundle.getNumElements();
      } else if (auto vec = inType.dyn_cast<FVectorType>()) {
        if (auto mVec = maskType.dyn_cast<FVectorType>())
          return mVec.getNumElements() == vec.getNumElements();
      } else
        return true;
      return false;
    };

    std::function<bool(Value, Value, Value)> flatAccess =
        [&](Value reg, Value input, Value mask) -> bool {
      FIRRTLType inType = input.getType().cast<FIRRTLType>();
      if (!isValidMask(inType, mask.getType().cast<FIRRTLType>())) {
        input.getDefiningOp()->emitOpError("Mask type is not valid");
        return false;
      }
      return TypeSwitch<FIRRTLType, bool>(inType)
          .Case<BundleType>([&](BundleType bundle) {
            for (size_t i = 0, e = bundle.getNumElements(); i != e; ++i)
              if (!flatAccess(builder.create<SubfieldOp>(reg, i),
                              builder.create<SubfieldOp>(input, i),
                              builder.create<SubfieldOp>(mask, i)))
                return false;
            return true;
          })
          .Case<FVectorType>([&](auto vector) {
            for (size_t i = 0, e = vector.getNumElements(); i != e; ++i)
              if (!flatAccess(builder.create<SubindexOp>(reg, i),
                              builder.create<SubindexOp>(input, i),
                              builder.create<SubindexOp>(mask, i)))
                return false;
            return true;
          })
          .Case<IntType>([&](auto iType) {
            results.push_back({reg, input, mask});
            return iType.getWidth().hasValue();
          })
          .Default([&](auto) { return false; });
    };
    if (flatAccess(reg, input, mask))
      return true;
    return false;
  }

  /// Generate the logic for implementing the memory using Registers.
  void generateMemory(MemOp memOp, FirMemory &firMem) {
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    moduleNamespace.add(memOp->getParentOfType<FModuleOp>());
    auto dataType = memOp.getDataType();

    RegOp regOfVec = {};
    for (size_t index = 0, rend = memOp.getNumResults(); index < rend;
         ++index) {
      auto result = memOp.getResult(index);
      // Create a temporary wire to replace the memory port. This makes it
      // simpler to delete the memOp.
      auto wire = builder.create<WireOp>(
          result.getType(),
          (memOp.name() + "_" + memOp.getPortName(index).getValue()).str());
      result.replaceAllUsesWith(wire.getResult());
      result = wire;
      // Create an access to all the common subfields.
      auto adr = getAddr(builder, result);
      auto enb = getEnable(builder, result);
      auto clk = getClock(builder, result);
      auto dta = getData(builder, result);
      // IF the register is not yet created.
      if (!regOfVec) {
        // Create the register corresponding to the memory.
        regOfVec = builder.create<RegOp>(
            FVectorType::get(dataType, firMem.depth), clk, memOp.nameAttr());
        // Copy all the memory annotations.
        if (!memOp.annotationsAttr().empty())
          regOfVec.annotationsAttr(memOp.annotationsAttr());
      }
      auto portKind = memOp.getPortKind(index);
      if (portKind == MemOp::PortKind::Read) {
        generateRead(firMem, clk, adr, enb, dta, regOfVec, builder);
      } else if (portKind == MemOp::PortKind::Write) {
        auto mask = getMask(builder, result);
        generateWrite(firMem, clk, adr, enb, mask, dta, regOfVec, builder);
      } else {
        auto wmode = getWmode(builder, result);
        auto wDta = getData(builder, result, true);
        auto mask = getMask(builder, result);
        generateReadWrite(firMem, clk, adr, enb, mask, wDta, dta, wmode,
                          regOfVec, builder);
      }
    }
  }

private:
  bool replSeqMem;
  bool ignoreReadEnable;
  ModuleNamespace moduleNamespace;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createMemToRegOfVecTransformPass(bool replSeqMem,
                                                bool ignoreReadEnable) {
  return std::make_unique<MemToRegOfVecTransformPass>(replSeqMem,
                                                      ignoreReadEnable);
}
