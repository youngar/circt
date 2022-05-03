//===- CreateSiFiveMetadata.cpp - Create various metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateSiFiveMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

static const char dutAnnoClass[] = "sifive.enterprise.firrtl.MarkDUTAnnotation";
/// Attribute that indicates where some json files should be dumped.
static const char metadataDirectoryAnnoClass[] =
    "sifive.enterprise.firrtl.MetadataDirAnnotation";

namespace {
class CreateSiFiveMetadataPass
    : public CreateSiFiveMetadataBase<CreateSiFiveMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
  LogicalResult emitSitestBlackboxMetadata();
  LogicalResult emitMemoryMetadata();
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  // The design under test module.
  FModuleOp dutMod;

public:
  CreateSiFiveMetadataPass(bool _replSeqMem, StringRef _replSeqMemCircuit,
                           StringRef _replSeqMemFile) {
    replSeqMem = _replSeqMem;
    replSeqMemCircuit = _replSeqMemCircuit.str();
    replSeqMemFile = _replSeqMemFile.str();
  }
};
} // end anonymous namespace

/// This function collects all the firrtl.mem ops and creates a verbatim op with
/// the relevant memory attributes.
LogicalResult CreateSiFiveMetadataPass::emitMemoryMetadata() {
  if (!replSeqMem)
    return success();

  CircuitOp circuitOp = getOperation();
  // The instance graph analysis will be required to print the hierarchy names
  // of the memory.
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());

  // This lambda, writes to the given Json stream all the relevant memory
  // attributes. Also adds the memory attrbutes to the string for creating the
  // memmory conf file.
  auto createMemMetadata = [&](FMemModuleOp mem,
                               llvm::json::OStream &jsonStream,
                               std::string &seqMemConfStr) {
    // Get the memory data width.
    auto width = mem.dataWidth();
    // Metadata needs to be printed for memories which are candidates for
    // macro replacement. The requirements for macro replacement::
    // 1. read latency and write latency of one.
    // 2. only one readwrite port or write port.
    // 3. zero or one read port.
    // 4. undefined read-under-write behavior.
    if (!((mem.readLatency() == 1 && mem.writeLatency() == 1) &&
          (mem.numWritePorts() + mem.numReadWritePorts() == 1) &&
          (mem.numReadPorts() <= 1) && width > 0))
      return;

    // Compute the mask granularity.
    auto isMasked = mem.isMasked();
    auto maskGran = width / mem.maskBits();
    // Now create the config string for the memory.
    std::string portStr;
    if (mem.numWritePorts() && isMasked)
      portStr += "mwrite";
    else if (mem.numWritePorts())
      portStr += "write";
    if (mem.numReadPorts()) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    if (mem.numReadWritePorts() && isMasked)
      portStr = "mrw";
    else if (mem.numReadWritePorts())
      portStr = "rw";
    auto memExtName = mem.getName();
    auto maskGranStr =
        !isMasked ? "" : " mask_gran " + std::to_string(maskGran);
    seqMemConfStr = (StringRef(seqMemConfStr) + "name " + memExtName +
                     " depth " + Twine(mem.depth()) + " width " + Twine(width) +
                     " ports " + portStr + maskGranStr + "\n")
                        .str();
    // This adds a Json array element entry corresponding to this memory.
    jsonStream.object([&] {
      jsonStream.attribute("module_name", memExtName);
      jsonStream.attribute("depth", (int64_t)mem.depth());
      jsonStream.attribute("width", (int64_t)width);
      jsonStream.attribute("masked", isMasked);
      jsonStream.attribute("read", mem.numReadPorts() > 0);
      jsonStream.attribute("write", mem.numWritePorts() > 0);
      jsonStream.attribute("readwrite", mem.numReadWritePorts() > 0);
      if (isMasked)
        jsonStream.attribute("mask_granularity", (int64_t)maskGran);
      jsonStream.attributeArray("extra_ports", [&] {});
      // Record all the hierarchy names.
      SmallVector<std::string> hierNames;
      jsonStream.attributeArray("hierarchy", [&] {
        // Get the absolute path for the parent memory, to create the
        // hierarchy names.
        auto paths = instancePathCache.getAbsolutePaths(mem);
        for (auto p : paths) {
          if (p.empty())
            continue;
          const InstanceOp &inst = p.front();
          std::string hierName =
              inst->getParentOfType<FModuleOp>().getName().str();
          for (InstanceOp inst : p) {
            hierName = hierName + "." + inst.name().str();
          }
          hierNames.push_back(hierName);
          jsonStream.value(hierName);
        }
      });
    });
  };

  SmallVector<FMemModuleOp> dutMems;
  SmallVector<FMemModuleOp> tbMems;
  for (auto mod : circuitOp.getOps<FMemModuleOp>()) {
    if (dutModuleSet.contains(mod))
      dutMems.push_back(mod);
    else
      tbMems.push_back(mod);
  }

  std::string testBenchJsonBuffer;
  llvm::raw_string_ostream testBenchOs(testBenchJsonBuffer);
  llvm::json::OStream testBenchJson(testBenchOs);
  std::string dutJsonBuffer;
  llvm::raw_string_ostream dutOs(dutJsonBuffer);
  llvm::json::OStream dutJson(dutOs);

  std::string seqMemConfStr;
  dutJson.array([&] {
    for (auto &dutM : dutMems)
      createMemMetadata(dutM, dutJson, seqMemConfStr);
  });
  testBenchJson.array([&] {
    // The tbConfStr is populated here, but unused, it will not be printed to
    // file.
    for (auto &tbM : tbMems)
      createMemMetadata(tbM, testBenchJson, seqMemConfStr);
  });

  auto *context = &getContext();
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  AnnotationSet annos(circuitOp);
  auto dirAnno = annos.getAnnotation(metadataDirectoryAnnoClass);
  StringRef metadataDir = "metadata";
  if (dirAnno)
    if (auto dir = dirAnno.getMember<StringAttr>("dirname"))
      metadataDir = dir.getValue();

  // Use unknown loc to avoid printing the location in the metadata files.
  auto tbVerbatimOp = builder.create<sv::VerbatimOp>(builder.getUnknownLoc(),
                                                     testBenchJsonBuffer);
  auto fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, "tb_seq_mems.json", /*excludeFromFilelist=*/true);
  tbVerbatimOp->setAttr("output_file", fileAttr);
  auto dutVerbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), dutJsonBuffer);
  fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
      context, metadataDir, "seq_mems.json", /*excludeFromFilelist=*/true);
  dutVerbatimOp->setAttr("output_file", fileAttr);

  if (!seqMemConfStr.empty()) {
    auto confVerbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), seqMemConfStr);
    if (replSeqMemFile.empty()) {
      circuitOp->emitError("metadata emission failed, the option "
                           "`-repl-seq-mem-file=<filename>` is mandatory for "
                           "specifying a valid seq mem metadata file");
      return failure();
    }

    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, replSeqMemFile, /*excludeFromFilelist=*/true);
    confVerbatimOp->setAttr("output_file", fileAttr);
  }

  return success();
}

/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult CreateSiFiveMetadataPass::emitRetimeModulesMetadata() {

  // Circuit level annotation.
  auto *outputFileNameAnnotation =
      "sifive.enterprise.firrtl.RetimeModulesAnnotation";
  // Per module annotation.
  auto *retimeModuleAnnoClass =
      "freechips.rocketchip.util.RetimeModuleAnnotation";

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, outputFileNameAnnotation,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBody()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(module.moduleNameAttr()));
    }
  });

  // Put the retime information in a verbatim operation.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), buffer, ValueRange(),
      builder.getArrayAttr(symbols));
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, filename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult CreateSiFiveMetadataPass::emitSitestBlackboxMetadata() {
  auto *dutBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
  auto *testBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";

  // Any extmodule with these annotations or one of these ScalaClass classes
  // should be excluded from the blackbox list.
  auto *scalaClassAnnoClass = "sifive.enterprise.firrtl.ScalaClassAnnotation";
  std::array<StringRef, 3> classBlackList = {
      "freechips.rocketchip.util.BlackBoxedROM", "chisel3.shim.CloneModule",
      "sifive.enterprise.grandcentral.MemTap"};
  std::array<StringRef, 6> blackListedAnnos = {
      "firrtl.transforms.BlackBox", "firrtl.transforms.BlackBoxInlineAnno",
      "sifive.enterprise.grandcentral.DataTapsAnnotation",
      "sifive.enterprise.grandcentral.MemTapAnnotation",
      "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"};

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, dutBlackboxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(circuitOp, testBlackboxAnnoClass,
                                          testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two groups
  // depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBody()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.defname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // If its a blacklisted scala class, skip it.
    if (auto scalaAnno = annos.getAnnotation(scalaClassAnnoClass)) {
      auto scalaClass = scalaAnno.getMember<StringAttr>("className");
      if (scalaClass &&
          llvm::is_contained(classBlackList, scalaClass.getValue()))
        continue;
    }

    // Record the defname of the module.
    if (dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.defname());
    } else {
      testModules.push_back(*extModule.defname());
    }
  }

  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os);
    j.array([&] {
      for (auto &name : names)
        j.value(name);
    });

    auto *body = circuitOp.getBody();
    // Put the information in a verbatim operation.
    auto builder = OpBuilder::atBlockEnd(body);
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
  };

  createOutput(testModules, testFilename);
  createOutput(dutModules, dutFilename);
  return success();
}

void CreateSiFiveMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect>();
}

void CreateSiFiveMetadataPass::runOnOperation() {
  auto circuitOp = getOperation();
  auto *body = circuitOp.getBody();

  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    auto *node = instanceGraph.lookup(&(*it));
    llvm::for_each(llvm::depth_first(node), [&](hw::InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }

  if (failed(emitRetimeModulesMetadata()) ||
      failed(emitSitestBlackboxMetadata()) || failed(emitMemoryMetadata()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateSiFiveMetadataPass(
    bool replSeqMem, StringRef replSeqMemCircuit, StringRef replSeqMemFile) {
  return std::make_unique<CreateSiFiveMetadataPass>(
      replSeqMem, replSeqMemCircuit, replSeqMemFile);
}
