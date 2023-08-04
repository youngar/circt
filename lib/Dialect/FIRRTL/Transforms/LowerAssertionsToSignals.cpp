
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;

namespace {
struct LowerAssertionsToSignals
    : public LowerAssertionsToSignalsBase<LowerAssertionsToSignals> {
  void processModule(FModuleOp module);
  void emitJSON(FModuleOp module);
  void runOnOperation() override;

  InstanceGraph *instanceGraph;

  // Maps a module to all assertions under that module.
  struct Assertion {
    SmallVector<Attribute> path;
    StringRef message;
  };
  DenseMap<Operation *, std::vector<Assertion>> signals;

  // Stashed useful things.
  ArrayAttr dontTouch;
  FIRRTLBaseType uint1;
  StringAttr portName;
};
} // namespace

void LowerAssertionsToSignals::processModule(FModuleOp module) {
  auto *context = &getContext();

  ModuleNamespace ns(module);
  auto moduleName = module.getNameAttr();

  auto &assertions = signals[module];
  SmallVector<Value> localSignals;

  auto getInnerRefTo = [&](Operation *op) -> hw::InnerRefAttr {
    return ::getInnerRefTo(
        op, "", [&](FModuleOp mod) -> ModuleNamespace & { return ns; });
  };

  module.getBodyBlock()->walk([&](Operation *op) {
    if (auto assertOp = dyn_cast<AssertOp>(op)) {

      auto loc = assertOp->getLoc();
      auto enable = assertOp.getEnable();
      auto pred = assertOp.getPredicate();

      OpBuilder b(assertOp);
      StringRef name = "assert";
      if (!assertOp.getName().empty())
        name = assertOp.getName();

      auto innerSym = StringAttr::get(context, ns.newName(name));

      auto sig =
          b.create<AndPrimOp>(loc, enable, b.create<NotPrimOp>(loc, pred));

      auto n =
          b.create<NodeOp>(loc, uint1, sig, name, NameKindEnum::InterestingName,
                           dontTouch, innerSym);
      localSignals.push_back(n);
      auto &assertion = assertions.emplace_back();
      assertion.path.push_back(hw::InnerRefAttr::get(moduleName, innerSym));
      assertion.message = assertOp.getMessage();
      assertOp->erase();
      return;
    }

    if (auto coverOp = dyn_cast<CoverOp>(op)) {
      coverOp->erase();
      return;
    }

    if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      assumeOp->erase();
      return;
    }

    if (auto instance = dyn_cast<InstanceOp>(op)) {
      auto submodule = instanceGraph->getReferencedModule(instance);
      hw::InnerRefAttr instanceRef;
      for (auto &subassertion : signals[submodule]) {
        auto &assertion = assertions.emplace_back();
        if (!instanceRef)
          instanceRef = getInnerRefTo(instance);
        assertion.message = subassertion.message;
        assertion.path.push_back(instanceRef);
        llvm::copy(subassertion.path, std::back_inserter(assertion.path));
      }

      return;
    }
  });

  // Fixup instances of this module to have the additional localSignals.
  if (!localSignals.empty()) {
    auto loc = module.getLoc();

    auto portType = FVectorType::get(uint1, localSignals.size());
    auto portNo = module.getNumPorts();

    std::pair<unsigned, PortInfo> port{portNo,
                                       {portName, portType, Direction::Out}};
    port.second.annotations = AnnotationSet(dontTouch);

    module.insertPorts(port);

    auto b = OpBuilder::atBlockEnd(module.getBodyBlock());
    auto concat = b.create<VectorCreateOp>(loc, portType, localSignals);
    b.create<StrictConnectOp>(loc, module.getArgument(portNo), concat);

    for (auto *instRec : instanceGraph->lookup(module)->uses()) {
      auto inst = cast<InstanceOp>(*instRec->getInstance());
      auto clone = inst.cloneAndInsertPorts(port);
      inst->replaceAllUsesWith(clone.getResults().drop_back());
      inst->erase();
    }
  }
}

void LowerAssertionsToSignals::emitJSON(FModuleOp module) {
  auto *context = &getContext();
  auto circuit = getOperation();

  auto moduleName = module.getName();

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream json(os, 2);

  // The current parameter to the verbatim op.
  unsigned paramIndex = 0;
  // Parameters to the verbatim op.
  SmallVector<Attribute> params;
  // Small cache to reduce the number of parameters passed to the verbatim.
  DenseMap<Attribute, unsigned> usedParams;

  // Helper to add a symbol to the verbatim, hitting the cache on the way.
  auto addSymbol = [&](Attribute ref) {
    auto it = usedParams.find(ref);
    unsigned index;
    if (it != usedParams.end()) {
      index = it->second;
    } else {
      index = paramIndex;
      usedParams[ref] = paramIndex++;
      params.push_back(ref);
    }
    return index;
  };

  auto addAssertion = [&](Assertion &assertion) {
    SmallString<128> buffer;
    llvm::raw_svector_ostream os(buffer);

    os << "" << moduleName;
    for (auto e : assertion.path)
      os << ".{{" << addSymbol(e) << "}}";

    json.attribute(buffer.str(), assertion.message);
  };

  auto &assertions = signals[module];

  json.object([&]() {
    for (auto assertion : assertions) {
      addAssertion(assertion);
    }
  });

  auto builder = OpBuilder::atBlockEnd(circuit.getBodyBlock());
  auto outputFile =
      hw::OutputFileAttr::getFromFilename(context, "assertion-signals.json");
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      builder.getUnknownLoc(), buffer, ValueRange{},
      builder.getArrayAttr(params));
  verbatimOp->setAttr("output_file", outputFile);
}

void LowerAssertionsToSignals::runOnOperation() {
  auto *context = &getContext();
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();

  dontTouch = ArrayAttr::get(
      context,
      DictionaryAttr::get(
          context,
          {NamedAttribute(StringAttr::get(context, "class"),
                          StringAttr::get(context, dontTouchAnnoClass))}));

  uint1 = UIntType::get(context, 1);
  portName = StringAttr::get(context, "debug_assert");

  auto *topNode = instanceGraph->getTopLevelNode();
  for (auto *node : llvm::post_order(topNode)) {
    Operation *module = node->getModule();
    if (auto fmodule = dyn_cast<FModuleOp>(module))
      processModule(fmodule);
  }

  emitJSON(cast<FModuleOp>(topNode->getModule()));
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerAssertionsToSignals() {
  return std::make_unique<LowerAssertionsToSignals>();
}
