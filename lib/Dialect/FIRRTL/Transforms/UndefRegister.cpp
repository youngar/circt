#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

using namespace circt;
using namespace firrtl;
using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// FieldRefPoint
//===----------------------------------------------------------------------===//

struct FieldRefPoint : public GenericProgramPointBase<FieldRefPoint, FieldRef> {
  using Base::Base;
  void print(raw_ostream &os) const override;
  Location getLoc() const override;
};

void FieldRefPoint::print(raw_ostream &os) const { os << getValue(); }

Location FieldRefPoint::getLoc() const { return getValue().getLoc(); }

//===----------------------------------------------------------------------===//
// FieldState
//===----------------------------------------------------------------------===//

struct FieldState : public AnalysisState {
  using AnalysisState::AnalysisState;
  
  FieldState(FieldRefPoint point) : AnalysisState(point) {}
  //FieldState(FieldRef field) : FieldState(FieldRefPoint(field)) {}

  void onUpdate(DataFlowSolver *solver) const override {
    // for (auto *user : point.get < Value().getUsers())
  }
};

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

struct FieldAnalysis : public DataFlowAnalysis {

};

//===----------------------------------------------------------------------===//
// ResetState
//===----------------------------------------------------------------------===//

struct ResetState : public FieldState {
  bool isInitialized() const { return initialized; }

  ResetState markInitialized() {
    if (!isInitialized()) {
      initialized = true;
      return ResetState::Change;
    }
    return ResetState::NoChange;
  }

  ResetState reset() {
    if (isInitialized()) {
      initialized = false;
      return ResetState::Change;
    }
    return ResetState::NoChange;
  }

  bool operator==(const FieldState &rhs) const {
    return initialized == rhs.initialized;
  }

  static FieldState join(const FieldState &lhs, const FieldState &rhs) {
    if (lhs.isInitialized())
      return lhs;
    if (rhs.isInitialized())
      return rhs;
    return lhs;
  }

  ResetState join(const FieldState &rhs) {
    if (isInitialized())
      return ResetState::NoChange;
    if (rhs.isInitialized()) {
      markInitialized();
      return ResetState::Change;
    }
    return ResetState::NoChange;
  }

  void print(raw_ostream &os) const override {
    os << (isInitialized() ? "reset" : "noreset");
  }

private:
  bool initialized = false;
};

//===----------------------------------------------------------------------===//
// ResetAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct ResetAnalysis : public DataFlowAnalysis {
  explicit ResetAnalysis(DataFlowSolver &solver, InstanceGraph &instanceGraph);
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  LogicalResult visitOperation(Operation *op);
  LogicalResult visitBlock(Block *block);
  InstanceGraph &instanceGraph;
};
} // namespace

ResetAnalysis::ResetAnalysis(DataFlowSolver &solver,
                             InstanceGraph &instanceGraph)
    : DataFlowAnalysis(solver), instanceGraph(instanceGraph) {}

LogicalResult ResetAnalysis::visitOperation(Operation *op) {
  if (auto reg = dyn_cast<RegOp>(op)) {
    /// auto state = getOrCreateFor<ResetValue>(reg.));
    // do nothing?
    return success();
  }
  if (auto regReset = dyn_cast<RegOp>(op)) {
    auto *state = getOrCreate<ResetValue>(op);
    propagateIfChanged(state, state->markInitialized());
    return success();
  }
  if (auto connect = dyn_cast<ConnectOp>(op)) {
    // Not sure what to do here?
    auto *dst = getOrCreate<ResetValue>(connect.getDest());
    auto *src = getOrCreate<ResetValue>(connect.getSrc());
    propagateIfChanged(dst, dst->join(*src));
    return success();
  }
  if (auto mux = dyn_cast<MuxPrimOp>(op)) {
    auto *state = getOrCreate<ResetValue>(mux);
    auto result = state->join(*getOrCreate<ResetValue>(mux.getHigh()));
    result |= state->join(*getOrCreate<ResetValue>(mux.getLow()));
    propagateIfChanged(state, result);
    return success();
  }
  auto *state = getOrCreate<ResetValue>(op);
  propagateIfChanged(state, state->markInitialized());
  return success();
}

LogicalResult ResetAnalysis::visitBlock(Block *block) { return success(); }

LogicalResult ResetAnalysis::initialize(Operation *top) {
  // Mark all the ports of the top-level module as
  auto circuit = cast<CircuitOp>(top);
  auto module = dyn_cast<FModuleOp>(instanceGraph.getTopLevelModule());

  // Initialize the top level module ports.
  for (auto &arg : module.getArguments()) {
    auto *state = getOrCreate<ResetValue>(arg);
    propagateIfChanged(state, state->markInitialized());
  }

  // Initialize the module body.
  module->walk([&](Operation *op) { visitOperation(op); });

  return success();
}

LogicalResult ResetAnalysis::visit(ProgramPoint point) {
  llvm::errs() << "point: " << point << "\n";
  auto value = point.dyn_cast<Value>();
  abort();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//



namespace {
class UndefRegPass : public UndefRegPassBase<UndefRegPass> {
  LogicalResult check(CircuitOp circuit, DataFlowSolver &solver);
  void runOnOperation() override;
};
} // namespace

LogicalResult UndefRegPass::check(CircuitOp circuit, DataFlowSolver &solver) {
  circuit->walk([&](Operation *op) {

  });
  return success();
}

void UndefRegPass::runOnOperation() {
  auto circuit = getOperation();
  DataFlowSolver solver;
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  solver.load<ResetAnalysis>(instanceGraph);
  if (failed(solver.initializeAndRun(circuit)))
    return signalPassFailure();
  if (failed(check(circuit, solver)))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createUndefRegPass() {
  return std::make_unique<UndefRegPass>();
}