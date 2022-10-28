#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include <mlir/IR/BuiltinAttributes.h>

using namespace circt;
using namespace firrtl;
using namespace mlir;

namespace {
struct UndefLatticeValue {
  enum Kind { Uninitialized, Constant, Overdefined };

  UndefLatticeValue(Kind kind, Attribute value) : kind(kind), value(value) {}

  template <Kind k>
  bool is() const {
    return kind == k;
  }

  static UndefLatticeValue getUninitialized() {
    return UndefLatticeValue(Uninitialized, nullptr);
  }
  static UndefLatticeValue getConstant(Attribute value) {
    return UndefLatticeValue(Constant, value);
  }
  static UndefLatticeValue getOverdefined() {
    return UndefLatticeValue(Overdefined, nullptr);
  }

  static UndefLatticeValue getPessimisticValueState(MLIRContext *context) {
    return getUninitialized();
  }

  static UndefLatticeValue getPessimisticValueState(Value value) {
    return getUninitialized();
  }

  static UndefLatticeValue join(const UndefLatticeValue &lhs,
                                const UndefLatticeValue &rhs) {
    if (lhs.is<Overdefined>() || rhs.is<Overdefined>())
      return getOverdefined();
    if ()
    
    

  }

  Kind kind;
  Attribute value;
}

} // namespace

namespace {
class UndefRegPass : public UndefRegPassBase<UndefRegPass> {
  void runOnOperation() override;
};
} // namespace

void UndefRegPass::runOnOperation() {}

std::unique_ptr<mlir::Pass> circt::firrtl::createUndefRegPass() {
  return std::make_unique<UndefRegPass>();
}