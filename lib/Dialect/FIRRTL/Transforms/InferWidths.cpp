//===- InferWidths.cpp - Infer width of types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferWidths pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "infer-widths"

using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Constraint Expressions
//===----------------------------------------------------------------------===//

namespace {
struct Expr;
} // namespace

/// Allow rvalue refs to `Expr` and subclasses to be printed to streams.
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T &e) {
  e.print(os);
  return os;
}

// Allow expression subclasses to be hashed.
namespace mlir {
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::hash_code hash_value(const T &e) {
  return e.hash_value();
}
} // namespace mlir

namespace {
#define EXPR_NAMES(x)                                                          \
  Root##x, Nil##x, Var##x, Known##x, Add##x, Pow##x, Max##x, Min##x
#define EXPR_KINDS EXPR_NAMES()
#define EXPR_CLASSES EXPR_NAMES(Expr)

/// An expression on the right-hand side of a constraint.
struct Expr {
  enum class Kind { EXPR_KINDS };
  llvm::Optional<int32_t> solution = {};
  Kind kind;

  /// Print a human-readable representation of this expr.
  void print(llvm::raw_ostream &os) const;

  // Iterators over the child expressions.
  typedef Expr *const *iterator;
  iterator begin() const;
  iterator end() const;

protected:
  Expr(Kind kind) : kind(kind) {}
  llvm::hash_code hash_value() const { return llvm::hash_value(kind); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct ExprBase : public Expr {
  ExprBase() : Expr(DerivedKind) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
  bool operator==(const Expr &other) const {
    if (auto otherSame = dyn_cast<DerivedT>(other))
      return *static_cast<DerivedT *>(this) == otherSame;
    return false;
  }
  iterator begin() const { return nullptr; }
  iterator end() const { return nullptr; }
};

/// A free variable.
struct RootExpr : public ExprBase<RootExpr, Expr::Kind::Root> {
  RootExpr(std::vector<Expr *> &exprs) : exprs(exprs) {}
  void print(llvm::raw_ostream &os) const { os << "root"; }
  iterator begin() const { return exprs.data(); }
  iterator end() const { return exprs.data() + exprs.size(); }
  std::vector<Expr *> &exprs;
};

/// A free variable.
struct NilExpr : public ExprBase<NilExpr, Expr::Kind::Nil> {
  void print(llvm::raw_ostream &os) const { os << "nil"; }
};

/// A free variable.
struct VarExpr : public ExprBase<VarExpr, Expr::Kind::Var> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable. Since
    // this is just for debug dumping, we wrap around at 4096 variables.
    os << "var" << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFF);
  }
  iterator begin() const { return &constraint; }
  iterator end() const { return &constraint + (constraint ? 1 : 0); }

  /// The constraint expression this variable is supposed to be greater than or
  /// equal to. This is not part of the variable's hash and equality property.
  Expr *constraint = nullptr;
};

/// A known constant value.
struct KnownExpr : public ExprBase<KnownExpr, Expr::Kind::Known> {
  KnownExpr(int32_t value) : ExprBase() { solution = value; }
  void print(llvm::raw_ostream &os) const { os << solution.getValue(); }
  bool operator==(const KnownExpr &other) const {
    return solution.getValue() == other.solution.getValue();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), solution.getValue());
  }
};

/// A unary expression. Contains the actual data. Concrete subclasses are merely
/// there for show and ease of use.
struct UnaryExpr : public Expr {
  bool operator==(const UnaryExpr &other) const {
    return kind == other.kind && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }
  iterator begin() const { return &arg; }
  iterator end() const { return &arg + 1; }

  /// The child expression.
  Expr *const arg;

protected:
  UnaryExpr(Kind kind, Expr *arg) : Expr(kind), arg(arg) { assert(arg); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct UnaryExprBase : public UnaryExpr {
  template <typename... Args>
  UnaryExprBase(Args &&...args)
      : UnaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// A power of two.
struct PowExpr : public UnaryExprBase<PowExpr, Expr::Kind::Pow> {
  using UnaryExprBase::UnaryExprBase;
  void print(llvm::raw_ostream &os) const { os << "2^" << arg; }
};

/// A binary expression. Contains the actual data. Concrete subclasses are
/// merely there for show and ease of use.
struct BinaryExpr : public Expr {
  bool operator==(const BinaryExpr &other) const {
    return kind == other.kind && lhs() == other.lhs() && rhs() == other.rhs();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *args);
  }
  Expr *lhs() const { return args[0]; }
  Expr *rhs() const { return args[1]; }
  iterator begin() const { return args; }
  iterator end() const { return args + 2; }

  /// The child expressions.
  Expr *const args[2];

protected:
  BinaryExpr(Kind kind, Expr *lhs, Expr *rhs) : Expr(kind), args{lhs, rhs} {
    assert(lhs);
    assert(rhs);
  }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct BinaryExprBase : public BinaryExpr {
  template <typename... Args>
  BinaryExprBase(Args &&...args)
      : BinaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// An addition.
struct AddExpr : public BinaryExprBase<AddExpr, Expr::Kind::Add> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "(" << *lhs() << " + " << *rhs() << ")";
  }
};

/// The maximum of two expressions.
struct MaxExpr : public BinaryExprBase<MaxExpr, Expr::Kind::Max> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "max(" << *lhs() << ", " << *rhs() << ")";
  }
};

/// The minimum of two expressions.
struct MinExpr : public BinaryExprBase<MinExpr, Expr::Kind::Min> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "min(" << *lhs() << ", " << *rhs() << ")";
  }
};

void Expr::print(llvm::raw_ostream &os) const {
  TypeSwitch<const Expr *>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { e->print(os); });
}

Expr::iterator Expr::begin() const {
  return TypeSwitch<const Expr *, Expr::iterator>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { return e->begin(); });
}

Expr::iterator Expr::end() const {
  return TypeSwitch<const Expr *, Expr::iterator>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { return e->end(); });
}

} // namespace

//===----------------------------------------------------------------------===//
// GraphTraits on constraint expressions
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct GraphTraits<Expr *> {
  using ChildIteratorType = Expr::iterator;
  using NodeRef = Expr *;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->end();
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Fast bump allocator with optional interning
//===----------------------------------------------------------------------===//

namespace {

/// An allocation slot in the `InternedAllocator`.
template <typename T>
struct InternedSlot {
  T *ptr;
  InternedSlot(T *ptr) : ptr(ptr) {}
};

/// A simple bump allocator that ensures only ever one copy per object exists.
/// The allocated objects must not have a destructor.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class InternedAllocator {
  using Slot = InternedSlot<T>;
  llvm::DenseSet<Slot> interned;
  llvm::BumpPtrAllocator &allocator;

public:
  InternedAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object if it does not yet exist, or return a pointer to the
  /// existing one. `R` is the type of the object to be allocated. `R` must be
  /// derived from or be the type `T`.
  template <typename R = T, typename... Args>
  std::pair<R *, bool> alloc(Args &&...args) {
    auto stack_value = R(std::forward<Args>(args)...);
    auto stack_slot = Slot(&stack_value);
    auto it = interned.find(stack_slot);
    if (it != interned.end())
      return std::make_pair(static_cast<R *>(it->ptr), false);
    auto heap_value = new (allocator) R(std::move(stack_value));
    interned.insert(Slot(heap_value));
    return std::make_pair(heap_value, true);
  }
};

/// A simple bump allocator. The allocated objects must not have a destructor.
/// This allocator is mainly there for symmetry with the `InternedAllocator`.
class VarAllocator {
  llvm::BumpPtrAllocator &allocator;
  using T = VarExpr;

public:
  VarAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object. `R` is the type of the object to be allocated. `R`
  /// must be derived from or be the type `T`.
  template <typename R = T, typename... Args>
  R *alloc(Args &&...args) {
    return new (allocator) R(std::forward<Args>(args)...);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Constraint Solver
//===----------------------------------------------------------------------===//

namespace {

/// A simple solver for width constraints.
class ConstraintSolver {
public:
  ConstraintSolver() = default;

  NilExpr *nil() { return &singletonNil; }
  VarExpr *var() {
    auto v = vars.alloc();
    exprs.push_back(v);
    if (currentInfo)
      info[v].insert(currentInfo);
    return v;
  }
  KnownExpr *known(int32_t value) { return alloc<KnownExpr>(knowns, value); }
  PowExpr *pow(Expr *arg) { return alloc<PowExpr>(uns, arg); }
  AddExpr *add(Expr *lhs, Expr *rhs) { return alloc<AddExpr>(bins, lhs, rhs); }
  MaxExpr *max(Expr *lhs, Expr *rhs) { return alloc<MaxExpr>(bins, lhs, rhs); }
  MinExpr *min(Expr *lhs, Expr *rhs) { return alloc<MinExpr>(bins, lhs, rhs); }

  /// Add a constraint `lhs >= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `max(a, b)` expr.
  Expr *addGeqConstraint(VarExpr *lhs, Expr *rhs) {
    lhs->constraint = lhs->constraint ? max(lhs->constraint, rhs) : rhs;
    return lhs->constraint;
  }

  void dumpConstraints(llvm::raw_ostream &os);
  LogicalResult solve();

  using ContextInfo = DenseMap<Expr *, llvm::SmallSetVector<Value, 1>>;
  const ContextInfo &getContextInfo() const { return info; }
  void setCurrentContextInfo(Value value) { currentInfo = value; }

private:
  // Allocator for constraint expressions.
  llvm::BumpPtrAllocator allocator;
  static NilExpr singletonNil;
  VarAllocator vars = {allocator};
  InternedAllocator<KnownExpr> knowns = {allocator};
  InternedAllocator<UnaryExpr> uns = {allocator};
  InternedAllocator<BinaryExpr> bins = {allocator};

  /// A list of expressions in the order they were created.
  std::vector<Expr *> exprs;
  RootExpr root = {exprs};

  /// Add an allocated expression to the list above.
  template <typename R, typename T, typename... Args>
  R *alloc(InternedAllocator<T> &allocator, Args &&...args) {
    auto it = allocator.template alloc<R>(std::forward<Args>(args)...);
    if (it.second)
      exprs.push_back(it.first);
    if (currentInfo)
      info[it.first].insert(currentInfo);
    return it.first;
  }

  /// Contextual information for each expression, indicating which values in the
  /// IR lead to this expression.
  ContextInfo info;
  Value currentInfo = {};

  // Forbid copyign or moving the solver, which would invalidate the refs to
  // allocator held by the allocators.
  ConstraintSolver(ConstraintSolver &&) = delete;
  ConstraintSolver(const ConstraintSolver &) = delete;
  ConstraintSolver &operator=(ConstraintSolver &&) = delete;
  ConstraintSolver &operator=(const ConstraintSolver &) = delete;
};

} // namespace

NilExpr ConstraintSolver::singletonNil;

/// Print all constraints in the solver to an output stream.
void ConstraintSolver::dumpConstraints(llvm::raw_ostream &os) {
  for (auto *e : exprs) {
    if (auto *v = dyn_cast<VarExpr>(e)) {
      if (v->constraint)
        os << "- " << *v << " >= " << *v->constraint << "\n";
      else
        os << "- " << *v << " unconstrained\n";
    }
  }
}

/// A canonicalized linear inequality that maps a constraint on var `x` to the
/// linear inequality `x >= max(a*x+b, c) + (failed ? ∞ : 0)`.
///
/// The inequality separately tracks recursive (a, b) and non-recursive (c)
/// constraints on `x`. This allows it to properly identify the combination of
/// the two constraints constraints `x >= x-1` and `x >= 4` to be satisfiable as
/// `x >= max(x-1, 4)`. If it only tracked inequality as `x >= a*x+b`, the
/// combination of these two constraints would be `x >= x+4` (due to max(-1,4) =
/// 4), which would be unsatisfiable.
///
/// The `failed` flag acts as an additional `∞` term that renders the inequality
/// unsatisfiable. It is used as a tombstone value in case an operation renders
/// the equality unsatisfiable (e.g. `x >= 2**x` would be represented as the
/// inequality `x >= ∞`).
///
/// Inequalities represented in this form can easily be checked for
/// unsatisfiability in the presence of recursion by inspecting the coefficients
/// a and b. The `sat` function performs this action.
struct LinIneq {
  // x >= max(a*x+b, c) + (failed ? ∞ : 0)
  int32_t rec_scale = 0;   // a
  int32_t rec_bias = 0;    // b
  int32_t nonrec_bias = 0; // c
  bool failed = false;

  /// Create a new unsatisfiable inequality `x >= ∞`.
  static LinIneq unsat() { return LinIneq(true); }

  /// Create a new inequality `x >= (failed ? ∞ : 0)`.
  explicit LinIneq(bool failed = false) : failed(failed) {}

  /// Create a new inequality `x >= bias`.
  explicit LinIneq(int32_t bias) : nonrec_bias(bias) {}

  /// Create a new inequality `x >= scale*x+bias`.
  explicit LinIneq(int32_t scale, int32_t bias) {
    if (scale != 0) {
      rec_scale = scale;
      rec_bias = bias;
    } else {
      nonrec_bias = bias;
    }
  }

  /// Create a new inequality `x >= max(rec_scale*x+rec_bias, nonrec_bias) +
  /// (failed ? ∞ : 0)`.
  explicit LinIneq(int32_t rec_scale, int32_t rec_bias, int32_t nonrec_bias,
                   bool failed = false)
      : failed(failed) {
    if (rec_scale != 0) {
      this->rec_scale = rec_scale;
      this->rec_bias = rec_bias;
      this->nonrec_bias = nonrec_bias;
    } else {
      this->nonrec_bias = std::max(rec_bias, nonrec_bias);
    }
  }

  /// Combine two inequalities by taking the maxima of corresponding
  /// coefficients.
  ///
  /// This essentially combines `x >= max(a1*x+b1, c1)` and `x >= max(a2*x+b2,
  /// c2)` into a new `x >= max(max(a1,a2)*x+max(b1,b2), max(c1,c2))`. This is
  /// a pessimistic upper bound, since e.g. `x >= 2x-10` and `x >= x-5` may both
  /// hold, but the resulting `x >= 2x-5` may pessimistically not hold.
  static LinIneq max(const LinIneq &lhs, const LinIneq &rhs) {
    return LinIneq(std::max(lhs.rec_scale, rhs.rec_scale),
                   std::max(lhs.rec_bias, rhs.rec_bias),
                   std::max(lhs.nonrec_bias, rhs.nonrec_bias),
                   lhs.failed || rhs.failed);
  }

  /// Combine two inequalities by summing up the two right hand sides.
  ///
  /// This is a tricky one, since the addition of the two max terms will lead to
  /// a maximum over four possible terms (similar to a binomial expansion). In
  /// order to shoehorn this back into a two-term maximum, we have to pick the
  /// recursive term that will grow the fastest.
  ///
  /// As an example for this problem, consider the following addition:
  ///
  ///   x >= max(a1*x+b1, c1) + max(a2*x+b2, c2)
  ///
  /// We would like to expand and rearrange this again into a maximum:
  ///
  ///   x >= max(a1*x+b1 + max(a2*x+b2, c2), c1 + max(a2*x+b2, c2))
  ///   x >= max(max(a1*x+b1 + a2*x+b2, a1*x+b1 + c2),
  ///            max(c1 + a2*x+b2, c1 + c2))
  ///   x >= max((a1+a2)*x+(b1+b2), a1*x+(b1+c2), a2*x+(b2+c1), c1+c2)
  ///
  /// Since we are combining two two-term maxima, there are four possible ways
  /// how the terms can combine, leading to the above four-term maximum. An easy
  /// upper bound of the form we want would be the following:
  ///
  ///   x >= max(max(a1+a2, a1, a2)*x + max(b1+b2, b1+c2, b2+c1), c1+c2)
  ///
  /// However, this is a very pessimistic upper-bound that will declare very
  /// common patterns in the IR as unbreakable cycles, despite them being very
  /// much breakable. For example:
  ///
  ///   x >= max(x, 42) + max(0, -3)  <-- breakable recursion
  ///   x >= max(max(1+0, 1, 0)*x + max(42+0, -3, 42), 42-2)
  ///   x >= max(x + 42, 39)          <-- unbreakable recursion!
  ///
  /// A better approach is to take the expanded four-term maximum, retain the
  /// non-recursive term (c1+c2), and estimate which one of the recursive terms
  /// (first three) will become dominant as we choose greater values for x.
  /// Since x never is inferred to be negative, the recursive term in the
  /// maximum with the highest scaling factor for x will end up dominating as
  /// x tends to ∞:
  ///
  ///   x >= max({
  ///     (a1+a2)*x+(b1+b2) if a1+a2 >= max(a1+a2, a1, a2) and a1>0 and a2>0,
  ///     a1*x+(b1+c2)      if    a1 >= max(a1+a2, a1, a2) and a1>0,
  ///     a2*x+(b2+c1)      if    a2 >= max(a1+a2, a1, a2) and a2>0,
  ///     0                 otherwise
  ///   }, c1+c2)
  ///
  /// In case multiple cases apply, the highest bias of the recursive term is
  /// picked. With this, the above problematic example triggers the second case
  /// and becomes:
  ///
  ///   x >= max(1*x+(0-3), 42-3) = max(x-3, 39)
  ///
  /// Of which the first case is chosen, as it has the lower bias value.
  static LinIneq add(const LinIneq &lhs, const LinIneq &rhs) {
    // Determine the maximum scaling factor among the three possible recursive
    // terms.
    auto enable1 = lhs.rec_scale > 0 && rhs.rec_scale > 0;
    auto enable2 = lhs.rec_scale > 0;
    auto enable3 = rhs.rec_scale > 0;
    auto scale1 = lhs.rec_scale + rhs.rec_scale; // (a1+a2)
    auto scale2 = lhs.rec_scale;                 // a1
    auto scale3 = rhs.rec_scale;                 // a2
    auto bias1 = lhs.rec_bias + rhs.rec_bias;    // (b1+b2)
    auto bias2 = lhs.rec_bias + rhs.nonrec_bias; // (b1+c2)
    auto bias3 = rhs.rec_bias + lhs.nonrec_bias; // (b2+c1)
    auto maxScale = std::max(scale1, std::max(scale2, scale3));

    // Among those terms that have a maximum scaling factor, determine the
    // largest bias value.
    Optional<int32_t> maxBias = llvm::None;
    if (enable1 && scale1 == maxScale)
      maxBias = bias1;
    if (enable2 && scale2 == maxScale && (!maxBias || bias2 > *maxBias))
      maxBias = bias2;
    if (enable3 && scale3 == maxScale && (!maxBias || bias3 > *maxBias))
      maxBias = bias3;

    // Pick from the recursive terms the one with maximum scaling factor and
    // minimum bias value.
    auto nonrec_bias = lhs.nonrec_bias + rhs.nonrec_bias; // c1+c2
    auto failed = lhs.failed || rhs.failed;
    if (enable1 && scale1 == maxScale && bias1 == *maxBias)
      return LinIneq(scale1, bias1, nonrec_bias, failed);
    if (enable2 && scale2 == maxScale && bias2 == *maxBias)
      return LinIneq(scale2, bias2, nonrec_bias, failed);
    if (enable3 && scale3 == maxScale && bias3 == *maxBias)
      return LinIneq(scale3, bias3, nonrec_bias, failed);
    return LinIneq(0, 0, nonrec_bias, failed);
  }

  /// Check if the inequality is satisfiable.
  ///
  /// The inequality becomes unsatisfiable if the RHS is ∞, or a>1, or a==1 and
  /// b <= 0. Otherwise there exists as solution for `x` that satisfies the
  /// inequality.
  bool sat() const {
    if (failed)
      return false;
    if (rec_scale > 1)
      return false;
    if (rec_scale == 1 && rec_bias > 0)
      return false;
    return true;
  }

  /// Dump the inequality in human-readable form.
  void print(llvm::raw_ostream &os) const {
    bool any = false;
    bool both = (rec_scale != 0 || rec_bias != 0) && nonrec_bias != 0;
    os << "x >= ";
    if (both)
      os << "max(";
    if (rec_scale != 0) {
      any = true;
      if (rec_scale != 1)
        os << rec_scale << "*";
      os << "x";
    }
    if (rec_bias != 0) {
      if (any) {
        if (rec_bias < 0)
          os << " - " << -rec_bias;
        else
          os << " + " << rec_bias;
      } else {
        any = true;
        os << rec_bias;
      }
    }
    if (both)
      os << ", ";
    if (nonrec_bias != 0) {
      any = true;
      os << nonrec_bias;
    }
    if (both)
      os << ")";
    if (failed) {
      if (any)
        os << " + ";
      os << "∞";
    }
    if (!any)
      os << "0";
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const LinIneq &l) {
  l.print(os);
  return os;
}

/// Compute the canonicalized linear inequality expression starting at `expr`,
/// for the `var` as the left hand side `x` of the inequality. `seenVars` is
/// used as a recursion breaker. Occurrences of `var` itself within the
/// expression are mapped to the `a` coefficient in the inequality. Any other
/// variables are substituted and, in the presence of a recursion in a variable
/// other than `var`, treated as zero. `info` is a mapping from constraint
/// expressions to values and operations that produced the expression, and is
/// used during error reporting. If `reportInto` is present, the function will
/// additionally attach unsatisfiable inequalities as notes to the diagnostic as
/// it encounters them.
static LinIneq checkCycles(VarExpr *var, Expr *expr,
                           SmallPtrSetImpl<Expr *> &seenVars,
                           const ConstraintSolver::ContextInfo &info,
                           InFlightDiagnostic *reportInto = nullptr,
                           unsigned indent = 1) {
  auto ineq =
      TypeSwitch<Expr *, LinIneq>(expr)
          .Case<NilExpr>([](auto) { return LinIneq(0); })
          .Case<KnownExpr>([&](auto *expr) { return LinIneq(*expr->solution); })
          .Case<VarExpr>([&](auto *expr) {
            if (expr == var)
              return LinIneq(1, 0); // x >= 1*x + 0
            if (!seenVars.insert(expr).second)
              // Count recursions in other variables as 0. This is sane
              // since the cycle is either breakable, in which case the
              // recursion does not modify the resulting value of the
              // variable, or it is not breakable and will be caught by
              // this very function once it is called on that variable.
              return LinIneq(0);
            if (!expr->constraint)
              // Count unconstrained variables as `x >= 0`.
              return LinIneq(0);
            auto l = checkCycles(var, expr->constraint, seenVars, info,
                                 reportInto, indent + 1);
            seenVars.erase(expr);
            return l;
          })
          .Case<PowExpr>([&](auto *expr) {
            // If we can evaluate `2**arg` to a sensible constant, do
            // so. This is the case if a == 0 and if c <= 32 such that 2**c is
            // representable.
            auto arg = checkCycles(var, expr->arg, seenVars, info, reportInto,
                                   indent + 1);
            if (arg.rec_scale != 0 || arg.nonrec_bias < 0 ||
                arg.nonrec_bias >= 32)
              return LinIneq::unsat();
            return LinIneq(1 << arg.nonrec_bias); // x >= 2**arg
          })
          .Case<AddExpr>([&](auto *expr) {
            return LinIneq::add(checkCycles(var, expr->lhs(), seenVars, info,
                                            reportInto, indent + 1),
                                checkCycles(var, expr->rhs(), seenVars, info,
                                            reportInto, indent + 1));
          })
          .Case<MaxExpr, MinExpr>([&](auto *expr) {
            // Combine the inequalities of the LHS and RHS into a single overly
            // pessimistic inequality. We treat `MinExpr` the same as `MaxExpr`,
            // since `max(a,b)` is an upper bound to `min(a,b)`.
            return LinIneq::max(checkCycles(var, expr->lhs(), seenVars, info,
                                            reportInto, indent + 1),
                                checkCycles(var, expr->rhs(), seenVars, info,
                                            reportInto, indent + 1));
          })
          .Default([](auto) { return LinIneq::unsat(); });

  // If we were passed an in-flight diagnostic and the current inequality is
  // unsatisfiable, attach notes to the diagnostic indicating the values or
  // operations that contributed to this part of the constraint expression.
  if (reportInto && !ineq.sat()) {
    auto it = info.find(expr);
    if (it != info.end())
      for (auto value : it->second) {
        auto &note = reportInto->attachNote(value.getLoc());
        note << "constrained width W >= ";
        if (ineq.rec_scale == -1)
          note << "-";
        if (ineq.rec_scale != 1)
          note << ineq.rec_scale;
        note << "W";
        if (ineq.rec_bias < 0)
          note << "-" << -ineq.rec_bias;
        if (ineq.rec_bias > 0)
          note << "+" << ineq.rec_bias;
        note << " here:";
      }
  }
  if (!reportInto)
    LLVM_DEBUG(llvm::dbgs().indent(indent * 2)
               << "- Visited " << *expr << ": " << ineq << "\n");

  return ineq;
}

using ExprSolution = std::pair<Optional<int32_t>, bool>;

static ExprSolution
computeUnary(ExprSolution arg, llvm::function_ref<int32_t(int32_t)> operation) {
  if (arg.first)
    arg.first = operation(*arg.first);
  return arg;
}

static ExprSolution
computeBinary(ExprSolution lhs, ExprSolution rhs,
              llvm::function_ref<int32_t(int32_t, int32_t)> operation) {
  auto result = ExprSolution{llvm::None, lhs.second || rhs.second};
  if (lhs.first && rhs.first)
    result.first = operation(*lhs.first, *rhs.first);
  else if (lhs.first)
    result.first = lhs.first;
  else if (rhs.first)
    result.first = rhs.first;
  return result;
}

/// Compute the value of a constraint expression`expr`. `seenVars` is used as a
/// recursion breaker. Recursive variables are treated as zero. Returns the
/// computed value and a boolean indicating whether a recursion was detected.
/// This may be used to memoize the result of expressions in case they were not
/// involved in a cycle (which may alter their value from the perspective of a
/// variable).
static ExprSolution solveExpr(Expr *expr, SmallPtrSetImpl<Expr *> &seenVars,
                              unsigned indent = 1) {
  // See if we have a memoized result we can return.
  bool isTrivial = isa<NilExpr, KnownExpr>(expr);
  if (expr->solution) {
    LLVM_DEBUG({
      if (!isTrivial)
        llvm::dbgs().indent(indent * 2)
            << "- Cached " << *expr << " = " << *expr->solution << "\n";
    });
    return {*expr->solution, false};
  }

  // Otherwise compute the value of the expression.
  LLVM_DEBUG({
    if (!isTrivial)
      llvm::dbgs().indent(indent * 2) << "- Solving " << *expr << "\n";
  });
  auto solution =
      TypeSwitch<Expr *, ExprSolution>(expr)
          .Case<NilExpr>([](auto) {
            // TODO: Maybe this can be an assert. Technically no expression for
            // a variable should contain a `nil`.
            return ExprSolution{llvm::None, false};
          })
          .Case<KnownExpr>([&](auto *expr) {
            return ExprSolution{*expr->solution, false};
          })
          .Case<VarExpr>([&](auto *expr) {
            // Count recursions in variables as 0. This is sane since the cycle
            // is breakable and therefore the recursion does not modify the
            // resulting value of the variable.
            if (!seenVars.insert(expr).second)
              return ExprSolution{llvm::None, true};
            // Set unconstrained variables to 0.
            if (!expr->constraint)
              return ExprSolution{0, false};
            auto solution = solveExpr(expr->constraint, seenVars, indent + 1);
            seenVars.erase(expr);
            // Constrain variables >= 0.
            if (solution.first && *solution.first < 0)
              solution.first = 0;
            return solution;
          })
          .Case<PowExpr>([&](auto *expr) {
            auto arg = solveExpr(expr->arg, seenVars, indent + 1);
            return computeUnary(arg, [](int32_t arg) { return 1 << arg; });
          })
          .Case<AddExpr>([&](auto *expr) {
            auto lhs = solveExpr(expr->lhs(), seenVars, indent + 1);
            auto rhs = solveExpr(expr->rhs(), seenVars, indent + 1);
            return computeBinary(
                lhs, rhs, [](int32_t lhs, int32_t rhs) { return lhs + rhs; });
          })
          .Case<MaxExpr>([&](auto *expr) {
            auto lhs = solveExpr(expr->lhs(), seenVars, indent + 1);
            auto rhs = solveExpr(expr->rhs(), seenVars, indent + 1);
            return computeBinary(lhs, rhs, [](int32_t lhs, int32_t rhs) {
              return std::max(lhs, rhs);
            });
          })
          .Case<MinExpr>([&](auto *expr) {
            auto lhs = solveExpr(expr->lhs(), seenVars, indent + 1);
            auto rhs = solveExpr(expr->rhs(), seenVars, indent + 1);
            return computeBinary(lhs, rhs, [](int32_t lhs, int32_t rhs) {
              return std::min(lhs, rhs);
            });
          })
          .Default([](auto) {
            return ExprSolution{llvm::None, false};
          });

  // Memoize the result.
  if (solution.first && !solution.second)
    expr->solution = *solution.first;

  // Produce some useful debug prints.
  LLVM_DEBUG({
    if (!isTrivial) {
      if (solution.first)
        llvm::dbgs().indent(indent * 2)
            << "= Solved " << *expr << " = " << *solution.first;
      else
        llvm::dbgs().indent(indent * 2) << "= Skipped " << *expr;
      llvm::dbgs() << " (" << (solution.second ? "cycle broken" : "unique")
                   << ")\n";
    }
  });

  return solution;
}

/// Solve the constraint problem. This is a very simple implementation that
/// does not fully solve the problem if there are weird dependency cycles
/// present.
LogicalResult ConstraintSolver::solve() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n===----- Constraints -----===\n\n";
    dumpConstraints(llvm::dbgs());
  });

  // Ensure that there are no adverse cycles around.
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Checking for unbreakable loops -----===\n\n");
  SmallPtrSet<Expr *, 16> seenVars;
  bool anyFailed = false;

  for (auto *expr : exprs) {
    // Only work on variables.
    auto *var = dyn_cast<VarExpr>(expr);
    if (!var || !var->constraint)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "- Checking " << *var << " >= " << *var->constraint << "\n");

    // Canonicalize the variable's constraint expression into a form that allows
    // us to easily determine if any recursion leads to an unsatisfiable
    // constraint. The `seenVars` set acts as a recursion breaker.
    seenVars.insert(var);
    auto ineq = checkCycles(var, var->constraint, seenVars, info);
    seenVars.clear();

    // If the constraint is satisfiable, we're done.
    // TODO: It's possible that this result is already sufficient to arrive at a
    // solution for the constraint, and the second pass further down is not
    // necessary. This would require more proper handling of `MinExpr` in the
    // cycle checking code.
    if (ineq.sat()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  = Breakable since " << ineq << " satisfiable\n");
      continue;
    }

    // If we arrive here, the constraint is not satisfiable at all. To provide
    // some guidance to the user, we call the cycle checking code again, but
    // this time with an in-flight diagnostic to attach notes indicating
    // unsatisfiable paths in the cycle.
    LLVM_DEBUG(llvm::dbgs()
               << "  = UNBREAKABLE since " << ineq << " unsatisfiable\n");
    anyFailed = true;
    for (auto value : info.find(var)->second) {
      // Depending on whether this value stems from an operation or not, create
      // an appropriate diagnostic identifying the value.
      auto op = value.getDefiningOp();
      auto diag =
          op ? op->emitOpError() : mlir::emitError(value.getLoc()) << "value ";
      diag << "is constrained to be wider than itself";

      // Re-run the cycle checking, but this time reporting into the diagnostic.
      seenVars.insert(var);
      checkCycles(var, var->constraint, seenVars, info, &diag);
      seenVars.clear();
    }
  }

  // Iterate over the constraint variables and solve each.
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Solving constraints -----===\n\n");
  for (auto *expr : exprs) {
    // Only work on variables.
    auto *var = dyn_cast<VarExpr>(expr);
    if (!var || !var->constraint)
      continue;

    // Compute the value for the variable.
    LLVM_DEBUG(llvm::dbgs()
               << "- Solving " << *var << " >= " << *var->constraint << "\n");
    seenVars.insert(var);
    auto solution = solveExpr(var->constraint, seenVars);
    seenVars.clear();

    // Constrain variables >= 0.
    if (solution.first && *solution.first < 0)
      solution.first = 0;

    // If successful, store the value as the variable's solution.
    // TODO: We might want to complain about unconstrained widths here.
    LLVM_DEBUG({
      if (solution.first)
        llvm::dbgs() << "  - Setting " << *var << " = " << solution.first
                     << " (" << (solution.second ? "cycle broken" : "unique")
                     << ")\n";
      else
        llvm::dbgs() << "  - Leaving " << *var << " unsolved\n";
    });
    if (solution.first)
      var->solution = *solution.first;
  }

  return failure(anyFailed);
}

//===----------------------------------------------------------------------===//
// Inference Constraint Problem Mapping
//===----------------------------------------------------------------------===//

namespace {

/// A helper class which maps the types and operations in a design to a set of
/// variables and constraints to be solved later.
class InferenceMapping {
public:
  InferenceMapping(ConstraintSolver &solver) : solver(solver) {}

  LogicalResult map(CircuitOp op);
  LogicalResult mapOperation(Operation *op);

  /// Declare all the variables in the value. If the value is a ground type,
  /// there is a single variable declared.  If the value is an aggregate type,
  /// it sets up variables for each unknown width.
  void declareVars(Value value, Location loc);

  /// Declare a variable associated with a specific field of an aggregate.
  Expr *declareVar(FieldRef fieldRef, Location loc);

  /// Declarate a variable for a type with an unknown width.  The type must be a
  /// non-aggregate.
  Expr *declareVar(FIRRTLType type, Location loc);

  /// Constrain the value "larger" to be greater than or equal to "smaller".
  /// These may be aggregate values. This is used for regular connects.
  void constrainTypes(Value larger, Value smaller);

  /// Constrain the value "larger" to be greater than or equal to "smaller".
  /// These may be aggregate values. This is used for partial connects.
  void partiallyConstrainTypes(Value larger, Value smaller);

  /// Constrain the expression "larger" to be greater than or equals to
  /// the expression "smaller".
  void constrainTypes(Expr *larger, Expr *smaller);

  /// Assign the constraint expressions of the fields in the `src` argument as
  /// the expressions for the `dst` argument. Both fields must be of the given
  /// `type`.
  void unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type);

  /// Get the expr associated with the value.  The value must be a non-aggregate
  /// type.
  Expr *getExpr(Value value);

  /// Get the expr associated with a specific field in a value.
  Expr *getExpr(FieldRef fieldRef);

  /// Get the expr associated with the value. If value is NULL, then this
  /// returns NULL. The value must be a non-aggregate type.
  Expr *getExprOrNull(Value value);

  /// Get the expr associated with a specific field in a value. If value is
  /// NULL, then this returns NULL.
  Expr *getExprOrNull(FieldRef fieldRef);

  /// Set the expr associated with the value. The value must be a non-aggregate
  /// type.
  void setExpr(Value value, Expr *expr);

  /// Set the expr associated with a specific field in a value.
  void setExpr(FieldRef fieldRef, Expr *expr);

  /// Return whether a module was skipped due to being fully inferred already.
  bool isModuleSkipped(ModuleOp module) { return skippedModules.count(module); }

  /// Return whether all modules in the mapping were fully inferred.
  bool areAllModulesSkipped() { return allModulesSkipped; }

private:
  /// The constraint solver into which we emit variables and constraints.
  ConstraintSolver &solver;

  /// The constraint exprs for each result type of an operation.
  // TODO: This should actually not map to `Expr *` directly, but rather a
  // view class that can represent aggregate exprs for bundles/arrays as well.
  DenseMap<FieldRef, Expr *> opExprs;

  /// The fully inferred modules that were skipped entirely.
  SmallPtrSet<Operation *, 16> skippedModules;
  bool allModulesSkipped = true;
};

} // namespace

/// Check if a type contains any FIRRTL type with uninferred widths.
static bool hasUninferredWidth(Type type) {
  if (auto ftype = type.dyn_cast<FIRRTLType>())
    return ftype.hasUninferredWidth();
  return false;
}

LogicalResult InferenceMapping::map(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs()
             << "\n===----- Mapping ops to constraint exprs -----===\n\n");

  // Ensure we have constraint variables established for all module ports.
  op.walk<WalkOrder::PostOrder>([&](FModuleOp module) {
    for (auto arg : module.getArguments()) {
      solver.setCurrentContextInfo(arg);
      declareVars(arg, module.getLoc());
    }
    return WalkResult::skip(); // no need to look inside the module
  });

  // Go through the module bodies and populate the constraint problem.
  auto result = op.walk<WalkOrder::PostOrder>([&](FModuleOp module) {
    // Check if the module contains *any* uninferred widths. This allows us to
    // do an early skip if the module is already fully inferred.
    bool anyUninferred = false;
    for (auto arg : module.getArguments()) {
      anyUninferred |= hasUninferredWidth(arg.getType());
      if (anyUninferred)
        break;
    }
    module.walk([&](Operation *op) {
      for (auto type : op->getResultTypes())
        anyUninferred |= hasUninferredWidth(type);
      if (anyUninferred)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (!anyUninferred) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping fully-inferred module '"
                              << module.getName() << "'\n");
      skippedModules.insert(module);
      return WalkResult::skip();
    }
    allModulesSkipped = false;

    // Go through operations in the module, creating type variables for results,
    // and generating constraints.
    auto result = module.getBody().walk(
        [&](Operation *op) { return WalkResult(mapOperation(op)); });
    if (result.wasInterrupted())
      return WalkResult::interrupt();
    return WalkResult::skip(); // walk above already visited module body
  });
  return failure(result.wasInterrupted());
}

LogicalResult InferenceMapping::mapOperation(Operation *op) {
  // In case the operation result has a type without uninferred widths, don't
  // even bother to populate the constraint problem and treat that as a known
  // size directly. This is done in `declareVars`, which will generate
  // `KnownExpr` nodes for all known widths -- which are the only ones in this
  // case.
  bool allWidthsKnown = true;
  for (auto result : op->getResults()) {
    if (!hasUninferredWidth(result.getType()))
      declareVars(result, op->getLoc());
    else
      allWidthsKnown = false;
  }
  if (allWidthsKnown && !isa<ConnectOp, PartialConnectOp, AttachOp>(op))
    return success();

  // Actually generate the necessary constraint expressions.
  bool mappingFailed = false;
  solver.setCurrentContextInfo(op->getNumResults() > 0 ? op->getResults()[0]
                                                       : Value{});
  TypeSwitch<Operation *>(op)
      .Case<ConstantOp>([&](auto op) {
        // If the constant has a known width, use that. Otherwise pick the
        // smallest number of bits necessary to represent the constant.
        Expr *e;
        if (auto width = op.getType().getWidth())
          e = solver.known(*width);
        else {
          auto v = op.value();
          auto w = v.getBitWidth() - (v.isNegative() ? v.countLeadingOnes()
                                                     : v.countLeadingZeros());
          if (v.isSigned())
            w += 1;
          e = solver.known(std::max(w, 1u));
        }
        setExpr(op.getResult(), e);
      })
      .Case<SpecialConstantOp>([&](auto op) {
        // Nothing required.
      })
      .Case<WireOp, InvalidValueOp, RegOp>(
          [&](auto op) { declareVars(op.getResult(), op.getLoc()); })
      .Case<RegResetOp>([&](auto op) {
        // The original Scala code also constrains the reset signal to be at
        // least 1 bit wide. We don't do this here since the MLIR FIRRTL
        // dialect enforces the reset signal to be an async reset or a
        // `uint<1>`.
        declareVars(op.getResult(), op.getLoc());
        // Contrain the register to be greater than or equal to the reset
        // signal.
        constrainTypes(op.getResult(), op.resetValue());
      })
      .Case<NodeOp>([&](auto op) {
        // Nodes have the same type as their input.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.input(), 0),
                   op.getType());
      })

      // Aggregate Values
      .Case<SubfieldOp>([&](auto op) {
        auto bundleType = op.input().getType().template cast<BundleType>();
        auto index = bundleType.getElementIndex(op.fieldname()).getValue();
        auto fieldID = bundleType.getFieldID(index);
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.input(), fieldID),
                   op.getType());
      })
      .Case<SubindexOp, SubaccessOp>([&](auto op) {
        // All vec fields unify to the same thing. Always use the first element
        // of the vector, which has a field ID of 1.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.input(), 1),
                   op.getType());
      })

      // Arithmetic and Logical Binary Primitives
      .Case<AddPrimOp, SubPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(solver.max(lhs, rhs), solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<MulPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DivPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        Expr *e;
        if (op.getType().isSigned()) {
          e = solver.add(lhs, solver.known(1));
        } else {
          e = lhs;
        }
        setExpr(op.getResult(), e);
      })
      .Case<RemPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.min(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<AndPrimOp, OrPrimOp, XorPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.max(lhs, rhs);
        setExpr(op.getResult(), e);
      })

      // Misc Binary Primitives
      .Case<CatPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DShlPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, solver.add(solver.pow(rhs), solver.known(-1)));
        setExpr(op.getResult(), e);
      })
      .Case<DShlwPrimOp, DShrPrimOp>([&](auto op) {
        auto e = getExpr(op.lhs());
        setExpr(op.getResult(), e);
      })

      // Unary operators
      .Case<NegPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<CvtPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = op.input().getType().template cast<IntType>().isSigned()
                     ? input
                     : solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Miscellaneous
      .Case<BitsPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.hi() - op.lo() + 1));
      })
      .Case<HeadPrimOp>(
          [&](auto op) { setExpr(op.getResult(), solver.known(op.amount())); })
      .Case<TailPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(-op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<PadPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.max(input, solver.known(op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShlPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShrPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.max(solver.add(input, solver.known(-op.amount())),
                            solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Handle operations whose output width matches the input width.
      .Case<NotPrimOp, AsSIntPrimOp, AsUIntPrimOp, AsPassivePrimOp,
            AsNonPassivePrimOp>(
          [&](auto op) { setExpr(op.getResult(), getExpr(op.input())); })

      // Handle operations with a single result type that always has a
      // well-known width.
      .Case<LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            AsClockPrimOp, AsAsyncResetPrimOp, AndRPrimOp, OrRPrimOp,
            XorRPrimOp>([&](auto op) {
        auto width = op.getType().getBitWidthOrSentinel();
        assert(width > 0 && "width should have been checked by verifier");
        setExpr(op.getResult(), solver.known(width));
      })

      .Case<MuxPrimOp>([&](auto op) {
        auto sel = getExpr(op.sel());
        constrainTypes(sel, solver.known(1));
        auto high = getExpr(op.high());
        auto low = getExpr(op.low());
        auto e = solver.max(high, low);
        setExpr(op.getResult(), e);
      })

      // Handle the various connect statements that imply a type constraint.
      .Case<ConnectOp>([&](auto op) { constrainTypes(op.dest(), op.src()); })
      .Case<PartialConnectOp>(
          [&](auto op) { partiallyConstrainTypes(op.dest(), op.src()); })
      .Case<AttachOp>([&](auto op) {
        // Attach connects multiple analog signals together. All signals must
        // have the same bit width. Signals without bit width inherit from the
        // other signals.
        if (op.operands().empty())
          return;
        auto prev = op.operands()[0];
        for (auto operand : op.operands().drop_front()) {
          auto e1 = getExpr(prev);
          auto e2 = getExpr(operand);
          constrainTypes(e1, e2);
          constrainTypes(e2, e1);
          prev = operand;
        }
      })

      // Handle the no-ops that don't interact with width inference.
      .Case<PrintFOp, SkipOp, StopOp, WhenOp, AssertOp, AssumeOp, CoverOp>(
          [&](auto) {})

      // Handle instances of other modules.
      .Case<InstanceOp>([&](auto op) {
        auto refdModule = op.getReferencedModule();
        auto module = dyn_cast<FModuleOp>(refdModule);
        if (!module) {
          auto diag = mlir::emitError(op.getLoc());
          diag << "extern module `" << op.moduleName()
               << "` has ports of uninferred width";
          diag.attachNote(op.getLoc())
              << "Only non-extern FIRRTL modules may contain unspecified "
                 "widths to be inferred automatically.";
          diag.attachNote(refdModule->getLoc())
              << "Module `" << op.moduleName() << "` defined here:";
          mappingFailed = true;
          return;
        }
        // Simply look up the free variables created for the instantiated
        // module's ports, and use them for instance port wires. This way,
        // constraints imposed onto the ports of the instance will transparently
        // apply to the ports of the instantiated module.
        for (auto it : llvm::zip(op->getResults(), module.getArguments())) {
          unifyTypes(FieldRef(std::get<0>(it), 0), FieldRef(std::get<1>(it), 0),
                     std::get<0>(it).getType().template cast<FIRRTLType>());
        }
      })

      // Handle memories.
      .Case<MemOp>([&](auto op) {
        // Create constraint variables for all ports.
        for (auto result : op.results())
          declareVars(result, op.getLoc());

        // A helper function that returns the indeces of the "data", "rdata",
        // and "wdata" fields in the bundle corresponding to a memory port.
        auto dataFieldIndices = [](MemOp::PortKind kind) {
          static const unsigned indices[] = {3, 4, 5};
          switch (kind) {
          case MemOp::PortKind::Read:
          case MemOp::PortKind::Write:
            return ArrayRef<unsigned>(indices, 1); // {3}
          case MemOp::PortKind::ReadWrite:
            return ArrayRef<unsigned>(indices + 1, 2); // {4, 5}
          }
        };

        // This creates independent variables for every data port. Yet, what we
        // actually want is for all data ports to share the same variable. To do
        // this, we find the first data port declared, and use that port's vars
        // for all the other ports.
        unsigned firstFieldIndex = dataFieldIndices(op.getPortKind(0))[0];
        FieldRef firstData(op.getResult(0), op.getPortType(0)
                                                .getPassiveType()
                                                .template cast<BundleType>()
                                                .getFieldID(firstFieldIndex));
        LLVM_DEBUG(llvm::dbgs() << "Adjusting memory port variables:\n");

        // Reuse data port variables.
        auto dataType = op.getDataType();
        for (unsigned i = 0, e = op.results().size(); i < e; ++i) {
          auto result = op.getResult(i);
          auto portType =
              op.getPortType(i).getPassiveType().template cast<BundleType>();
          for (auto fieldIndex : dataFieldIndices(op.getPortKind(i)))
            unifyTypes(FieldRef(result, portType.getFieldID(fieldIndex)),
                       firstData, dataType);
        }
      })

      .Default([&](auto op) {
        op->emitOpError("not supported in width inference");
        mappingFailed = true;
      });

  return failure(mappingFailed);
}

/// Declare free variables for the type of a value, and associate the resulting
/// set of variables with that value.
void InferenceMapping::declareVars(Value value, Location loc) {
  auto ftype = value.getType().dyn_cast<FIRRTLType>();

  // Unknown types are set to nil.
  if (!ftype)
    setExpr(FieldRef(value, 0), solver.nil());

  // Declare a variable for every unknown width in the type. If this is a Bundle
  // type or a FVector type, we will have to potentially create many variables.
  unsigned fieldID = 0;
  std::function<void(FIRRTLType)> declare = [&](FIRRTLType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integer create a known expression.
      setExpr(FieldRef(value, fieldID), solver.known(width));
      fieldID++;
    } else if (width == -1) {
      // Unkown width integers create a variable.
      setExpr(FieldRef(value, fieldID), solver.var());
      fieldID++;
    } else if (auto bundleType = type.dyn_cast<BundleType>()) {
      // Bundle types recursively declare all bundle elements.
      fieldID++;
      for (auto &element : bundleType.getElements()) {
        declare(element.type);
      }
    } else if (auto vecType = type.dyn_cast<FVectorType>()) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0) {
        declare(vecType.getElementType());
      }
      // Skip past the rest of the elements
      fieldID = save + vecType.getMaxFieldID();
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  declare(ftype);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type. Types have to be
/// compatible in the sense that they may only differ in the presence or absence
/// of bit widths.
///
/// This function is used to apply regular connects.
void InferenceMapping::constrainTypes(Value larger, Value smaller) {
  // Recurse to every leaf element and set larger >= smaller.
  auto type = larger.getType().cast<FIRRTLType>();
  auto fieldID = 0;
  std::function<void(FIRRTLType, Value, Value)> constrain =
      [&](FIRRTLType type, Value larger, Value smaller) {
        if (auto bundleType = type.dyn_cast<BundleType>()) {
          fieldID++;
          for (auto &element : bundleType.getElements()) {
            if (element.isFlip)
              constrain(element.type, smaller, larger);
            else
              constrain(element.type, larger, smaller);
          }
        } else if (auto vecType = type.dyn_cast<FVectorType>()) {
          fieldID++;
          auto save = fieldID;
          // Skip 0 length vectors.
          if (vecType.getNumElements() > 0) {
            constrain(vecType.getElementType(), larger, smaller);
          }
          fieldID = save + vecType.getMaxFieldID();
        } else if (type.isGround()) {
          // Leaf element, look up their expressions, and create the constraint.
          constrainTypes(getExpr(FieldRef(larger, fieldID)),
                         getExpr(FieldRef(smaller, fieldID)));
          fieldID++;
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };

  constrain(type, larger, smaller);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type. The types do not have to
/// be identical, but they must be similar in the sense that corresponding
/// fields must have the same kind (scalars, bundles, vectors).
///
/// This function is used to apply partial connects.
void InferenceMapping::partiallyConstrainTypes(Value larger, Value smaller) {
  // Recurse to every leaf element and set larger >= smaller.
  std::function<void(FIRRTLType, Value, unsigned, FIRRTLType, Value, unsigned)>
      constrain = [&](FIRRTLType aType, Value a, unsigned aID, FIRRTLType bType,
                      Value b, unsigned bID) {
        if (auto aBundle = aType.dyn_cast<BundleType>()) {
          auto bBundle = bType.cast<BundleType>();
          for (unsigned aIndex = 0, e = aBundle.getNumElements(); aIndex < e;
               ++aIndex) {
            auto aField = aBundle.getElements()[aIndex].name.getValue();
            auto bIndex = bBundle.getElementIndex(aField);
            if (!bIndex)
              continue;
            auto &aElt = aBundle.getElements()[aIndex];
            auto &bElt = bBundle.getElements()[*bIndex];
            if (aElt.isFlip)
              constrain(bElt.type, b, bID + bBundle.getFieldID(*bIndex),
                        aElt.type, a, aID + aBundle.getFieldID(aIndex));
            else
              constrain(aElt.type, a, aID + aBundle.getFieldID(aIndex),
                        bElt.type, b, bID + bBundle.getFieldID(*bIndex));
          }
        } else if (auto aVecType = aType.dyn_cast<FVectorType>()) {
          // Do not constrain the elements of a zero length vector.
          if (aVecType.getNumElements() == 0)
            return;
          auto bVecType = bType.cast<FVectorType>();
          constrain(aVecType.getElementType(), a, aID + 1,
                    bVecType.getElementType(), b, bID + 1);
        } else if (aType.isGround()) {
          // Leaf element, look up their expressions, and create the constraint.
          constrainTypes(getExpr(FieldRef(a, aID)), getExpr(FieldRef(b, bID)));
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };

  auto largerType = larger.getType().cast<FIRRTLType>();
  auto smallerType = smaller.getType().cast<FIRRTLType>();
  constrain(largerType, larger, 0, smallerType, smaller, 0);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type.
void InferenceMapping::constrainTypes(Expr *larger, Expr *smaller) {
  assert(larger && "Larger expression should be specified");
  assert(smaller && "Smaller expression should be specified");
  // Mimic the Scala implementation here by simply doing nothing if the larger
  // expr is not a free variable. Apparently there are many cases where
  // useless constraints can be added, e.g. on multiple well-known values. As
  // long as we don't want to do type checking itself here, but only width
  // inference, we should be fine ignoring expr we cannot constraint anyway.
  if (auto largerVar = dyn_cast<VarExpr>(larger)) {
    LLVM_ATTRIBUTE_UNUSED auto c = solver.addGeqConstraint(largerVar, smaller);
    LLVM_DEBUG(llvm::dbgs()
               << "Constrained " << *largerVar << " >= " << *c << "\n");
  }
}

/// Assign the constraint expressions of the fields in the `src` argument as the
/// expressions for the `dst` argument. Both fields must be of the given `type`.
void InferenceMapping::unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type) {
  // Fast path for `unifyTypes(x, x, _)`.
  if (lhs == rhs)
    return;

  // Co-iterate the two field refs, recurring into every leaf element and set
  // them equal.
  auto fieldID = 0;
  std::function<void(FIRRTLType)> unify = [&](FIRRTLType type) {
    if (type.isGround()) {
      // Leaf element, unify the fields!
      FieldRef lhsFieldRef(lhs.getValue(), lhs.getFieldID() + fieldID);
      FieldRef rhsFieldRef(rhs.getValue(), rhs.getFieldID() + fieldID);
      LLVM_DEBUG(llvm::dbgs() << "Unify " << getFieldName(lhsFieldRef) << " = "
                              << getFieldName(rhsFieldRef) << "\n");
      setExpr(lhsFieldRef, getExpr(rhsFieldRef));
      fieldID++;
    } else if (auto bundleType = type.dyn_cast<BundleType>()) {
      fieldID++;
      for (auto &element : bundleType.getElements()) {
        unify(element.type);
      }
    } else if (auto vecType = type.dyn_cast<FVectorType>()) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0) {
        unify(vecType.getElementType());
      }
      fieldID = save + vecType.getMaxFieldID();
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  unify(type);
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(Value value) {
  assert(value.getType().cast<FIRRTLType>().isGround());
  // A field ID of 0 indicates the entire value.
  return getExpr(FieldRef(value, 0));
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(FieldRef fieldRef) {
  auto expr = getExprOrNull(fieldRef);
  assert(expr && "constraint expr should have been constructed for value");
  return expr;
}

/// Get the constraint expression for a value, or null if no expression exists
/// for the value.
Expr *InferenceMapping::getExprOrNull(Value value) {
  assert(value.getType().cast<FIRRTLType>().isGround());
  return getExprOrNull(FieldRef(value, 0));
}

Expr *InferenceMapping::getExprOrNull(FieldRef fieldRef) {
  auto it = opExprs.find(fieldRef);
  return it != opExprs.end() ? it->second : nullptr;
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(Value value, Expr *expr) {
  assert(value.getType().cast<FIRRTLType>().isGround());
  // A field ID of 0 indicates the entire value.
  setExpr(FieldRef(value, 0), expr);
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(FieldRef fieldRef, Expr *expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "Expr " << *expr << " for " << fieldRef.getValue();
    if (fieldRef.getFieldID())
      llvm::dbgs() << " '" << getFieldName(fieldRef) << "'";
    llvm::dbgs() << "\n";
  });
  opExprs[fieldRef] = expr;
}

//===----------------------------------------------------------------------===//
// Inference Result Application
//===----------------------------------------------------------------------===//

namespace {
/// A helper class which maps the types and operations in a design to a set
/// of variables and constraints to be solved later.
class InferenceTypeUpdate {
public:
  InferenceTypeUpdate(InferenceMapping &mapping) : mapping(mapping) {}

  LogicalResult update(CircuitOp op);
  bool updateOperation(Operation *op);
  bool updateValue(Value value);
  FIRRTLType updateType(FieldRef fieldRef, FIRRTLType type);

private:
  bool anyFailed;
  InferenceMapping &mapping;
};

} // namespace

/// Update the types throughout a circuit.
LogicalResult InferenceTypeUpdate::update(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Update types -----===\n\n");
  anyFailed = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // Skip this module if it had no widths to be inferred at all.
    if (auto module = dyn_cast<ModuleOp>(op))
      if (mapping.isModuleSkipped(module))
        return WalkResult::skip();

    updateOperation(op);
    return WalkResult(failure(anyFailed));
  });
  return failure(anyFailed);
}

/// Update the result types of an operation.
bool InferenceTypeUpdate::updateOperation(Operation *op) {
  bool anyChanged = false;
  for (Value v : op->getResults()) {
    anyChanged |= updateValue(v);
    if (anyFailed)
      return false;
  }

  // If this is a connect operation, width inference might have inferred a RHS
  // that is wider than the LHS, in which case an additional BitsPrimOp is
  // necessary to truncate the value.
  if (auto con = dyn_cast<ConnectOp>(op)) {
    auto lhs = con.dest().getType().cast<FIRRTLType>();
    auto rhs = con.src().getType().cast<FIRRTLType>();
    auto lhsWidth = lhs.getBitWidthOrSentinel();
    auto rhsWidth = rhs.getBitWidthOrSentinel();
    if (lhsWidth >= 0 && rhsWidth >= 0 && lhsWidth < rhsWidth) {
      OpBuilder builder(op);
      auto trunc = builder.createOrFold<TailPrimOp>(con.getLoc(), con.src(),
                                                    rhsWidth - lhsWidth);
      if (rhs.isa<SIntType>())
        trunc = builder.createOrFold<AsSIntPrimOp>(con.getLoc(), lhs, trunc);

      LLVM_DEBUG(llvm::dbgs()
                 << "Truncating RHS to " << lhs << " in " << con << "\n");
      con->replaceUsesOfWith(con.src(), trunc);
    }
  }

  // If this is a module, update its ports.
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Update the block argument types.
    bool argsChanged = false;
    std::vector<Type> argTypes;
    argTypes.reserve(module.getArguments().size());
    for (auto arg : module.getArguments()) {
      argsChanged |= updateValue(arg);
      argTypes.push_back(arg.getType());
      if (anyFailed)
        return false;
    }

    // Update the module function type if needed.
    if (argsChanged) {
      auto type =
          FunctionType::get(op->getContext(), argTypes, /*resultTypes*/ {});
      module->setAttr(FModuleOp::getTypeAttrName(), TypeAttr::get(type));
      anyChanged = true;
    }
  }
  return anyChanged;
}

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLType resizeType(FIRRTLType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<UIntType>(
          [&](auto type) { return UIntType::get(context, newWidth); })
      .Case<SIntType>(
          [&](auto type) { return SIntType::get(context, newWidth); })
      .Case<AnalogType>(
          [&](auto type) { return AnalogType::get(context, newWidth); })
      .Default([&](auto type) { return type; });
}

/// Update the type of a value.
bool InferenceTypeUpdate::updateValue(Value value) {
  // Check if the value has a type which we can update.
  auto type = value.getType().dyn_cast<FIRRTLType>();
  if (!type)
    return false;

  // Fast path for types that have fully inferred widths.
  if (!hasUninferredWidth(type))
    return false;

  // Ignore `InvalidValueOp`, as we generally lack the information to infer a
  // type for it. These ops tend to end up on one side of a multiplexer, and are
  // removed at a later canonicalization stage. Since they are directly used as
  // a multiplexer input, there are no constraints on their size. Inferring them
  // would involved potentially duplicating them once for each use site, and
  // then trying to infer their size from context (which we do nowhere else in
  // FIRRTL).
  if (isa_and_nonnull<InvalidValueOp>(value.getDefiningOp()))
    return false;

  // If this is an operation that does not generate any free variables that
  // are determined during width inference, simply update the value type based
  // on the operation arguments.
  if (auto op = dyn_cast_or_null<InferTypeOpInterface>(value.getDefiningOp())) {
    SmallVector<Type, 2> types;
    auto res =
        op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                            op->getAttrDictionary(), op->getRegions(), types);
    if (failed(res)) {
      anyFailed = true;
      return false;
    }
    assert(types.size() == op->getNumResults());
    for (auto it : llvm::zip(op->getResults(), types)) {
      LLVM_DEBUG(llvm::dbgs() << "Inferring " << std::get<0>(it) << " as "
                              << std::get<1>(it) << "\n");
      std::get<0>(it).setType(std::get<1>(it));
    }
    return true;
  }

  // Recreate the type, substituting the solved widths.
  auto context = type.getContext();
  unsigned fieldID = 0;
  std::function<FIRRTLType(FIRRTLType)> update = [&](FIRRTLType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integers return themselves.
      fieldID++;
      return type;
    } else if (width == -1) {
      // Unkown width integers return the solved type.
      auto newType = updateType(FieldRef(value, fieldID), type);
      fieldID++;
      return newType;
    } else if (auto bundleType = type.dyn_cast<BundleType>()) {
      // Bundle types recursively update all bundle elements.
      fieldID++;
      llvm::SmallVector<BundleType::BundleElement, 3> elements;
      for (auto &element : bundleType.getElements()) {
        elements.emplace_back(element.name, element.isFlip,
                              update(element.type));
      }
      return BundleType::get(elements, context);
    } else if (auto vecType = type.dyn_cast<FVectorType>()) {
      fieldID++;
      auto save = fieldID;
      // TODO: this should recurse into the element type of 0 length vectors and
      // set any unknown width to 0.
      if (vecType.getNumElements() > 0) {
        auto newType = FVectorType::get(update(vecType.getElementType()),
                                        vecType.getNumElements());
        fieldID = save + vecType.getMaxFieldID();
        return newType;
      }
      // If this is a 0 length vector return the original type.
      return type;
    }
    llvm_unreachable("Unknown type inside a bundle!");
  };

  // Update the type.
  auto newType = update(type);
  LLVM_DEBUG(llvm::dbgs() << "Update " << value << " to " << newType << "\n");
  value.setType(newType);

  // If this is a ConstantOp, adjust the width of the underlying APInt.
  // Unsized constants have APInts which are *at least* wide enough to hold
  // the value, but may be larger. This can trip up the verifier.
  if (auto op = value.getDefiningOp<ConstantOp>()) {
    auto k = op.value();
    auto bitwidth = op.getType().cast<FIRRTLType>().getBitWidthOrSentinel();
    if (k.getBitWidth() > unsigned(bitwidth))
      k = k.trunc(bitwidth);
    op->setAttr("value", IntegerAttr::get(op.getContext(), k));
  }

  return newType != type;
}

/// Update a type.
FIRRTLType InferenceTypeUpdate::updateType(FieldRef fieldRef, FIRRTLType type) {
  assert(type.isGround() && "Can only pass in ground types.");
  auto value = fieldRef.getValue();
  // Get the inferred width.
  Expr *expr = mapping.getExprOrNull(fieldRef);
  if (!expr || !expr->solution.hasValue()) {
    anyFailed = true;
    // Emit an error that indicates where we have failed to infer a width.
    // Note that we only get here if the IR contained some operation or FIRRTL
    // type that is not yet supported by width inference. Errors related to
    // widths not being inferrable due to contradictory constraints are
    // handled earlier in the solver, and the pass never proceeds to perform
    // this type update. TL;DR: This is for compiler hackers.
    // TODO: Convert this to an assertion once we support all operations and
    // types for width inference.
    auto diag = mlir::emitError(value.getLoc(), "failed to infer width");
    auto fieldName = getFieldName(fieldRef);
    if (!fieldName.empty())
      diag << " for '" << fieldName << "'";
    else if (auto blockArg = value.dyn_cast<BlockArgument>())
      diag << " for port #" << blockArg.getArgNumber();
    else if (auto op = value.getDefiningOp())
      diag << " for op '" << op->getName() << "'";
    else
      diag << " for value";
    diag << " of type " << type;
    // Return the original unsolved type.
    return type;
  }
  int32_t solution = expr->solution.getValue();
  assert(solution >= 0); // The solver infers variables to be 0 or greater.
  return resizeType(type, solution);
}

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

// Hash slots in the interned allocator as if they were the pointed-to value
// itself.
namespace llvm {
template <typename T>
struct DenseMapInfo<InternedSlot<T>> {
  using Slot = InternedSlot<T>;
  static Slot getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return Slot(static_cast<T *>(pointer));
  }
  static Slot getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return Slot(static_cast<T *>(pointer));
  }
  static unsigned getHashValue(Slot val) { return mlir::hash_value(*val.ptr); }
  static bool isEqual(Slot LHS, Slot RHS) {
    auto empty = getEmptyKey().ptr;
    auto tombstone = getTombstoneKey().ptr;
    if (LHS.ptr == empty || RHS.ptr == empty || LHS.ptr == tombstone ||
        RHS.ptr == tombstone)
      return LHS.ptr == RHS.ptr;
    return *LHS.ptr == *RHS.ptr;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsPass : public InferWidthsBase<InferWidthsPass> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsPass::runOnOperation() {
  // Collect variables and constraints
  ConstraintSolver solver;
  InferenceMapping mapping(solver);
  if (failed(mapping.map(getOperation()))) {
    signalPassFailure();
    return;
  }
  if (mapping.areAllModulesSkipped()) {
    markAllAnalysesPreserved();
    return; // fast path if no inferrable widths are around
  }

  // Solve the constraints.
  if (failed(solver.solve())) {
    signalPassFailure();
    return;
  }

  // Update the types with the inferred widths.
  if (failed(InferenceTypeUpdate(mapping).update(getOperation())))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsPass() {
  return std::make_unique<InferWidthsPass>();
}
