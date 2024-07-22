#ifndef CIRCT_HWL_HWLAST_H
#define CIRCT_HWL_HWLAST_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <string>

namespace circt {
namespace hwl {

template <typename T>
struct ValPtr {
  static_assert(!std::is_same_v<T, void>);
  ValPtr(T *ptr) : ptr(ptr) {}
  T &operator&() const { return *ptr; }
  T *operator->() const { return ptr; }
  T *get() const { return ptr; }
  operator T *() const { return get(); }
  bool operator==(const ValPtr &other) const { return *ptr == *other; }
  bool operator!=(const ValPtr &other) const { return !(*this == other); }
  friend llvm::hash_code hash_value(const ValPtr &ptr) {
    return llvm::hash_value(*ptr);
  }

private:
  T *ptr;
};

///
/// Types
///

struct Type {
  enum class Kind { TConst, TArrow };

  Type() = delete;
  Type(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

private:
  Kind kind;
};

struct TConst : public Type {
  static Kind kind() { return Kind::TConst; }
  TConst(const std::string &value) : Type(kind()), value(value) {}
  static bool classof(const Type *type) { return type->getKind() == kind(); }
  const std::string &getValue() const { return value; }

private:
  std::string value;
};

struct TArrow : public Type {
  static Kind kind() { return Kind::TArrow; }
  TArrow(Type *left, Type *right) : Type(kind()), left(left), right(right) {}
  static bool classof(const Type *type) { return type->getKind() == kind(); }
  Type *getLeft() const { return left; }
  Type *getRight() const { return right; }

private:
  Type *left;
  Type *right;
};

template <typename T>
struct PtrKey {};

///
/// Expressions
///

struct Expr {
  enum class Kind { Const, Anno, Var, Abs, App };
  Expr() = delete;
  Expr(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

private:
  Kind kind;
};

struct Const : public Expr {
  static Kind kind() { return Kind::Const; }
  Const(const std::string &value) : Expr(kind()), value(value) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  const std::string &getValue() const { return value; }

private:
  std::string value;
};

struct Anno : public Expr {
  static Kind kind() { return Kind::Anno; }
  Anno(Expr *expr, Type *type) : Expr(kind()), expr(expr), type(type) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  Expr *getExpr() const { return expr; }
  Type *getType() const { return type; }

private:
  Expr *expr;
  Type *type;
};

struct Var : public Expr {
  static Kind kind() { return Kind::Var; }
  Var(uint64_t index) : Expr(kind()), index(index) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  uint64_t getIndex() const { return index; }

private:
  uint64_t index;
};

struct Abs : public Expr {
  static Kind kind() { return Kind::Abs; }
  Abs(Expr *body) : Expr(kind()), body(body) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  Expr *getBody() const { return body; }

private:
  Expr *body;
};

struct App : public Expr {
  static Kind kind() { return Kind::App; }
  App(Expr *fun, Expr *arg) : Expr(kind()), fun(fun), arg(arg) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }

  Expr *getFun() const { return fun; }
  Expr *getArg() const { return arg; }

private:
  Expr *fun;
  Expr *arg;
};

///
/// Statements
///

struct Stmt {
  enum class Kind { Declaration, Definition };
  Stmt() = delete;
  Stmt(Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }

private:
  Kind kind;
};

struct Declaration : public Stmt {
  Declaration(StringRef name, Type *type)
      : Stmt(Kind::Declaration), name(name), type(type) {}
  const std::string &getName() const { return name; }
  Type *getType() const { return type; }

private:
  std::string name;
  Type *type;
};

struct Definition : public Stmt {
  Definition(StringRef name, Expr *value)
      : Stmt(Kind::Definition), name(name), value(value) {}

  StringRef getName() const { return name; }
  Expr *getValue() const { return value; }

private:
  StringRef name;
  Expr *value;
};

///
/// Compilation Unit
///

struct CompUnit {
  CompUnit(const std::vector<Stmt *> &statements) : statements(statements) {}
  CompUnit(std::vector<Stmt *> &&statements)
      : statements(std::move(statements)) {}
  const std::vector<Stmt *> &getStatements() const { return statements; }

private:
  std::vector<Stmt *> statements;
};

struct ASTEquivalence {

  ///
  /// Types
  ///

  static bool equal(const TConst *lhs, const TConst *rhs) {
    return lhs->getValue() == rhs->getValue();
  }

  static llvm::hash_code hashValue(const TConst *tconst) {
    return llvm::hash_combine(tconst->getKind(), tconst->getValue());
  }

  static bool equal(const TArrow *lhs, const TArrow *rhs) {
    return lhs->getLeft() == rhs->getRight() &&
           lhs->getRight() == rhs->getRight();
  }

  static llvm::hash_code hashValue(const TArrow *tarrow) {
    return llvm::hash_combine(tarrow->getKind(), tarrow->getLeft(),
                              tarrow->getRight());
  }

  static bool equal(const Type *lhs, const Type *rhs) {
    if (lhs->getKind() != rhs->getKind())
      return false;
    if (auto *tconst = dyn_cast<TConst>(lhs))
      return equal(tconst, cast<TConst>(rhs));
    if (auto *tarrow = dyn_cast<TArrow>(lhs))
      return equal(tarrow, cast<TArrow>(rhs));
    return false;
  }

  static llvm::hash_code hashValue(const Type *type) {
    if (auto *tconst = dyn_cast<TConst>(type))
      return hashValue(tconst);
    if (auto *tarrow = dyn_cast<TArrow>(type))
      return hashValue(tarrow);
    llvm_unreachable("forgot to handle a case");
  }

  ///
  /// Expressions
  ///

  static bool equal(const Const *lhs, const Const *rhs) {
    return lhs->getValue() == rhs->getValue();
  }

  static llvm::hash_code hashValue(const Const *konst) {
    return llvm::hash_combine(konst->getKind(), konst->getValue());
  }

  static bool equal(const Anno *lhs, const Anno *rhs) {
    return equal(lhs->getExpr(), rhs->getExpr()) &&
           equal(lhs->getType(), rhs->getType());
  }

  static llvm::hash_code hashValue(const Anno *anno) {
    return llvm::hash_combine(anno->getKind(), anno->getExpr(),
                              anno->getType());
  }

  static bool equal(const Var *lhs, const Var *rhs) {
    return lhs->getIndex() == rhs->getIndex();
  }

  static llvm::hash_code hashValue(const Var *var) {
    return llvm::hash_combine(var->getKind(), var->getIndex());
  }

  static bool equal(const Abs *lhs, const Abs *rhs) {
    return equal(lhs->getBody(), rhs->getBody());
  }

  static llvm::hash_code hashValue(const Abs *abs) {
    return llvm::hash_combine(abs->getKind(), abs->getBody());
  }

  static bool equal(const App *lhs, const App *rhs) {
    return equal(lhs->getFun(), rhs->getFun()) &&
           equal(lhs->getArg(), rhs->getArg());
  }

  static llvm::hash_code hashValue(const App *app) {
    return llvm::hash_combine(app->getKind(), app->getFun(), app->getArg());
  }

  static bool equal(const Expr *lhs, const Expr *rhs) {
    if (lhs->getKind() != rhs->getKind())
      return false;
    if (auto *konst = dyn_cast<Const>(lhs))
      return equal(konst, cast<Const>(rhs));
    if (auto *anno = dyn_cast<Anno>(lhs))
      return equal(anno, cast<Anno>(rhs));
    if (auto *var = dyn_cast<Var>(lhs))
      return equal(var, cast<Var>(rhs));
    if (auto *abs = dyn_cast<Abs>(lhs))
      return equal(abs, cast<Abs>(rhs));
    if (auto *app = dyn_cast<App>(lhs))
      return equal(app, cast<App>(rhs));
    llvm_unreachable("forgot to handle a case");
  }

  static llvm::hash_code hashValue(const Expr *expr) {
    if (auto *konst = dyn_cast<Const>(expr))
      return hashValue(konst);
    if (auto *anno = dyn_cast<Anno>(expr))
      return hashValue(anno);
    if (auto *var = dyn_cast<Var>(expr))
      return hashValue(var);
    if (auto *abs = dyn_cast<Abs>(expr))
      return hashValue(abs);
    if (auto *app = dyn_cast<App>(expr))
      return hashValue(app);
    llvm_unreachable("forgot to handle a case");
  }

  ///
  /// Statements
  ///

  static bool equal(const Declaration *lhs, const Declaration *rhs) {
    return lhs->getName() == rhs->getName() &&
           equal(lhs->getType(), rhs->getType());
  }

  static llvm::hash_code hashValue(const Declaration *decl) {
    return llvm::hash_combine(decl->getKind(), decl->getName(),
                              decl->getType());
  }

  static bool equal(const Definition *lhs, const Definition *rhs) {
    return lhs->getName() == rhs->getName() &&
           equal(lhs->getValue(), rhs->getValue());
  }

  static llvm::hash_code hashValue(const Definition *def) {
    return llvm::hash_combine(def->getKind(), def->getName(), def->getValue());
  }

  static bool equal(const Stmt *lhs, const Stmt *rhs) {
    if (lhs->getKind() != rhs->getKind())
      return false;
    if (auto *decl = dyn_cast<Declaration>(lhs))
      return equal(decl, cast<Declaration>(rhs));
    if (auto *def = dyn_cast<Definition>(lhs))
      return equal(def, cast<Definition>(rhs));
    llvm_unreachable("forgot to handle a case");
  }

  inline llvm::hash_code hashValue(const Stmt *stmt) {
    if (auto *decl = dyn_cast<Declaration>(stmt))
      return hashValue(decl);
    if (auto *def = dyn_cast<Definition>(stmt))
      return hashValue(def);
    llvm_unreachable("forgot to handle a case");
  }

  ///
  /// Compilation Unit
  ///

  static bool equal(const CompUnit *lhs, const CompUnit *rhs) {
    auto &ls = lhs->getStatements();
    auto &rs = rhs->getStatements();
    if (ls.size() != rs.size())
      return false;
    return llvm::all_of(
        llvm::zip(ls, rs), [](auto l, auto r); { return equal(l, r); });
  }

  static llvm::hash_code hasValue(const CompUnit *compUnit) {
    return llvm::hash_combine(compUnit->getStatements());
  }
};

void print(Type *type);
void print(Expr *expr);

} // namespace hwl
} // namespace circt

namespace llvm {

template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_base_of_v<::circt::hwl::Type, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(From target) {
    // Allow constant upcasting.  This also gets around the fact that AnnoTarget
    // does not implement classof.
    if constexpr (std::is_base_of_v<To, From>)
      return true;
    else
      return To::classof(target);
  }
  static inline To doCast(From target) { return To(target.getImpl()); }
};

template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_base_of_v<::circt::hwl::Expr, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(From target) {
    // Allow constant upcasting.  This also gets around the fact that AnnoTarget
    // does not implement classof.
    if constexpr (std::is_base_of_v<To, From>)
      return true;
    else
      return To::classof(target);
  }
  static inline To doCast(From target) { return To(target.getImpl()); }
};

} // namespace llvm

#endif
