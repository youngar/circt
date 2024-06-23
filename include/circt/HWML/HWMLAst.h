#ifndef CIRCT_HWML_HWMLAST_H
#define CIRCT_HWML_HWMLAST_H

#include "llvm/Support/Casting.h"
#include "circt/Support/LLVM.h"
#include <cassert>
#include <string>

namespace circt {
namespace hwml {

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
  std::string value;
};

struct TArrow : public Type {
  static Kind kind() { return Kind::TArrow; }
  TArrow(Type *left, Type *right) : Type(kind()), left(left), right(right) {}
  static bool classof(const Type *type) { return type->getKind() == kind(); }
  Type *left;
  Type *right;
};

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
  std::string value;
};

struct Anno : public Expr {
  static Kind kind() { return Kind::Anno; }
  Anno(Expr *expr, Type *type) : Expr(kind()), expr(expr), type(type) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  Expr *expr;
  Type *type;
};

struct Var : public Expr {
  static Kind kind() { return Kind::Var; }
  Var(uint64_t index) : Expr(kind()), index(index) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  uint64_t index;
};

struct Abs : public Expr {
  static Kind kind() { return Kind::Abs; }
  Abs(Expr *body) : Expr(kind()), body(body) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  Expr *body;
};

struct App : public Expr {
  static Kind kind() { return Kind::App; }
  App(Expr *fun, Expr *arg) : Expr(kind()), fun(fun), arg(arg) {}
  static bool classof(const Expr *expr) { return expr->getKind() == kind(); }
  Expr *fun;
  Expr *arg;
};

void print(Type *type);
void print(Expr *expr);
Expr *eval(Expr *expr);

///
/// Type Checking
///

Type *typeSynth(std::vector<Type *> ctxt, Expr *expr);

bool typeCheck(std::vector<Type *> ctxt, Expr *expr, Type *type);

} // namespace hwml
} // namespace circt

namespace llvm {

template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_base_of_v<::circt::hwml::Type, From>>>
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
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_base_of_v<::circt::hwml::Expr, From>>>
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
