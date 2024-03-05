#ifndef CIRCT_HWML_HWMLAST_H
#define CIRCT_HWML_HWMLAST_H

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
  TConst(std::string value) : Type(kind()), value(value) {}
  std::string value;
};

struct TArrow : public Type {
  static Kind kind() { return Kind::TArrow; }
  TArrow(Type *left, Type *right) : Type(kind()), left(left), right(right) {}
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
  Const(std::string value) : Expr(kind()), value(value) {}
  std::string value;
};

struct Anno : public Expr {
  static Kind kind() { return Kind::Anno; }
  Anno(Expr *expr, Type *type) : Expr(kind()), expr(expr), type(type) {}
  Expr *expr;
  Type *type;
};

struct Var : public Expr {
  static Kind kind() { return Kind::Var; }
  Var(uint64_t index) : Expr(kind()), index(index) {}
  uint64_t index;
};

struct Abs : public Expr {
  static Kind kind() { return Kind::Abs; }
  Abs(Expr *body) : Expr(kind()), body(body) {}
  Expr *body;
};

struct App : public Expr {
  static Kind kind() { return Kind::App; }
  App(Expr *fun, Expr *arg) : Expr(kind()), fun(fun), arg(arg) {}
  Expr *fun;
  Expr *arg;
};

template <typename ToT, typename FromT>
bool isa(FromT *value) {
  return value->getKind() == ToT::kind();
}

template <typename ToT, typename FromT>
ToT *cast(FromT *value) {
  assert(isa<ToT>(value));
  return static_cast<ToT *>(value);
}

template <typename ToT, typename FromT>
ToT *dyn_cast(FromT *value) {
  return isa<ToT>(value) ? cast<ToT>(value) : nullptr;
}

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

#endif