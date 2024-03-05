#include "circt/HWML/HWMLAst.h"
#include <iostream>
#include <vector>

///
/// Print
///

namespace circt {
namespace hwml {

void print(Type *type) {
  if (auto *tconst = dyn_cast<TConst>(type)) {
    std::cout << tconst->value;
    return;
  }
  if (auto *tarrow = dyn_cast<TArrow>(type)) {
    print(tarrow->left);
    std::cout << " -> ";
    print(tarrow->right);
    return;
  }

  abort();
}

void print(Expr *expr) {
  if (auto *anno = dyn_cast<Anno>(expr)) {
    print(anno->expr);
    std::cout << " : ";
    print(anno->type);
    return;
  }

  if (auto *c = dyn_cast<Const>(expr)) {
    std::cout << c->value;
    return;
  }

  if (auto *var = dyn_cast<Var>(expr)) {
    std::cout << "\\" << var->index;
    return;
  }

  if (auto *app = dyn_cast<App>(expr)) {
    print(app->fun);
    std::cout << " ";
    print(app->arg);
    return;
  }

  if (auto *abs = dyn_cast<Abs>(expr)) {
    std::cout << "(λ ";
    print(abs->body);
    std::cout << ")";
    return;
  }

  abort();
}

///
/// Eval
///

Expr *eval(std::vector<Expr *> ctxt, Expr *expr, uint64_t binderDepth) {
  if (auto *anno = dyn_cast<Anno>(expr)) {
    return eval(ctxt, anno->expr, binderDepth);
  }

  if (auto *c = dyn_cast<Const>(expr)) {
    return c;
  }

  if (auto *var = dyn_cast<Var>(expr)) {
    return ctxt[ctxt.size() - 1 - var->index];
  }

  if (auto *app = dyn_cast<App>(expr)) {
    auto fun = dyn_cast<Abs>(eval(ctxt, app->fun, binderDepth));
    if (!fun)
      return nullptr;

    auto arg = eval(ctxt, app->arg, binderDepth);
    if (!arg)
      return nullptr;

    ctxt.push_back(arg);
    auto result = eval(ctxt, fun->body, binderDepth);
    ctxt.pop_back();
    return result;
  }

  if (auto *abs = dyn_cast<Abs>(expr)) {
  }

  return expr;
}

Expr *eval(Expr *expr) { return eval({}, expr, 0); }

bool typeEq(Type *lhs, Type *rhs) {
  if (auto larrow = dyn_cast<TArrow>(lhs)) {
    auto rarrow = dyn_cast<TArrow>(rhs);
    if (!rarrow)
      return false;
    return typeEq(larrow->left, rarrow->left) &&
           typeEq(larrow->right, rarrow->right);
  }

  if (auto ltconst = dyn_cast<TConst>(lhs)) {
    auto rtconst = dyn_cast<TConst>(rhs);
    if (!rtconst)
      return false;
    return ltconst->value == rtconst->value;
  }

  return false;
}

bool typeCheck(std::vector<Type *> ctxt, Expr *expr, Type *expected) {
  if (auto *c = dyn_cast<Const>(expr)) {
    auto *synth = typeSynth(ctxt, c);
    return typeEq(synth, expected);
  }

  if (auto *anno = dyn_cast<Anno>(expr)) {
    auto *synth = typeSynth(ctxt, anno);
    return typeEq(synth, expected);
  }

  if (auto *abs = dyn_cast<Abs>(expr)) {
    auto *tarrow = dyn_cast<TArrow>(expected);
    if (!tarrow)
      return false;

    ctxt.push_back(tarrow->left);
    auto result = typeCheck(ctxt, abs->body, tarrow->right);
    ctxt.pop_back();
    return result;
  }

  if (auto *app = dyn_cast<App>(expr)) {
    auto tfun = typeSynth(ctxt, app->fun);
    if (!tfun)
      return false;
    auto tarrow = dyn_cast<TArrow>(tfun);
    if (!tarrow)
      return false;
  }

  abort();
  return false;
}

Type *typeSynth(std::vector<Type *> ctxt, Expr *expr) {

  if (auto *anno = dyn_cast<Anno>(expr)) {
    if (!typeCheck(ctxt, anno->expr, anno->type))
      return nullptr;
    return anno->type;
  }

  if (auto *var = dyn_cast<Var>(expr)) {
    auto index = var->index;
    auto size = ctxt.size();
    if (index >= size)
      return nullptr;
    return ctxt[size - index];
  }

  if (auto *c = dyn_cast<Const>(expr)) {
    // return new TConst();
  }

  abort();
  return nullptr;
}

} // namespace hwml
} // namespace circt