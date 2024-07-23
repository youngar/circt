#include "circt/HWL/HWLAst.h"
#include "circt/HWL/HWLTypeCheck.h"
#include <iostream>
#include <vector>

///
/// Print
///

namespace circt {
namespace hwl {

void print(Type *type) {
  if (auto *tconst = dyn_cast<TConst>(type)) {
    std::cerr << tconst->getValue();
    return;
  }
  if (auto *tarrow = dyn_cast<TArrow>(type)) {
    print(tarrow->getLeft());
    std::cerr << " -> ";
    print(tarrow->getRight());
    return;
  }

  abort();
}

void print(Expr *expr) {
  if (auto *anno = dyn_cast<Anno>(expr)) {
    print(anno->getExpr());
    std::cerr << " : ";
    print(anno->getType());
    return;
  }

  if (auto *c = dyn_cast<Const>(expr)) {
    std::cerr << c->getValue();
    return;
  }

  if (auto *var = dyn_cast<Var>(expr)) {
    std::cerr << "\\" << var->getIndex();
    return;
  }

  if (auto *app = dyn_cast<App>(expr)) {
    print(app->getFun());
    std::cerr << " ";
    print(app->getArg());
    return;
  }

  if (auto *abs = dyn_cast<Abs>(expr)) {
    std::cerr << "(λ ";
    print(abs->getBody());
    std::cerr << ")";
    return;
  }

  abort();
}

///
/// Eval
///

Expr *eval(std::vector<Expr *> ctxt, Expr *expr, uint64_t binderDepth) {
  if (auto *anno = dyn_cast<Anno>(expr))
    return eval(ctxt, anno->getExpr(), binderDepth);

  if (auto *c = dyn_cast<Const>(expr))
    return c;

  if (auto *var = dyn_cast<Var>(expr))
    return ctxt[ctxt.size() - 1 - var->getIndex()];

  if (auto *app = dyn_cast<App>(expr)) {
    auto *fun = dyn_cast<Abs>(eval(ctxt, app->getFun(), binderDepth));
    if (!fun)
      return nullptr;

    auto *arg = eval(ctxt, app->getArg(), binderDepth);
    if (!arg)
      return nullptr;

    ctxt.push_back(arg);
    auto *result = eval(ctxt, fun->getBody(), binderDepth);
    ctxt.pop_back();
    return result;
  }

  if (auto *abs = dyn_cast<Abs>(expr)) {
    // no idea lol
  }

  abort();
  return expr;
}

Expr *eval(Expr *expr) { return eval({}, expr, 0); }

bool typeEq(Type *lhs, Type *rhs) {
  if (auto larrow = dyn_cast<TArrow>(lhs)) {
    auto rarrow = dyn_cast<TArrow>(rhs);
    if (!rarrow)
      return false;
    return typeEq(larrow->getLeft(), rarrow->getLeft()) &&
           typeEq(larrow->getRight(), rarrow->getRight());
  }

  if (auto ltconst = dyn_cast<TConst>(lhs)) {
    auto rtconst = dyn_cast<TConst>(rhs);
    if (!rtconst)
      return false;
    return ltconst->getValue() == rtconst->getValue();
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

    ctxt.push_back(tarrow->getLeft());
    auto result = typeCheck(ctxt, abs->getBody(), tarrow->getRight());
    ctxt.pop_back();
    return result;
  }

  if (auto *app = dyn_cast<App>(expr)) {
    auto tfun = typeSynth(ctxt, app->getFun());
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
    if (!typeCheck(ctxt, anno->getExpr(), anno->getType()))
      return nullptr;
    return anno->getType();
  }

  if (auto *var = dyn_cast<Var>(expr)) {
    auto index = var->getIndex();
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

} // namespace hwl
} // namespace circt
