#include "augur/VM/Augur.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include <iostream>

using namespace mlir;
using namespace aug;

namespace {

using Env = llvm::ScopedHashTable<Atom *, Object *>;
using EnvScope = Env::ScopeTy;

struct TypeChecker {

  TypeChecker(VirtualMachine &vm) : vm(vm) {}

  Object *eval(Object *obj) {
    if (auto *atom = dyn_cast<Atom>(obj)) {
      // If the atom has a value, return it.
      if (auto *value = env.lookup(atom))
        return value;
      // Irreducible:
      return atom;
    }

    if (auto *apply = dyn_cast<Apply>(obj)) {
      auto *fun = eval(apply->fun);
      auto *arg = eval(apply->arg);

      // If we are applying a lambda, perform beta substitution.
      if (auto *lambda = dyn_cast<Lambda>(fun)) {
        EnvScope scope(env);
        env.insert(lambda->variable, arg);
        return eval(lambda->body);
      }
      // Apply is irreducible, but maybe its fun or arg were.
      return vm.create<Apply>(fun, arg);
    }

    if (auto *lambda = dyn_cast<Lambda>(obj)) {
      // Since we are cloning the lambda, we need to replace
      // uses of the argument on the old lambda with uses on
      // of the new argument.
      auto *oldVariable = lambda->variable;
      auto *newVariable = vm.create<Atom>(oldVariable->name);
      EnvScope scope(env);
      env.insert(oldVariable, newVariable);

      auto *type = eval(lambda->type);
      auto *body = eval(lambda->body);
      return vm.create<Lambda>(newVariable, type, body);
    }

    if (auto *pi = dyn_cast<Pi>(obj)) {
      auto *oldVariable = pi->variable;
      auto *newVariable = vm.create<Atom>(oldVariable->name);
      EnvScope scope(env);
      env.insert(oldVariable, newVariable);

      auto *type = eval(pi->type);
      auto *body = eval(pi->body);
      return vm.create<Pi>(newVariable, type, body);
    }

    if (auto *module = dyn_cast<Module>(obj)) {
      EnvScope scope(env);
      std::vector<Object *> decls;
      for (auto *object : module->decls) {
        if (auto *decl = dyn_cast<Decl>(object)) {
          // TODO: we are copying the memory we might not own.
          decls.push_back(decl);
        } else if (auto *defn = dyn_cast<Defn>(object)) {
          auto *value = eval(defn->value);
          env.insert(defn->atom, value);
          decls.push_back(vm.create<Defn>(defn->atom, defn->type, value));
        } else {
          std::cout << "unknown element in module: " << *object << "\n";
          abort();
          return nullptr;
        }
      }
      return vm.create<Module>(decls);
    }

    std::cout << "!!!! what is this: " << *obj << "\n";
    abort();
  }

  Object *typeSynth(Object *obj) {
    if (auto *atom = dyn_cast<Atom>(obj)) {
      auto *synth = context.lookup(atom);
      if (!synth) {
        std::cout << "atom not found in typing context " << *atom << "\n";
        return nullptr;
      }
      return synth;
    }

    if (auto *apply = dyn_cast<Apply>(obj)) {
      auto *funType = typeSynth(apply->fun);
      if (!funType)
        return nullptr;
      auto *pi = dyn_cast<Pi>(funType);
      if (!pi) {
        std::cout << "expected pi type, got: " << *funType << "\n";
        return nullptr;
      }

      if (failed(typeCheck(apply->arg, pi->type)))
        return nullptr;
      auto *arg = eval(apply->arg);

      EnvScope scope(env);
      env.insert(pi->variable, arg);
      return eval(pi->body);
    }

    if (auto *pi = dyn_cast<Pi>(obj)) {
      if (failed(typeCheck(pi->type, type)))
        return nullptr;
      auto *varType = eval(pi->type);

      
      EnvScope ctxScope(context);
      context.insert(pi->variable, varType);

      // EnvScope envScope(env);
      // env.insert(pi->variable, varType);
      // TODO: do we need to evaluate the body here? I don't think so.
      auto *bodyType = eval(pi->body);
      
      if (failed(typeCheck(bodyType, type)))
        return nullptr;
      return type;
    }

    std::cout << "unable to synthesize thy type " << *obj << "\n";
    abort();
  }

  LogicalResult typeCheck(Object *expr, Object *type) {
    if (auto *lambda = dyn_cast<Lambda>(expr)) {
      auto *pi = dyn_cast<Pi>((type));
      if (!pi) {
        std::cout << "expected type " << *type << " for expression " << *expr
                  << "\n";
        return failure();
      }

      EnvScope scope(context);
      context.insert(lambda->variable, pi->type);
      if (failed(typeCheck(lambda->body, pi->body)))
        return failure();
      return success();
    }

    if (auto *exprType = typeSynth(expr)) {
      if (exprType != type) {
        std::cout << "expected type " << *type << ", but got " << *exprType
                  << " for expression " << *expr << "\n";
        return failure();
      }
      return success();
    }

    std::cout << "expected type " << *type << ", for expression " << *expr
              << "\n";
    return failure();
  }

  LogicalResult typeCheck(Module *module) {
    EnvScope ctxScope(context);
    EnvScope envScope(env);
    for (auto *object : module->decls) {
      if (auto *decl = dyn_cast<Decl>(object)) {
        auto *atom = decl->atom;
        // Major hack.
        if (atom->name == "type") {
          type = atom;
          // env.insert(atom, atom);
          context.insert(atom, atom);
        }
        if (failed(typeCheck(decl->type, type))) {
          std::cout << "failed to typecheck " << *decl << "\n";
          return failure();
        }
        context.insert(decl->atom, decl->type);
      } else if (auto *defn = dyn_cast<Defn>(object)) {
        if (failed(typeCheck(defn->type, type))) {
          std::cout << "failed to typecheck " << *defn << "\n";
          return failure();
        }
        auto *defnType = eval(defn->type);
        if (failed(typeCheck(defn->value, defnType))) {
          std::cout << "failed to typecheck " << *defn << "\n";
          return failure();
        }
        context.insert(defn->atom, defnType);
      } else {
        std::cout << "unknown element in module: " << *object << "\n";
        return failure();
      }
    }
    return success();
  }

  VirtualMachine &vm;
  Object *type;
  // For evaluation, maps an atom to its value.
  Env env;
  // For typechecking, maps an atom to its type.
  Env context;
};
} // namespace

LogicalResult aug::typeCheck(VirtualMachine &vm, Module *module) {
  return TypeChecker(vm).typeCheck(module);
}

Object *aug::eval(VirtualMachine &vm, Module *module) {
  return TypeChecker(vm).eval(module);
}
