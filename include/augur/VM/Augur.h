#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlir/Support/LogicalResult.h"

////////////////////////////////

using namespace llvm;

namespace aug {
enum class Kind {
  // Objects
  Hole,
  Atom,
  Decl,
  Defn,
  Sigma,
  Pi,
  Lambda,
  Apply,
  Module,
};

//===----------------------------------------------------------------------===//
// Object
//===----------------------------------------------------------------------===//

struct Object {
  explicit Object(Kind kind) : kind(kind) {}
  Kind kind;
  Object *leader = nullptr;
};

template <Kind k>
struct ObjectImpl : public Object {
  using BaseT = ObjectImpl<k>;
  ObjectImpl() : Object(k){};
  static bool classof(const Object *object) { return object->kind == k; }
};

//===----------------------------------------------------------------------===//
// Hole
//===----------------------------------------------------------------------===//

struct Hole : public ObjectImpl<Kind::Hole> {
  Hole() : BaseT() {}
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Hole &hole) {
  return stream << "Hole";
}

//===----------------------------------------------------------------------===//
// Atom
//===----------------------------------------------------------------------===//

struct Atom : public ObjectImpl<Kind::Atom> {
  Atom(std::string name) : BaseT(), name(name) {}
  std::string name;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Atom &atom) {
  stream << "" << atom.name;
  // stream << "(" << &atom << ")";
  return stream;
}

//===----------------------------------------------------------------------===//
// Pi
//===----------------------------------------------------------------------===//

struct Pi : public ObjectImpl<Kind::Pi> {
  Pi(Atom *variable, Object *type, Object *body)
      : BaseT(), variable(variable), type(type), body(body) {}
  Atom *variable;
  Object *type;
  Object *body;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Pi &pi) {
  stream << "∀ " << *pi.variable << " : " << *pi.type
         << ", " << *pi.body;
  return stream;
}

//===----------------------------------------------------------------------===//
// Sigma
//===----------------------------------------------------------------------===//

struct Sigma : public ObjectImpl<Kind::Sigma> {
  Sigma(Atom *lhs, Object *rhs) : BaseT(), lhs(lhs), rhs(rhs) {}
  Atom *lhs;
  Object *rhs;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Sigma &sigma) {
  return stream << "Sigma(variable=" << *sigma.lhs << ", body=" << *sigma.rhs
                << ")";
}

//===----------------------------------------------------------------------===//
// Apply
//===----------------------------------------------------------------------===//

struct Apply : public ObjectImpl<Kind::Apply> {
  Apply(Object *fun, Object *arg) : BaseT(), fun(fun), arg(arg) {}
  Object *fun;
  Object *arg;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Apply &apply) {
  return stream << "(" << *apply.fun << " " << *apply.arg << ")";
}

//===----------------------------------------------------------------------===//
// Lambda
//===----------------------------------------------------------------------===//

struct Lambda : public ObjectImpl<Kind::Lambda> {
  Lambda(Atom *variable, Object *type, Object *body)
      : BaseT(), variable(variable), type(type), body(body) {}
  Atom *variable;
  Object *type;
  Object *body;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Lambda &lambda) {
  return stream << "λ " << *lambda.variable
                << ": " << *lambda.type << ", " << *lambda.body
                << "";
}

//===----------------------------------------------------------------------===//
// Decl
//===----------------------------------------------------------------------===//

struct Decl : public ObjectImpl<Kind::Decl> {
  Decl(Atom *atom, Object *type) : BaseT(), atom(atom), type(type) {}
  Atom *atom;
  Object *type;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Decl &decl) {
  return stream << "" << *decl.atom << " : " << *decl.type << "";
}

//===----------------------------------------------------------------------===//
// Defn
//===----------------------------------------------------------------------===//

struct Defn : public ObjectImpl<Kind::Defn> {
  Defn(Atom *atom, Object *type, Object *value)
      : BaseT(), atom(atom), type(type), value(value) {}
  Atom *atom;
  Object *type;
  Object *value;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Defn &defn) {
  return stream << "" << *defn.atom << " : " << *defn.type
                << " = " << *defn.value << "";
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

struct Module : public ObjectImpl<Kind::Module> {
  Module(std::vector<Object *> decls) : BaseT(), decls(decls) {}
  std::vector<Object *> decls;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, Module &module) {
  stream << "Module<\n";
  for (auto &decl : module.decls)
    stream << "  " << *decl << "\n";
  return stream << ">";
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

template <typename Stream>
Stream &operator<<(Stream &&stream, Object &object) {
  auto *o = &object;
  if (auto *hole = dyn_cast<Hole>(o))
    stream << *hole;
  else if (auto *atom = dyn_cast<Atom>(o))
    stream << *atom;
  else if (auto *decl = dyn_cast<Decl>(o))
    stream << *decl;
  else if (auto *defn = dyn_cast<Defn>(o))
    stream << *defn;
  else if (auto *pi = dyn_cast<Pi>(o))
    stream << *pi;
  else if (auto *sigma = dyn_cast<Sigma>(o))
    stream << *sigma;
  else if (auto *lambda = dyn_cast<Lambda>(o))
    stream << *lambda;
  else if (auto *app = dyn_cast<Apply>(o))
    stream << *app;
  else if (auto *module = dyn_cast<Module>(o))
    stream << *module;
  else
    stream << "<UNKNOWN-KIND>";
  return stream;
}

//===----------------------------------------------------------------------===//
// MemoryManager
//===----------------------------------------------------------------------===//

struct MemoryManager {

  bool initialize() { return true; }

  template <typename T, typename... Args>
  T *allocate(Args &&...args) {
    return new (allocator) T(std::forward<Args>(args)...);
  }

  llvm::BumpPtrAllocator allocator;
};

//===----------------------------------------------------------------------===//
// VirtualMachine
//===----------------------------------------------------------------------===//

struct VirtualMachine {
  VirtualMachine() {}

  bool initialize() {
    if (!memoryManager.initialize())
      return false;
    auto *atom = create<Atom>("type");
    type = create<Decl>(atom, nullptr);
    type->type = type;
    return true;
  }

  Atom *getType() { return type->atom; }

  template <typename ObjectT, typename... Args>
  ObjectT *create(Args... args) {
    return memoryManager.allocate<ObjectT>(args...);
  }

  MemoryManager memoryManager;
  Decl *type;
};


mlir::LogicalResult typeCheck(VirtualMachine &vm, Module *module);

Object *eval(VirtualMachine &vm, Module *module);

} // namespace aug