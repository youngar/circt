#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
Stream &operator<<(Stream &&stream, const Hole &hole) {
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
Stream &operator<<(Stream &&stream, const Atom &atom) {
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
Stream &operator<<(Stream &&stream, const Pi &pi) {
  stream << "∀ " << *pi.variable << " : " << *pi.type << ", " << *pi.body;
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
Stream &operator<<(Stream &&stream, const Sigma &sigma) {
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
Stream &operator<<(Stream &&stream, const Apply &apply) {
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
Stream &operator<<(Stream &&stream, const Lambda &lambda) {
  return stream << "λ " << *lambda.variable << ": " << *lambda.type << ", "
                << *lambda.body << "";
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
Stream &operator<<(Stream &&stream, const Decl &decl) {
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
Stream &operator<<(Stream &&stream, const Defn &defn) {
  return stream << "" << *defn.atom << " : " << *defn.type << " = "
                << *defn.value << "";
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

struct Module : public ObjectImpl<Kind::Module> {
  Module(std::vector<Object *> decls) : BaseT(), decls(decls) {}
  std::vector<Object *> decls;
};

template <typename Stream>
Stream &operator<<(Stream &&stream, const Module &module) {
  stream << "Module<\n";
  for (auto &decl : module.decls)
    stream << "  " << *decl << "\n";
  return stream << ">";
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

template <typename Stream>
Stream &operator<<(Stream &&stream, const Object &object) {
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

//===----------------------------------------------------------------------===//
// Unification of Terms
//===----------------------------------------------------------------------===//

// class LVar {
//   LVar *value;
// };

// class SubstitutionTable {

// };

// inline void unify(Object *lhs, Object *rhs) {
//   if (auto lhs = dyn_cast<
// }

//===----------------------------------------------------------------------===//
// Equality
//===----------------------------------------------------------------------===//

class IndexScope;
class IndexMap {
  using Map = llvm::DenseMap<const Atom *, size_t>;

public:
  IndexMap() : map(), scope(nullptr) {}

  size_t get(const Atom *atom) const;

  size_t set(const Atom *atom);

private:
  friend class IndexScope;

  Map map;
  IndexScope *scope;
};

class IndexScope {
public:
  IndexScope(IndexMap &map) : map(map), prev(map.scope), elements() {
    map.scope = this;
  }

  IndexScope(IndexScope &&) = delete;
  IndexScope(const IndexScope &) = delete;

  ~IndexScope() {
    for (auto &entry : elements)
      map.map.erase(entry);
    map.scope = prev;
  }

  size_t get(const Atom *atom);

  size_t set(const Atom *atom);

private:
  IndexMap &map;
  IndexScope *prev;
  llvm::SmallVector<const Atom *> elements;
};

inline size_t IndexScope::get(const Atom *atom) {
  auto id = map.map[atom];
  // assert(id != 0);
  return id;
}

inline size_t IndexScope::set(const Atom *atom) {
  assert(map.map[atom] == 0);

  auto id = map.map.size() + 1;
  map.map[atom] = id;
  elements.push_back(atom);
  return id;
}

inline size_t IndexMap::get(const Atom *atom) const { return scope->get(atom); }

inline size_t IndexMap::set(const Atom *atom) { return scope->set(atom); }

struct IndexTable {
  IndexMap lhs;
  IndexMap rhs;
};

bool equal(const Object *lhs, const Object *rhs, IndexTable &table);

inline bool equalAtom(const Atom *lhs, const Object *obj, IndexTable &table) {
  auto rhs = dyn_cast<Atom>(obj);
  if (!rhs)
    return false;

  return lhs == rhs || table.lhs.get(lhs) == table.rhs.get(rhs);
}

inline bool equalPi(const Pi *lhs, const Object *obj, IndexTable &table) {
  auto *rhs = dyn_cast<Pi>(obj);
  if (!rhs)
    return false;

  IndexScope lhsScope(table.lhs);
  IndexScope rhsScope(table.rhs);
  auto lhsId = table.lhs.set(lhs->variable);
  auto rhsId = table.rhs.set(rhs->variable);
  if (lhsId != rhsId)
    return false;

  if (!equal(lhs->type, rhs->type, table))
    return false;

  if (!equal(lhs->body, rhs->body, table))
    return false;

  return true;
}

inline bool equalApply(const Apply *lhs, const Object *obj, IndexTable &table) {
  auto rhs = dyn_cast<Apply>(obj);
  if (!rhs)
    return false;

  return equal(lhs->fun, rhs->fun, table) && equal(lhs->arg, rhs->arg, table);
}

inline bool equalLambda(const Lambda *lhs, const Object *obj,
                        IndexTable &table) {
  std::cerr << "====================\n";

  auto rhs = dyn_cast<Lambda>(obj);
  if (!rhs)
    return false;

  std::cerr << "lhs=" << *lhs << "\n";
  std::cerr << "rhs=" << *rhs << "\n";

  IndexScope lhsScope(table.lhs);
  IndexScope rhsScope(table.rhs);
  auto lhsId = table.lhs.set(lhs->variable);
  auto rhsId = table.rhs.set(rhs->variable);

  // std::cerr << "====lhs====\n";
  // for (const auto &binding : table.lhs)
  //   std::cerr << binding.first->name << "=" << binding.second << "\n";

  // std::cerr << "====rhs====\n";
  // for (const auto &binding : table.rhs)
  //   std::cerr << binding.first->name << "=" << binding.second << "\n";

  if (lhsId != rhsId)
    return false;

  return equal(lhs->body, rhs->body, table);
}

inline bool equal(const Object *lhs, const Object *rhs, IndexTable &table) {
  std::cerr << "--------\n";
  std::cerr << "lhs = " << *lhs << "\n";
  std::cerr << "rhs = " << *rhs << "\n";

  if (auto atom = dyn_cast<Atom>(lhs)) {
    return equalAtom(atom, rhs, table);
  }

  if (auto pi = dyn_cast<Pi>(lhs)) {
    return equalPi(pi, rhs, table);
  }

  if (auto lambda = dyn_cast<Lambda>(lhs)) {
    return equalLambda(lambda, rhs, table);
  }

  if (auto apply = dyn_cast<Apply>(lhs)) {
    return equalApply(apply, rhs, table);
  }

  return false;
}

inline bool equal(const Object *lhs, const Object *rhs) {
  IndexTable table;
  IndexScope lhsScope(table.lhs);
  IndexScope rhsScope(table.rhs);
  return equal(lhs, rhs, table);
}

} // namespace aug
