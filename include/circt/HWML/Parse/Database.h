#ifndef CIRCT_HWML_PARSE_DATABASE_H
#define CIRCT_HWML_PARSE_DATABASE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>

namespace circt {
namespace hwml {

//// Underlying

struct Revision;
struct Context;
struct Input;
struct Output;

struct Revision {
  Revision();
  operator std::uintptr_t();

private:
  std::uintptr_t revision;
};

template <typename K, typename V>
struct DatabaseStorage {
  DenseMap<K, V> underlying;
};

template <typename K, typename V>
struct Database {
  Context *getContext();

  Revision getRevision();

  template <typename ReturnT>
  ReturnT get() {}

private:
  static DatabaseStorage<K, V> storage;
};

struct Context {

  template <typename K, typename V>
  V getValue(K key) {
    return DatabaseStorage<K, V>::get();
  }

private:
};

struct Input {
  virtual bool calculate();
};

struct Output {

  Output() {}

  bool isValid();

private:
  bool valid = false;
  Revision verifiedAt;
};

template <typename T>
struct Query {};

struct GetAST {
  void operator()(Context *context) {}
};

static DatabaseStorage<GetAST>();


} // namespace hwml
} // namespace circt

#endif