#ifndef CIRCT_HWML_INCREMENTAL_DATABASE_H
#define CIRCT_HWML_INCREMENTAL_DATABASE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdint.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace circt {
namespace hwml {

namespace inc {

struct Revision {
  constexpr Revision() : value(0) {}
  explicit constexpr Revision(uintptr_t value) : value(value) {}
  constexpr bool operator==(Revision rhs) const { return value == rhs.value; }
  constexpr bool operator>(Revision rhs) const { return value > rhs.value; }
  constexpr bool operator<(Revision rhs) const { return value < rhs.value; }
  constexpr bool operator>=(Revision rhs) const { return value >= rhs.value; }
  constexpr bool operator<=(Revision rhs) const { return value <= rhs.value; }
  constexpr bool operator!=(Revision rhs) const { return value <= rhs.value; }
  Revision &operator++() {
    ++value;
    return *this;
  }
  // friend constexpr size_t hash_value(Revision r) {
  //   return llvm::hash_value(r.value);
  // }
  uintptr_t value;
};

template <typename D>
class Context;
template <typename D>
class QueryContext;

template <typename T>
class Query;
template <typename T>
struct CachedQueryInfo;

class Database {
  template <typename D>
  friend class Context;
  template <typename D>
  friend class QueryContext;

public:
  constexpr Database() = default;
  constexpr Database(Revision revision) : revision(revision) {}

  Revision getRevision() const { return revision; }

protected:
  Revision incRevision() { return ++revision; }

  Revision revision;
};

template <typename T>
struct TupleIndexSequence;

template <typename... Ts>
struct TupleIndexSequence<std::tuple<Ts...>> {
  using type = std::index_sequence_for<Ts...>;
};

template <typename T>
using TupleIndexSeq = typename TupleIndexSequence<T>::type;

template <typename T>
using Plain = std::remove_reference_t<std::remove_cv_t<T>>;

template <typename T, typename U>
constexpr bool IsA = std::is_convertible_v<Plain<T> *, Plain<U> *>;

template <typename T>
constexpr bool IsDatabase = IsA<T, Database>;

template <typename T, typename D>
constexpr bool IsContext = IsA<T, Context<D>>;

template <typename T, typename D>
constexpr bool IsQueryContext = IsA<T, QueryContext<D>>;

template <typename F, typename = void>
struct QueryFnTraits;

template <typename R, typename D, typename... As>
struct QueryFnTraits<R(QueryContext<D>, As...)> {
  using Database = std::decay_t<D>;
  using Val = std::decay_t<R>;
  using Key = std::tuple<std::decay_t<As>...>;
  static_assert(IsDatabase<D>);
};

template <typename R, typename D, typename... As>
struct QueryFnTraits<R (*)(D, As...)> : QueryFnTraits<R(D, As...)> {};

template <typename T, typename R, typename D, typename... As>
struct QueryFnTraits<R (T::*)(D, As...)> : QueryFnTraits<R(D, As...)> {};

template <typename T, typename R, typename D, typename... As>
struct QueryFnTraits<R (T::*)(D, As...) const> : QueryFnTraits<R(D, As...)> {};

template <typename T>
struct QueryFnTraits<T, std::enable_if_t<std::is_class_v<T>>>
    : QueryFnTraits<decltype(&T::operator())> {};

template <typename F>
struct QueryFnTraits<F &> : QueryFnTraits<F> {};
template <typename F>
using KeyTy = typename QueryFnTraits<F>::Key;
template <typename F>
using ValTy = typename QueryFnTraits<F>::Val;
template <typename F>
using DatabaseTy = typename QueryFnTraits<F>::Database;

template <typename D>
class DatabaseHandle {
  static_assert(IsDatabase<D>);

public:
  DatabaseHandle(const DatabaseHandle &) = default;
  explicit DatabaseHandle(const D *database) : database(database) {}

  const D *operator->() const { return database; }
  const D &operator*() const { return *database; }
  const D *getDatabase() const { return database; }
  Revision getRevision() const { return database->getRevision(); }

protected:
  const D *database;
};

// template <typename D>
// class ContextBase : public DatabaseHandle<D> {
// public:
//   ContextBase(D *database) : DatabaseHandle<D>(database) {}
// };

template <typename D>
class Context final : public DatabaseHandle<D> {
  template <typename T>
  friend class Input;

public:
  Context(D *database) : DatabaseHandle<D>(database) {}
  Context(const Context &) = default;
  Context(Context &&) = default;
  Context() = delete;

protected:
  Revision incRevision() const { return this->getDatabase()->incRevision(); }
};

template <typename D>
Context(D &) -> Context<Plain<D>>;

template <typename T>
class Input;

class InputBase {
  template <typename T>
  friend class Input;

public:
  Revision getModifiedAt() const { return modifiedAt; }

protected:
  InputBase(Revision world) : modifiedAt(world) {}

  Revision modifiedAt;
};

template <typename T>
class Input final : InputBase {
public:
  Input(Input &&) = delete;
  Input(const Input &) = delete;

  /// Construct an initial Input object.
  template <typename D, typename... Args>
  Input(Context<D> ctx, Args &&...args)
      : InputBase(ctx.incRevision()), value(std::forward<Args>(args)...) {
    static_assert(IsDatabase<D>);
  }

  /// Mutable access to the underyling value.
  /// Increments the modifiedAt stamp of this input, and the database.
  template <typename D>
  T &modify(Context<D> ctx) {
    static_assert(IsDatabase<D>);
    auto revision = ctx.incRevision();
    assert(modifiedAt < revision);
    modifiedAt = revision;
    return value;
  }

  /// Read-only access to the underyling value.
  template <typename D>
  const T &get(Context<D> ctx) const {
    static_assert(IsDatabase<D>);
    assert(ctx.getRevision() <= modifiedAt);
    return value;
  }

  /// Read-only access to the underyling value.
  /// Records that the ongoing query depends on this input.
  template <typename D>
  const T &get(QueryContext<D> ctx) const {
    static_assert(IsDatabase<D>);
    assert(ctx.getRevision() <= modifiedAt);
    ctx.recordInputDependency(this);
    return value;
  }

private:
  T value;
};

template <typename T>
Input(T &&) -> Input<T>;
template <typename T>
Input(const T &) -> Input<T>;

class CachedQueryBase;

struct DependencySet {
  void clear() {
    inputSet.clear();
    querySet.clear();
  }
  llvm::DenseSet<const InputBase *> inputSet;
  llvm::DenseSet<CachedQueryBase *> querySet;
};

class CachedQueryBase {
  template <typename D>
  friend class QueryContext;
  template <typename T>
  friend class Query;

protected:
  CachedQueryBase(Revision computedAt, DependencySet &&dependencySet)
      : computedAt(computedAt), dependencySet(std::move(dependencySet)) {}

  ~CachedQueryBase() = default;

  virtual bool recompute() = 0;

  bool refresh(Revision world) {
    if (world == computedAt)
      return false;

    for (auto *input : dependencySet.inputSet)
      if (computedAt < input->getModifiedAt())
        return recompute();

    for (auto *query : dependencySet.querySet)
      if (query->refresh(world))
        return recompute();

    computedAt = world;
    return false;
  }

  Revision computedAt;
  DependencySet dependencySet;
};

template <typename D>
class QueryContext final : public DatabaseHandle<D> {
  template <typename T>
  friend class Query;
  template <typename T>
  friend class CachedQuery;
  template <typename T>
  friend class Input;

protected:
  QueryContext(const D *database, DependencySet *dependencySet)
      : DatabaseHandle<D>(database), dependencySet(dependencySet) {}

  void recordInputDependency(const InputBase *input) {
    dependencySet->inputSet.insert(input);
  }

  void recordQueryDependency(CachedQueryBase *query) {
    dependencySet->querySet.insert(query);
  }

  DependencySet *dependencySet;
};

template <typename T>
class CachedQuery final : CachedQueryBase {
  using K = KeyTy<T>;
  using V = ValTy<T>;
  using D = DatabaseTy<T>;

  friend struct CachedQueryInfo<T>;
  friend class Query<T>;

protected:
  CachedQuery(const D *database, const Query<T> *query, K &&key, V &&val,
              DependencySet &&dependencySet)
      : CachedQueryBase(database->getRevision(), std::move(dependencySet)),
        database(database), query(query), key(key), val(val) {}

  virtual bool recompute() final { return recompute(TupleIndexSeq<K>()); }

private:
  friend constexpr llvm::hash_code hash_value(const CachedQuery &cached) {
    return llvm::hash_value(cached.key);
  }

  template <size_t... Is>
  constexpr bool recompute(std::index_sequence<Is...>) {
    dependencySet.clear();
    QueryContext<D> ctx(database, &dependencySet);
    auto val = query->method(ctx, std::get<Is>(key)...);
    if (val == this->val)
      return false;
    this->val = std::move(val);
    return true;
  }

  const D *database;
  const Query<T> *query;
  K key;
  V val;
};

template <typename T>
struct KeyInfo {};

template <typename T>
struct CachedQueryInfo : llvm::DenseMapInfo<CachedQuery<T> *> {
  using llvm::DenseMapInfo<CachedQuery<T> *>::getTombstoneKey;
  using llvm::DenseMapInfo<CachedQuery<T> *>::getEmptyKey;

  static unsigned getHashValue(CachedQuery<T> *cached) {
    if (cached == getEmptyKey() || cached == getEmptyKey())
      return 0;
    return hash_value(*cached);
  }

  static bool isEqual(CachedQuery<T> *lhs, CachedQuery<T> *rhs) {
    if (lhs == getTombstoneKey() || lhs == getEmptyKey())
      return rhs == lhs;
    if (rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return lhs->key == rhs->key;
  }

  static unsigned getHashValue(const KeyTy<T> &key) {
    return llvm::hash_value(key);
  }

  static bool isEqual(const KeyTy<T> &key, CachedQuery<T> *entry) {
    return key == entry->key;
  }

  template <typename... Ts>
  static bool isEqual(const std::tuple<Ts...> &args, CachedQuery<T> *entry) {
    return args == entry->key;
  }
};

template <typename T>
using CachedQueryTable = llvm::DenseSet<CachedQuery<T> *, CachedQueryInfo<T>>;

template <typename T>
using CachedQueryAlloc = llvm::SpecificBumpPtrAllocator<CachedQuery<T>>;

template <typename T>
class Query {
  friend class CachedQuery<T>;

  using K = KeyTy<T>;
  using V = ValTy<T>;
  using D = DatabaseTy<T>;

public:
  template <typename... Args>
  explicit Query(Args &&...args) : method(std::forward<Args>(args)...) {}

  template <typename... Args>
  const V &operator()(Context<D> ctx, Args &&...args) const {
    return get(ctx.getDatabase(), std::forward<Args>(args)...)->val;
  }

  template <typename DX, typename... Args>
  const V &operator()(const D *database, Context<DX> ctx, Args &&...args) {
    return get(database, std::forward<Args>(args)...)->val;
  }

  template <typename... Args>
  const V &operator()(QueryContext<D> ctx, Args &&...args) const {
    auto *result = get(ctx.getDatabase(), std::forward<Args>(args)...);
    ctx.recordQueryDependency(result);
    return result->val;
  }

  template <typename DX, typename... Args>
  const V &operator()(const D *database, QueryContext<DX> ctx,
                      Args &&...args) const {
    auto *result = get(database, std::forward<Args>(args)...);
    ctx.recordQueryDependency(result);
    return result->val;
  }

private:
  template <typename... Args>
  typename CachedQueryTable<T>::iterator lookup(const Args &...args) const {
    return table.find_as(std::tuple<const Args &...>(args...));
  }

  template <typename... Args>
  CachedQuery<T> *get(const D *database, Args &&...args) const {
    auto it = lookup(args...);
    if (it != table.end()) {
      auto *entry = *it;
      entry->refresh(database->getRevision());
      return entry;
    }

    DependencySet dependencySet;
    QueryContext<D> ctx(database, &dependencySet);
    auto val = method(ctx, const_cast<const Args &>(args)...);
    auto entry = new (alloc.Allocate())
        CachedQuery<T>(database, this, K(std::forward<Args>(args)...),
                       std::move(val), std::move(dependencySet));
    table.insert(entry);
    return entry;
  }

  T method;
  mutable CachedQueryTable<T> table;
  mutable CachedQueryAlloc<T> alloc;
};

template <typename T>
Query(const T &) -> Query<T>;
template <typename T>
Query(T &&) -> Query<T>;

} // namespace inc

} // namespace hwml
} // namespace circt

#endif