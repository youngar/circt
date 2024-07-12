#ifndef CIRCT_HWML_INCREMENTAL_DATABASE_H
#define CIRCT_HWML_INCREMENTAL_DATABASE_H

#include "circt/Support/LLVM.h"
#include <cstdint>
#include <functional>

namespace circt {
namespace hwml {

///
/// Revision
///

struct Revision {
  Revision();

  Revision &operator++() {
    ++value;
    return *this;
  }

  bool operator<=(Revision rhs) const { return value <= rhs.value; }

  bool operator==(Revision rhs) const { return value == rhs.value; }

private:
  std::size_t value = 0;
};

///
/// Database
///

struct AbstractDatabase {

  void recordUpdate() { ++revision; }

  Revision getRevision() { return revision; }

private:
  Revision revision;
};

///
/// Function
///

// template <typename Database, typename RetTy, typename... Args>
// struct Function {
//   using KeyTy = std::tuple<Args...>;

//   RetTy run(FunctionCtx<Database> ctx, Args... args) {}

//   RetTy operator()(FunctionCtx<Database> ctx, Args... args) {}

//   DenseMap<KeyTy, RetTy> memoization;
// };

///
/// Input
///

struct AbstractInput {
protected:
  Revision getLastUpdated() const { return lastUpdated; }
  bool hasBeenUpdated(Revision revision) const {
    return getLastUpdated() <= revision;
  }
  void setLastUpdated(Revision lastUpdated) { this->lastUpdated = lastUpdated; }

private:
  Revision lastUpdated;
};

template <typename Database, typename Value>
struct Input : public AbstractInput {

  const Value &getValue() const { return value; }

  void setValue(const Database &database, const Value &value) {
    setLastUpdated(database.getRevision());
    this->value = value;
  }

  void setValue(const Database &database, Value &&value) {
    setLastUpdated(database.getRevision());
    this->value = std::move(value);
  }

private:
  Value value;
};

///
/// Queries
///

template <typename Database, typename QueryData>
struct QueryCtx {

  template <typename Query>
  QueryCtx(Database &database, Query &query) {}

  // Deduction guideline.
  template <typename Query>
  QueryCtx(Database &database, Query &query)->typename Query::QueryResultType;

  QueryCtx(Database &database, QueryData &queryData)
      : database(database), queryData(queryData) {}

  template <typename T>
  auto getInput(T input) {
    queryData.recordInput(input);
    return input.getValue();
  }

  template <typename Q>
  auto getQuery(Q query) {}

private:
  Database &database;
  QueryData &queryData;
};

template <typename Database>
struct AbstractQuery {

  Revision getLastChecked() { return lastChecked; }
  void setLastChecked(Revision lastChecked) { this->lastChecked = lastChecked; }

  void recordInput(AbstractInput *input) { inputs.emplace_back(input); }
  void recordQuery(AbstractQuery *query) { queries.emplace_back(query); }

  /// Rerun the query.  If the result changed, return true, false otherwise.
  virtual bool recompute(Database &database) = 0;

  bool refresh(Database &database) {
    auto revision = database.getRevision();
    assert(getLastChecked() <= revision);

    // If we have already calculated the query for this revision, we can return
    // it now.
    if (getLastChecked() == revision)
      return false;

    // We may need to recalculate our query, if we depend on any inputs which
    // changed.
    for (auto *input : inputs)
      if (getLastChecked() < input->getLastUpdated())
        return recompute(database);

    // Check if any subqueries were updated.
    for (auto *query : queries)
      if (query->recalculate(database))
        return recompute(database);

    return false;
  }

  Revision lastChecked;
  std::vector<AbstractInput *> inputs;
  std::vector<AbstractQuery *> queries;

private:
};

template <typename Database, typename QueryFamily, typename Ret,
          typename... Args>
struct Query : public AbstractQuery<Database> {

  using RetTy = Ret;
  using KeyTy = std::tuple<Args...>;

  template <typename... Ts>
  Query(const Ret &result, Ts &&...args)
      : key(std::forward<Ts>(args)...), result(result) {}

  template <typename... Ts>
  Query(Database &database, Ret &&result, Ts &&...args)
      : key(std::forward<Ts>(args)...), result(std::move(result)) {}

  bool recompute(Database &database) override {
    inputs.clear();
    queries.clear();
    auto newResult = QueryFamily::run(database, key);
    if (newResult == result)
      return false;
    result = std::move(newResult);
    return true;
  }

  RetTy getCachedResult() { return result; }

private:
  Revision revision;
  KeyTy key;
  RetTy result;
};

template <typename Self, typename Database, typename Signature>
struct QueryFamily;

template <typename Self, typename Database, typename Ret, typename... Args>
struct QueryFamily<Self, Database, Ret(Args...)> {

  using RetTy = decl_type(Self::compute(declval));
  using RetTy = decl_type(Self::compute(declval));

  using RetTy = Ret;
  using KeyTy = std::tuple<Args...>;
  using QueryTy = Query<Database, QueryFamily, Ret, Args...>;

  Ret get(Database database, Args... args) {
    auto oldQuery = queries.find(args...);
    if (oldQuery != queries.end()) {
      // We have run this query before.  Refresh the result and return it.
      oldQuery->refresh();
      return oldQuery->getCachedValue();
    }
    // We have never run this query before.
    auto result = run(database, args...);
    auto &[newQuery, inserted] = queries.try_emplace(result, args...);
    assert(inserted && "recursive query");
    return newQuery->getCachedValue();
  }

  static Ret run(Database &database, KeyTy &key) {
    return std::apply(Self::compute,
                      std::tuple_cat(std::make_tuple(database), key));
  }

  static Ret run(Database &database, Args... args) {
    return std::apply(Self::compute, std::make_tuple(database, args...));
  }

private:
  DenseSet<QueryTy> queries;
};

} // namespace hwml
} // namespace circt

#endif