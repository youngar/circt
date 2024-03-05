#ifndef CIRCT_HWML_PARSE_DATABASE_H
#define CIRCT_HWML_PARSE_DATABASE_H

#include <cstdint>

namespace inc {

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

struct Database {
  Context *getContext();

  Revision getRevision();

  template <typename ReturnT>
  ReturnT get() {}

private:
  Revision revision;
};

struct Context {
  Database *getDatabase();

private:
  Database *database;
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

struct Query {};

/// Task

/// Implementation
Revision::Revision() : revision(0) {}
Revision::operator uintptr_t() { return revision; }

Context *Database::getContext() { return new Context(); }

Revision Database::getRevision() { return revision; }

Database *Context::getDatabase() { return database; }

} // namespace inc
#endif