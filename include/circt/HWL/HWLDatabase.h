#ifndef CIRCT_HWL_HWLDATABASE_H
#define CIRCT_HWL_HWLDATABASE_H

#include "circt/HWL/HWLParser.h"
#include "circt/HWL/Incremental/Database.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace circt {
namespace hwl {

struct HWLDocument;
struct HWLDatabase;
struct HWLParser;

struct HWLDatabase : public inc::Database {

  void addDocument(const std::string &filename, const std::string &contents) {
    auto [_, inserted] = fileTable.try_emplace(filename, contents);
    assert(inserted && "file was already added");
  }

  bool removeDocument(const std::string &filename) {
    return fileTable.erase(filename);
  }

  inc::Input<HWLDocument> &getDocument(const std::string &filename) {
    return *fileTable.at(filename);
  }

  const inc::Input<HWLDocument> &
  getDocument(const std::string &filename) const {
    return *fileTable.at(filename);
  }

private:
  HWLParser parser;
  DenseMap<std::string, std::unique_ptr<inc::Input<HWLDocument>>> fileTable;
};

using Context = inc::Context<HWLDatabase>;
using QueryContext = inc::QueryContext<HWLDatabase>;

} // namespace hwl
} // namespace circt

#endif