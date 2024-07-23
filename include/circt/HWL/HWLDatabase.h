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

  void addDocument(StringRef filename, StringRef contents) {
    auto [_, inserted] = fileTable.try_emplace(
        filename, std::make_unique<inc::Input<HWLDocument>>(contents.str()));
    assert(inserted && "file was already added");
  }

  bool removeDocument(StringRef filename) { return fileTable.erase(filename); }

  inc::Input<HWLDocument> &getDocument(StringRef filename) {
    return *fileTable.at(filename);
  }

  const inc::Input<HWLDocument> &getDocument(StringRef filename) const {
    return *fileTable.at(filename);
  }

private:
  HWLParser parser;
  DenseMap<StringRef, std::unique_ptr<inc::Input<HWLDocument>>> fileTable;
};

using Context = inc::Context<HWLDatabase>;
using QueryContext = inc::QueryContext<HWLDatabase>;

} // namespace hwl
} // namespace circt

#endif