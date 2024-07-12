#ifndef CIRCT_HWML_HWMLDATABASE_H
#define CIRCT_HWML_HWMLDATABASE_H

#include "circt/HWML/Incremental/Database.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace circt {
namespace hwml {

struct HWMLDatabase;

struct CST {};

struct FileContents {
  std::string contents;
};

CST parseFile(const std::string &contents) { return {}; }

struct GetCST : public QueryFamily<GetCST, HWMLDatabase> {
  static CST compute(HWMLDatabase &database, const FileContents &contents) {
    return parseFile(contents.contents);
  }
};

struct HWMLDatabase {
  // Inputs.
  Input<HWMLDatabase, FileContents> fileContents;

  // Queries.
  GetCST getCST;
  

  // Functions
};

} // namespace hwml
} // namespace circt

#endif