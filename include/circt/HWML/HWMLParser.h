#ifndef CIRCT_HWML_HWMLPARSER_H
#define CIRCT_HWML_HWMLPARSER_H

#include "circt/HWML/Parse/Machine.h"
#include "circt/HWML/Parse/MemoTable.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace hwml {

struct HWMLParser {
  HWMLParser();

  bool parse(StringRef contents, std::vector<Capture> &captures,
             std::vector<Diagnostic> &diagnostics);
private:
  MemoTable memoTable;
  Program program;
};

} // namespace hwml
} // namespace circt
#endif
