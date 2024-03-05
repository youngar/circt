#ifndef CIRCT_HWML_HWML_H
#define CIRCT_HWML_HWML_H

#include "circt/HWML/Parse/MemoTable.h"

namespace circt {
namespace hwml {

struct MemoTable;

struct Compiler {

  void parseFile();

private:
  MemoTable memoTable;
};

} // namespace hwml
} // namespace circt
#endif // CIRCT_HWML_HWML_H
