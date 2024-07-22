#ifndef CIRCT_HWL_HWL_H
#define CIRCT_HWL_HWL_H

#include "circt/HWL/Parse/MemoTable.h"

namespace circt {
namespace hwl {

struct MemoTable;

struct Compiler {

  void parseFile();

private:
  MemoTable memoTable;
};

} // namespace hwl
} // namespace circt
#endif // CIRCT_HWL_HWL_H
