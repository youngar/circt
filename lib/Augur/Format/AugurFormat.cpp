
#include "augur/Format/AugurFormat.h"
#include "circt/Support/PrettyPrinterHelpers.h"

using namespace circt::pretty;
using namespace aug;

namespace {}

struct Printer {
  void print(Object *object) {}
  PrettyPrinter pp;
};

void print(Object *object) {}