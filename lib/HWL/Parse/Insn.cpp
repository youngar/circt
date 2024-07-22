#include "circt/HWL/Parse/Insn.h"

using namespace circt;
using namespace circt::hwl;

void circt::hwl::print(std::ostream &os, const Program &program) {
  ProgramPrinter(program).print(os);
}