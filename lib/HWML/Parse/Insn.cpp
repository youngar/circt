#include "circt/HWML/Parse/Insn.h"

using namespace circt;
using namespace circt::hwml;

void circt::hwml::print(std::ostream &os, const Program &program) {
  ProgramPrinter(program).print(os);
}