#include "circt/HWML/HWMLParser.h"
#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/Parse/Machine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwml;

namespace {
class Parser {

};

} // namespace

HWMLParser::HWMLParser() {}

bool HWMLParser::parse(StringRef contents, std::vector<Capture> &captures,
                       std::vector<Diagnostic> &diagnostics) {
  const uint8_t *sp = contents.bytes_begin();
  const uint8_t *se = contents.bytes_end();
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  return result;
  return false;
}