#include "circt/HWML/HWMLParser.h"
#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/Parse/Machine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwml;

namespace {
class Parser {};

} // namespace

HWMLParser::HWMLParser() {

  InsnStream s;

  auto ws = s.label();
  auto num = s.label();
  auto id = s.label();
  auto expression = s.label();
  auto declaration = s.label();
  auto definition = s.label();
  auto file = s.label();

  using namespace circt::hwml::p;
  // clang-format off
  p::program(file,
    rule(ws, star(set(" \n\t"))),
    rule(id, plus(set("abcdefghijklmnopqrstuvwxyz"))),
    rule(num, plus(set("1234567890"))),
    rule(expression, plus(alt(id, num)), ws, expression),
    rule(declaration, id, ws, ":", ws, expression, ws, ";"),
    rule(definition, plus(id), ws, "=", ws, expression, ";"),
    rule(file, ws, star(alt(declaration, definition))))(s);
  program = s.finalize();
  // clang-format on
}

bool HWMLParser::parse(StringRef contents, std::vector<Capture> &captures,
                       std::vector<Diagnostic> &diagnostics) {
  const uint8_t *sp = contents.bytes_begin();
  const uint8_t *se = contents.bytes_end();
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  return result;
}