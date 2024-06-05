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
  auto atom = s.label();
  auto expression = s.label();
  auto group = s.label();
  auto declaration = s.label();
  auto definition = s.label();
  auto statement = s.label();
  auto file = s.label();

  enum CaptureId {
    IdId,
    NumId,
    ExprId,
    DeclId,
    DefId,
    StmtId,
    TrailingId,
  };

  using namespace circt::hwml::p;

  // clang-format off
  p::program(file,
    rule(ws, star(p::set(" \n\t"))),
    rule(id, capture(IdId, p::plus(p::set("abcdefghijklmnopqrstuvwxyz")))),
    rule(num, capture(NumId, p::plus(p::set("1234567890")))),
    rule(atom, alt(id, num, group)),
    rule(expression, capture(ExprId, atom, star(ws, atom))),
    rule(group,
      "(",
      require("require expression inside group", expression),
      require("missing closing parenthesis", ")")),
    rule(declaration, capture(DeclId, 
      id, ws, ":", ws,
      require("missing type expression", expression), ws)),
    rule(definition, capture(DefId,
      p::plus(id), ws, "=", ws, expression)),
    rule (statement, capture(StmtId,
      alt(definition, declaration),
      ws,
      require("missing semicolon", ";"))),
    rule(file, star(ws, statement), ws, require("expected declaration or definition", p::failIf(p::any()))))(s);
  // clang-format on
  program = s.finalize();
}

bool HWMLParser::parse(StringRef contents, MemoTable &memoTable,
                       std::vector<Node *> &captures,
                       std::vector<Node *> &diagnostics) {
  const uint8_t *sp = contents.bytes_begin();
  const uint8_t *se = contents.bytes_end();
  auto result =
      Machine::parse(program, memoTable, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  return result;
}