#include "circt/HWML/HWMLParser.h"
#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/Parse/Machine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwml;

struct AstKinds {
  enum AstKind {
    Number,
    Identifier,
    Apply,
    Error,
    Decl,
    Def,
    Program,
  };
};

HWMLParser::HWMLParser() {

  InsnStream s;
  auto ws = s.label();
  auto wsp = s.label();
  auto num = s.label();
  auto id = s.label();
  auto parens = s.label();
  // auto apply = s.label();
  auto basicExpr = s.label();
  auto expr = s.label();
  auto decl = s.label();
  auto stmt = s.label();
  auto program = s.label();

  using namespace p;
  // clang-format off
  p::program(program, rule(ws, star(set("\t\n\r "))),
    rule(wsp, plus(set("\t\n\r "))),
    rule(num, memo(AstKinds::Number,
        plus(set("1234567890")))),
    rule(id, memo(AstKinds::Identifier,
        plus(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")))),
    rule(parens, "(", ws, expr, ws, require("expected \")\"", ")")),
    rule(basicExpr, alt(parens, num, id)),
    rule(expr, basicExpr, star(wsp, basicExpr)),
    rule(decl, memo(AstKinds::Decl, "def", wsp, id, wsp,
      "=", wsp, expr, ws,
      require("expected statement to terminate in \".\"", "."))),
     rule(stmt, decl),
     rule(program, star(ws, stmt))
  )(s);
  this->program = s.finalize();
  print(std::cerr, this->program);
  // clang-format on

  this->program = s.finalize();
  print(std::cerr, this->program);
}

bool HWMLParser::parse(StringRef contents, std::vector<Capture> &captures,
                       std::vector<Diagnostic> &diagnostics) {
  const uint8_t *sp = contents.bytes_begin();
  const uint8_t *se = contents.bytes_end();
  auto result = Machine::parse(program, sp, se, captures, diagnostics);
  print(std::cerr, captures);
  return result;
  return false;
}