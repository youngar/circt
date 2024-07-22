#include "circt/HWL/HWLParser.h"
#include "circt/HWL/HWLAst.h"
#include "circt/HWL/Parse/Machine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwl;

HWLParser::HWLParser() {

  InsnStream s;

  auto ws = s.label();
  auto lineComment = s.label();
  auto extendedChar = s.label();
  auto charWithContinueBit = s.label();
  auto anySimpleCharExceptNewline = s.label();
  auto decNum = s.label();
  auto hexNum = s.label();
  auto num = s.label();
  auto id = s.label();
  auto atom = s.label();
  auto expression = s.label();
  auto group = s.label();
  auto declaration = s.label();
  auto definition = s.label();
  auto statement = s.label();
  auto file = s.label();

  // std::string alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  using namespace circt::hwl::p;
  // clang-format off
  p::program(file,
    rule(ws, star(anyOf(" \n\t"))),
    rule(lineComment, "#", star(noneOf("\n\r")))),
    rule(nestedComment, "#-")
    rule(id, capture(IdId,
      anyOf(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "_-")),
      p::star(anyOf(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-"))),
    rule(decNum, capture(DecNumId,
      optionally("-"),
      p::plus(p::anyOf("1234567890")))),
    rule(hexNum, capture(HexNumId,
      optionally("-"),
      "0x",
      require("expected digit in number", anyOf("1234567890")),
      p::star(p::anyOf("1234567890")
    ))),
    rule(atom, alt(id, decNum, hexNum, group)),
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
    rule(file, star(ws, statement), ws,
      require("expected declaration or definition", p::failIf(p::any()))))(s);
  // clang-format on
  program = s.finalize();
}

HWLParseResult HWLParser::parse(const HWLDocument &document) {
  auto contents = document.getContents();
  const uint8_t *sp = (const uint8_t *)contents.c_str();
  const uint8_t *se = sp + contents.size();
  std::vector<Node *> captures;
  std::vector<Diagnostic> diagnostics;
  auto result = Machine::parse(program, document.memoTable, sp, se, captures,
                               diagnostics);
  return HWLParseResult{result, std::move(captures), std::move(diagnostics)};
}

void HWLDocument::replaceContents(std::string contents) {
  // The entire document is being replaced, so replace all contents.
  this->contents = std::move(contents);
  memoTable.invalidateAll();
}

LogicalResult HWLDocument::updateContents(const std::string &contents,
                                          Position start, std::size_t removed) {
  // Check that the position of this change is valid for the file.
  assert(start < this->contents.size());
  // Count the number of inserted characters.
  auto inserted = contents.size();
  // Invalidate the memoization table on this range.
  memoTable.invalidate(start, inserted, removed);
}