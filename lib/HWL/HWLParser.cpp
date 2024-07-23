#include "circt/HWL/HWLParser.h"
#include "circt/HWL/HWLAst.h"
#include "circt/HWL/Parse/Machine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwl;

HWLParser::HWLParser() {}

HWLParseResult HWLParser::parse(const HWLDocument &document) {
  auto contents = document.getContents();
  const uint8_t *sp = (const uint8_t *)contents.c_str();
  const uint8_t *se = sp + contents.size();
  std::vector<Node *> captures;
  std::vector<Diagnostic> diagnostics;
  // auto result = Machine::parse(program, document.memoTable, sp, se, captures,
  //                              diagnostics);
  return HWLParseResult{true, HWLSyntaxTree(), std::move(diagnostics)};
}

void HWLDocument::replaceContents(const std::string &contents) {
  // The entire document is being replaced, so replace all contents.
  this->contents = contents;
}
