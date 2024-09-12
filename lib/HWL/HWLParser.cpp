#include "circt/HWL/HWLParser.h"
#include "circt/HWL/HWLAst.h"
#include "llvm/ADT/AllocatorList.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::hwl;
using namespace circt::hwl::cst;

llvm::raw_ostream &dump(llvm::raw_ostream &stream, File file) { return stream; }

namespace {

using Position = const unsigned char *;

struct Parser {

  static HWLParseResult parse(StringRef contents) {
    Parser parser(contents);
    return parser.run();
  }

private:
  Parser(StringRef contents)
      : contents(contents), position(contents.bytes_begin()),
        end(contents.bytes_end()) {}

  bool atEnd() { return position == end; }

  ParseResult parseStatement() { retern success(); }

  HWLParseResult run() {
    // while (!atEnd()) {
    //   if (parseStatement())

    // }
    return HWLParseResult{true, File(), {}};
  }

  // llvm::ScopedHashTable<StringRef, e> identifiers;
  StringRef contents;

  Position position;
  Position end;

  /// The current column.
  std::size_t column;
  /// The current line.
  std::size_t line;
};

} // namespace

HWLParser::HWLParser() = default;

HWLParseResult HWLParser::parse(const HWLDocument &document) {
  auto contents = document.getContents();
  const uint8_t *sp = (const uint8_t *)contents.c_str();
  const uint8_t *se = sp + contents.size();
  std::vector<Diagnostic> diagnostics;
  return Parser::parse(document.getContents());
}

void HWLDocument::replaceContents(const std::string &contents) {
  // The entire document is being replaced, so replace all contents.
  this->contents = contents;
}
