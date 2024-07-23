#ifndef CIRCT_HWL_HWLPARSER_H
#define CIRCT_HWL_HWLPARSER_H

#include "circt/HWL/Incremental/Database.h"
#include "circt/HWL/Parse/LineInfoTable.h"
#include "circt/HWL/Parse/Machine.h"
#include "circt/HWL/Parse/MemoTable.h"
#include "circt/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace hwl {

struct HWLParser;
struct HWLDocument;

struct HWLSyntaxTree {

};

struct HWLParseResult {
  bool success;
  HWLSyntaxTree tree;
  std::vector<Diagnostic> diagnostics;
};

struct HWLParser {

  enum CaptureId {
    IdId,
    HexNumId,
    DecNumId,
    ExprId,
    DeclId,
    DefId,
    StmtId,
    TrailingId,
  };

  HWLParser();

  HWLParseResult parse(const HWLDocument &document);

private:
  Program program;
};

struct HWLDocument {
  HWLDocument(const std::string &contents) : contents(contents) {}
  HWLDocument(const HWLDocument &) = delete;
  HWLDocument &operator=(const HWLDocument &) = delete;

  std::string getContents() const { return contents; }
  void replaceContents(const std::string &contents);

private:
  friend HWLParser;

  /// The full string contents of the file.
  std::string contents;
};

} // namespace hwl
} // namespace circt
#endif
