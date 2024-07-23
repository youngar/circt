#include "circt/HWL/LSPServer/HWLServer.h"
#include "circt/HWL/HWLDatabase.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace circt;
using namespace circt::hwl;
using namespace mlir;
using namespace mlir::lsp;

//===--------------------------------------------------------------------===//
// HWLDocument
//===--------------------------------------------------------------------===//

static std::vector<mlir::lsp::Diagnostic>
convertDiagnosticsToLSPDiagnostics(const LineInfoTable &lineInfoTable,
                                   std::vector<hwl::Diagnostic> &diagnostics) {
  if (diagnostics.empty())
    return {};
  std::vector<lsp::Diagnostic> lspDiagnostics;
  lspDiagnostics.reserve(diagnostics.size());
  for (const auto &diag : diagnostics) {
    lsp::Diagnostic lspDiag;
    lspDiag.source = "hwl";
    lspDiag.category = "Parse Error";
    auto [line, column] =
        lineInfoTable.getLineAndColumnForOffset(diag.getPosition());
    lspDiag.range = lsp::Range(lsp::Position(line, column));
    lspDiag.message = diag.getMessage();
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    lspDiagnostics.emplace_back(std::move(lspDiag));
  }
  return lspDiagnostics;
}

// HWLDocument::HWLDocument(const lsp::URIForFile &uri, int64_t version,
//                          StringRef contents,
//                          std::vector<lsp::Diagnostic> &diagnostics)
//     : contents(contents.str()), version(version), lineInfoTable(contents) {

//   std::vector<Node *> caps;
//   std::vector<Diagnostic> diags;
//   parser.parse(contents, memoTable, caps, diags);
//   lineInfoTable.update(0, contents, 0);
//   diagnostics = convertDiagnosticsToLSPDiagnostics(lineInfoTable, diags);
// }

// void HWLDocument::update(const lsp::URIForFile &uri, int64_t version,
//                          ArrayRef<lsp::TextDocumentContentChangeEvent>
//                          changes, std::vector<lsp::Diagnostic> &diagnostics)
//                          {

//   if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes,
//   contents))) {
//     lsp::Logger::error("Failed to update contents of {0}", uri.file());
//     return;
//   }

//   for (const auto &change : changes) {
//     if (change.range) {
//       auto &range = *change.range;
//       auto &startPos = range.start;
//       // Get the offset of the start of the change.
//       auto start =
//           lineInfoTable.getOffsetForLineCol(startPos.line,
//           startPos.character);
//       // Get the amount of text inserted.
//       auto inserted = change.text.size();
//       // Get the amount of text removed or replaced.
//       auto removed = *change.rangeLength;
//       // Invalidate all parses which intersect the updated section of text.
//       memoTable.invalidate(start, inserted, removed);
//       // Update the line information.
//       lineInfoTable.update(start, change.text, removed);
//     } else {
//       // If there is no position, then that indicates the whole document
//       // changed. We invalidate all parses and recompute all line
//       information. memoTable.invalidateAll(); lineInfoTable.update(contents);
//     }
//   }

//   std::vector<Node *> caps;
//   std::vector<Diagnostic> diags;
//   // Repase the file.
//   parser.parse(contents, memoTable, caps, diags);
//   // Return the new list of diagnostics.
//   diagnostics = convertDiagnosticsToLSPDiagnostics(lineInfoTable, diags);
// }

//===--------------------------------------------------------------------===//
// HWLServer
//===--------------------------------------------------------------------===//

HWLServer::HWLServer(mlir::DialectRegistry &registry) : registry(registry) {}

HWLServer::~HWLServer() = default;

void HWLServer::addDocument(const URIForFile &uri, StringRef contents,
                            int64_t version,
                            std::vector<lsp::Diagnostic> &diagnostics) {
  database.addDocument(uri.file().str(), contents.str());
}

void HWLServer::updateDocument(const URIForFile &uri,
                               ArrayRef<TextDocumentContentChangeEvent> changes,
                               int64_t version,
                               std::vector<lsp::Diagnostic> &diagnostics) {

  Context context(&database);
  auto &document = database.getDocument(uri.file()).modify(context);
  auto contents = document.getContents();
  if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes, contents))) {
    lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return;
  }

  // Update the file.
  document.replaceContents(contents);
}

std::optional<int64_t> HWLServer::removeDocument(const URIForFile &uri) {
  return std::nullopt;
}
