#ifndef CIRCT_HWML_SERVER_SERVER_H
#define CIRCT_HWML_SERVER_SERVER_H

#include "circt/HWML/HWMLParser.h"
#include "circt/HWML/Parse/LineInfoTable.h"
#include "circt/HWML/Parse/MemoTable.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <optional>

namespace mlir {
class DialectRegistry;
namespace lsp {
struct MLIRServer;
struct JSONTransport;
struct CodeAction;
struct CodeActionContext;
struct CompletionList;
struct Diagnostic;
struct DocumentSymbol;
struct Hover;
struct Location;
struct TextDocumentContentChangeEvent;
struct MLIRConvertBytecodeResult;
struct Position;
struct Range;
struct Diagnostic;
class URIForFile;
} // namespace lsp
} // namespace mlir

namespace circt {
namespace hwml {

struct HWMLDocument {
  HWMLDocument(const mlir::lsp::URIForFile &uri, int64_t version,
               StringRef contents,
               std::vector<mlir::lsp::Diagnostic> &diagnostics);
  HWMLDocument(const HWMLDocument &) = delete;
  HWMLDocument &operator=(const HWMLDocument &) = delete;

  void update(const mlir::lsp::URIForFile &uri, int64_t version,
              ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
              std::vector<mlir::lsp::Diagnostic> &diagnostics);

  int64_t getVersion() const { return version; }

private:
  HWMLParser parser;
  /// The full string contents of the file.
  std::string contents;
  /// The version of this file.
  int64_t version;
  /// Memoization of the AST.
  MemoTable memoTable;
  /// Mapping of file offsets to their (line, coloumn) information.
  LineInfoTable lineInfoTable;
};

struct HWMLServer {

  HWMLServer(mlir::DialectRegistry &registry);
  ~HWMLServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void addDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                   int64_t version,
                   std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void
  updateDocument(const mlir::lsp::URIForFile &uri,
                 ArrayRef<mlir::lsp::TextDocumentContentChangeEvent> changes,
                 int64_t version,
                 std::vector<mlir::lsp::Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  std::optional<int64_t> removeDocument(const mlir::lsp::URIForFile &uri);

private:
  /// The registry containing dialects that can be recognized in parsed .mlir
  /// files.
  mlir::DialectRegistry &registry;

  llvm::StringMap<std::unique_ptr<HWMLDocument>> files;

  void run();
};

/// Run the main loop of the LSP server using the given HWML server and
/// transport.
LogicalResult runHWMLLSPServer(HWMLServer &server,
                               mlir::lsp::JSONTransport &transport);

} // namespace hwml
} // namespace circt

#endif // CIRCT_HWML_SERVER_SERVER_H