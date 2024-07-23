#ifndef CIRCT_HWL_SERVER_SERVER_H
#define CIRCT_HWL_SERVER_SERVER_H

#include "circt/HWL/HWLParser.h"
#include "circt/HWL/Parse/LineInfoTable.h"
#include "circt/HWL/Parse/MemoTable.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>
#include "circt/HWL/HWLDatabase.h"
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
namespace hwl {

struct HWLServer {

  HWLServer(mlir::DialectRegistry &registry);
  ~HWLServer();

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
  
  HWLDatabase database;

  void run();
};

/// Run the main loop of the LSP server using the given HWL server and
/// transport.
LogicalResult runHWLLSPServer(HWLServer &server,
                               mlir::lsp::JSONTransport &transport);

} // namespace hwl
} // namespace circt

#endif // CIRCT_HWL_SERVER_SERVER_H