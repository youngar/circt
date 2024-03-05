#include "circt/HWML/LSPServer/HWMLServer.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace circt;
using namespace circt::hwml;
using namespace mlir;
using namespace mlir::lsp;

//===--------------------------------------------------------------------===//
// HWMLDocument
//===--------------------------------------------------------------------===//

HWMLDocument::HWMLDocument(const mlir::lsp::URIForFile &uri, StringRef contents,
                           const std::vector<std::string> &extraDirs,
                           std::vector<mlir::lsp::Diagnostic> &diagnostics) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());

  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
}

//===--------------------------------------------------------------------===//
// HWMLServer
//===--------------------------------------------------------------------===//

HWMLServer::HWMLServer(mlir::DialectRegistry &registry) : registry(registry) {}

HWMLServer::~HWMLServer() = default;

void HWMLServer::addDocument(const URIForFile &uri, StringRef contents,
                             int64_t version,
                             std::vector<lsp::Diagnostic> &diagnostics) {
  Range range(0);
  auto severity = DiagnosticSeverity::Error;
  auto *source = "hwml";
  auto *message = "this is a fake error message";
  std::optional<std::vector<DiagnosticRelatedInformation>> relatedInformation;
  auto *category = "Parsing";
  diagnostics.push_back(
      {range, severity, source, message, relatedInformation, category});
  // Build the set of additional include directories.
  //   std::vector<std::string> additionalIncludeDirs = impl->options.extraDirs;
  //   const auto &fileInfo = impl->compilationDatabase.getFileInfo(uri.file());
  //   llvm::append_range(additionalIncludeDirs, fileInfo.includeDirs);

  //   impl->files[uri.file()] = std::make_unique<PDLTextFile>(
  //       uri, contents, version, additionalIncludeDirs, diagnostics);
}

void HWMLServer::updateDocument(
    const URIForFile &uri, ArrayRef<TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<lsp::Diagnostic> &diagnostics) {
  // Check that we actually have a document for this uri.
  //   auto it = impl->files.find(uri.file());
  //   if (it == impl->files.end())
  //     return;

  //   // Try to update the document. If we fail, erase the file from the
  //   server. A
  //   // failed updated generally means we've fallen out of sync somewhere.
  //   if (failed(it->second->update(uri, version, changes, diagnostics)))
  //     impl->files.erase(it);
}

std::optional<int64_t> HWMLServer::removeDocument(const URIForFile &uri) {
  //   auto it = impl->files.find(uri.file());
  //   if (it == impl->files.end())
  //     return std::nullopt;

  //   int64_t version = it->second->getVersion();
  //   impl->files.erase(it);
  //   return version;
  return 0;
}
