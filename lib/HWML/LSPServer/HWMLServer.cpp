#include "circt/HWML/LSPServer/HWMLServer.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace circt;
using namespace circt::hwml;
using namespace mlir;
using namespace mlir::lsp;

//===--------------------------------------------------------------------===//
// HWMLDocument
//===--------------------------------------------------------------------===//

namespace {
/// A table mapping file offsets into lines and column positions.
///
class LineInfo {
public:
  LineInfo() {
    // The start of the first line is implicit.
    store(0);
  }

  LineInfo(StringRef content) : LineInfo() {
    for (auto i = content.begin(), e = content.end(); i != e; ++i) {
      if (*i == '\n')
        store(std::distance(content.begin(), i));
    }
  }

  void update(std::size_t offset, StringRef content, std::size_t removed) {
    // Create a list of all the new lines inserted.
    std::vector<std::size_t> inserted;
    for (auto i = content.begin(), e = content.end(); i != e; ++i) {
      if (*i == '\n')
        inserted.push_back(offset + std::distance(content.begin(), i));
    }

    auto line = getLineForOffset(offset);
    auto insertedAmt = inserted.size();
    auto removedAmt = line - getLineForOffset(offset + removed);
    // ssize_t lineDelta = insertedAmt - removedAmt;
    auto contentDelta = content.size() - removed;

    if (insertedAmt > removedAmt) {
      for (auto i = 0ul; i < removedAmt; ++i)
        breaks[i + line] = inserted[i];

      // Insert any extra elements.
      breaks.insert(breaks.begin() + line + removedAmt,
                    inserted.begin() + removedAmt, inserted.end());

    } else if (removedAmt > insertedAmt) {
      auto delta = removedAmt - insertedAmt;
      for (auto i = 0ul; i < insertedAmt; ++i)
        breaks[i + line] = inserted[i];

      // Remove any lost elements.
      breaks.erase(breaks.begin() + line + insertedAmt,
                   breaks.begin() + line + insertedAmt + delta);
    }

    // Update the offsets of any trailing elements.
    for (auto i = breaks.begin() + line + insertedAmt, e = breaks.end(); i != e;
         ++i)
      *i += contentDelta;
  }

  /// Get the start offset of a line.
  std::size_t getOffsetForLine(std::size_t n) const { return breaks[n]; }

  /// Get the end offset of a line.
  std::size_t getOffsetForLineEnd(std::size_t n) const { return breaks[n + 1]; }

  /// True if offset is within line number n.
  bool isOffsetInLine(std::size_t offset, std::size_t n) const {
    return getOffsetForLine(n) <= offset && offset < getOffsetForLineEnd(n);
  }

  /// Find the line number of a position. Lines are 0-indexed.
  std::size_t getLineForOffset(std::size_t offset) const {
    const auto n = breaks.size();

    // If the offset is _after_ the last line break, the offset occurs on
    // the last line. The last line has no upper-limit, so we have to treat
    // it specially.
    if (breaks[n - 1] <= offset) {
      return n - 1;
    }

    // Otherwise, search for the line who's interval contains the offset.
    std::size_t l = 0;
    std::size_t r = n - 1;
    while (l <= r) {
      std::size_t m = (l + r) / 2;
      if (getOffsetForLineEnd(m) < offset + 1) {
        l = m + 1;
      } else if (offset < getOffsetForLine(m)) {
        r = m - 1;
      } else {
        return m;
      }
    }
    llvm_unreachable("should not reach here");
    return 0;
  }

  /// Find the column number of a position. Columns are 0-indexed.
  std::size_t getColumnForOffset(std::size_t o) const {
    return o - getOffsetForLine(getLineForOffset(o));
  };

  std::pair<std::size_t, std::size_t>
  getLineAndColumnForOffset(std::size_t o) const {
    auto line = getLineForOffset(o);
    auto column = o - getOffsetForLine(line);
    return {line, column};
  }

private:
  /// Note a linebreak at position p.
  void store(std::size_t offset) { breaks.push_back(offset); }

  std::vector<std::size_t> breaks;
};

} // namespace

static std::vector<mlir::lsp::Diagnostic>
convertDiagnosticsToLSPDiagnostics(StringRef buffer,
                                   std::vector<hwml::Diagnostic> &diagnostics) {
  if (diagnostics.empty())
    return {};
  LineInfo lineInfo(buffer);
  std::vector<lsp::Diagnostic> lspDiagnostics;
  lspDiagnostics.resize(diagnostics.size());
  for (const auto &diag : diagnostics) {
    lsp::Diagnostic lspDiag;
    lspDiag.source = "hwml";
    lspDiag.category = "Parse Error";
    auto offset = buffer.bytes_begin() - diag.sp;
    auto [line, column] = lineInfo.getLineAndColumnForOffset(offset);
    lspDiag.range = {line, column};
    lspDiag.message = diag.message;
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    lspDiagnostics.emplace_back(std::move(lspDiag));
  }
  return lspDiagnostics;
}

HWMLDocument::HWMLDocument(const lsp::URIForFile &uri, int64_t version,
                           StringRef contents,
                           std::vector<lsp::Diagnostic> &diagnostics)
    : contents(contents.str()), version(version) {

  std::vector<Capture> caps;
  std::vector<Diagnostic> diags;
  parser.parse(contents, caps, diags);
  diagnostics = convertDiagnosticsToLSPDiagnostics(contents, diags);
}

void HWMLDocument::update(const lsp::URIForFile &uri, int64_t version,
                          ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                          std::vector<lsp::Diagnostic> &diagnostics) {

  if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes, contents))) {
    lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return;
  }

  // for (const auto &change : changes) {
  //   if (change.range) {
  //     auto range = *change.range;
  //     auto position = range.start;
  //     auto inserted = change.text.size();
  //     auto removed = range.end.
  //     memoTable.invalidate(range.start, change.text.size(),
  //                          range.end - range.start);
  //   } else {
  //     // TODO: the whole document changed, invalidate everything.
  //   }
}

//===--------------------------------------------------------------------===//
// HWMLServer
//===--------------------------------------------------------------------===//

HWMLServer::HWMLServer(mlir::DialectRegistry &registry) : registry(registry) {}

HWMLServer::~HWMLServer() = default;

void HWMLServer::addDocument(const URIForFile &uri, StringRef contents,
                             int64_t version,
                             std::vector<lsp::Diagnostic> &diagnostics) {
  files[uri.file()] =
      std::make_unique<HWMLDocument>(uri, version, contents, diagnostics);
}

void HWMLServer::updateDocument(
    const URIForFile &uri, ArrayRef<TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<lsp::Diagnostic> &diagnostics) {

  // Check that we has this document open.
  auto it = files.find(uri.file());
  if (it == files.end())
    return;

  // Update the file.
  it->second->update(uri, version, changes, diagnostics);
}

std::optional<int64_t> HWMLServer::removeDocument(const URIForFile &uri) {
  auto it = files.find(uri.file());
  if (it == files.end())
    return std::nullopt;

  auto version = it->second->getVersion();
  files.erase(it);
  return version;
}
