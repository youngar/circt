#include "circt/HWML/LSPServer/HWMLServer.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"

#include <iostream>

using namespace mlir;
using namespace mlir::lsp;
using namespace circt;
using namespace circt::hwml;

int main(int argc, char *argv[]) {
  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual LSP protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Info),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "HWML LSP Language Server");

  if (litTest) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  DialectRegistry registry;

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  JSONTransport transport(stdin, llvm::outs(), inputStyle, prettyPrint);

  // Configure the servers and start the main language server.
  HWMLServer server(registry);
  return failed(runHWMLLSPServer(server, transport));
}
