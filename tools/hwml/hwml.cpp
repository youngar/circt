#include "circt/HWML/HWMLAst.h"
#include "circt/HWML/HWMLIncremental.h"
#include "circt/HWML/HWMLParser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace mlir;
using namespace circt;
using namespace hwml;
using namespace llvm;

int main(int argc, char *argv[]) {
  cl::OptionCategory mainCategory("HWML Options");
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"), cl::cat(mainCategory));

  cl::ParseCommandLineOptions(argc, argv, "HWML");

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  // SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
  //                                             &context /*, shouldShow */);
  // FileLineColLocsAsNotesDiagnosticHandler addLocs(&context);
  auto buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
  hwml::HWMLParser parser;
  std::vector<Node *> captures;
  std::vector<hwml::Diagnostic> diagnostics;
  MemoTable memoTable;
  parser.parse(buffer, memoTable, captures, diagnostics);

  // for (auto &diag : diagnostics) {
  //   sourceMgr.PrintMessage(llvm::SMLoc::getFromPointer((const char *)diag.sp),
  //                          SourceMgr::DK_Error, diag.getMessage());
  // }

  return EXIT_SUCCESS;
}
