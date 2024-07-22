add_circt_library(CIRCTHWL
  HWL.cpp
  HWLAst.cpp
  HWLDatabase.cpp
  HWLParser.cpp
  HWLTypecheck.cpp

  LINK_LIBS PUBLIC
  CIRCTHWLParse
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)

add_subdirectory(Incremental)
add_subdirectory(Parse)
add_subdirectory(LSPServer)