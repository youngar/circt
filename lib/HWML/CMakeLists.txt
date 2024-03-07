add_circt_library(CIRCTHWML
  HWML.cpp
  HWMLAst.cpp
  HWMLParser.cpp

  LINK_LIBS PUBLIC
  CIRCTHWMLParse
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)

add_subdirectory(Parse)
add_subdirectory(LSPServer)