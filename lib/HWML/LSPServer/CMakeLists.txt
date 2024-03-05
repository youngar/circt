add_circt_library(CIRCTHWMLLSPServer
  HWMLServer.cpp
  LSPServer.cpp

  LINK_LIBS PUBLIC
  CIRCTHWML
  MLIRLspServerSupportLib
  MLIRSupport
)
