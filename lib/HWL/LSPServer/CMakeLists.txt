add_circt_library(CIRCTHWLLSPServer
  HWLServer.cpp
  LSPServer.cpp

  LINK_LIBS PUBLIC
  CIRCTHWL
  MLIRLspServerSupportLib
  MLIRSupport
)
