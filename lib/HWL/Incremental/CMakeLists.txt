add_circt_library(CIRCTHWLIncremental
  Database.cpp

  LINK_LIBS PUBLIC
  CIRCTHWL
  MLIRLspServerSupportLib
  MLIRSupport
)

