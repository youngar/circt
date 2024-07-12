add_circt_library(CIRCTHWMLIncremental
  Database.cpp

  LINK_LIBS PUBLIC
  CIRCTHWML
  MLIRLspServerSupportLib
  MLIRSupport
)

