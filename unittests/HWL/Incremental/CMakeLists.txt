add_circt_unittest(CIRCTHWLIncrementalTests
  IncrementalTest.cpp
)

target_link_libraries(CIRCTHWLIncrementalTests
  PRIVATE
  CIRCTHWL
  CIRCTHWLIncremental
)
