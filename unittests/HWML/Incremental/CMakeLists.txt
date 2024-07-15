add_circt_unittest(CIRCTHWMLIncrementalTests
  IncrementalTest.cpp
)

target_link_libraries(CIRCTHWMLIncrementalTests
  PRIVATE
  CIRCTHWML
  CIRCTHWMLIncremental
)
