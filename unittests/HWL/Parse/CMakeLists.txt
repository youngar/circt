add_circt_unittest(CIRCTHWLParseTests
  MemoTableTest.cpp
  ParseTest.cpp
)

target_link_libraries(CIRCTHWLParseTests
  PRIVATE
  CIRCTHWL
)
