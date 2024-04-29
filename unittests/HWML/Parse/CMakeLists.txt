add_circt_unittest(CIRCTHWMLParseTests
  MemoTableTest.cpp
  ParseTest.cpp
)

target_link_libraries(CIRCTHWMLParseTests
  PRIVATE
  CIRCTHWML
)
