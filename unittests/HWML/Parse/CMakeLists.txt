add_circt_unittest(CIRCTHWMLParseTests
  ParseTest.cpp
)

target_link_libraries(CIRCTHWMLParseTests
  PRIVATE
  CIRCTHWML
)
