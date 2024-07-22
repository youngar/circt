set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(hwl
  hwl.cpp

  DEPENDS
  SUPPORT_PLUGINS
)
llvm_update_compile_flags(hwl)
target_link_libraries(hwl PRIVATE
  CIRCTHWL
  
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
