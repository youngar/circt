set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(hwml
  hwml.cpp

  DEPENDS
  SUPPORT_PLUGINS
)
llvm_update_compile_flags(hwml)
target_link_libraries(hwml PRIVATE
  CIRCTHWML
  
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
