add_circt_tool(hwl-lsp-server
  hwl-lsp-server.cpp
)

target_link_libraries(hwl-lsp-server PRIVATE
    CIRCTHWLLSPServer
)
