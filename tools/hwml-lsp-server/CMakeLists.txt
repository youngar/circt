add_circt_tool(hwml-lsp-server
  hwml-lsp-server.cpp
)

target_link_libraries(hwml-lsp-server PRIVATE
    CIRCTHWMLLSPServer
)
