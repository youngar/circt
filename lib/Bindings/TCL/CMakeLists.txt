
add_library (CIRCTTCL
    SHARED
    hello.c
    )

# Force no prefix on the library name.
set_target_properties (CIRCTTCL PROPERTIES PREFIX "")

target_compile_definitions(CIRCTTCL PUBLIC -DUSE_TCL_STUBS)
target_link_libraries(CIRCTTCL PUBLIC ${TCL_STUB_LIBRARY})
target_include_directories(CIRCTTCL PUBLIC ${TCL_INCLUDE_PATH})

