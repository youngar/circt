# This command can be used to regenerate the SystemVerilog for all regress tests
# in the event that the FIRRTL compiler is no longer passing formal
# equivalence.  These regenerated tests should be checked-in and included in
# the behaviour changing pull request.
add_custom_target(circt-regenerate-gold-tests)

set(REGRESS_SOURCES
  AddNot.fir
  FPU.fir
  HwachaSequencer.fir
  ICache.fir
  Ops.fir
  Rob.fir
  RocketCore.fir
)

function(add_fir_source source output)
  add_custom_command(
    TARGET circt-regenerate-gold-tests POST_BUILD
    COMMAND firtool "-lowering-options=disallowLocalVariables,disallowPackedArrays,noAlwaysComb" -o "${output}" "${source}"
    VERBATIM
  )
endfunction(add_fir_source)


foreach(source ${REGRESS_SOURCES})
  set(abs_source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  cmake_path(REPLACE_EXTENSION abs_source
             LAST_ONLY ".sv"
             OUTPUT_VARIABLE output)
  add_fir_source("${abs_source}" "${output}")
  #add_dependencies(circt-regenerate-gold-tests "${output}")
endforeach()
