# REQUIRES: bindings_tcl
# RUN: %TCL% %s %shlibdir | FileCheck %s

set lib [file join [lindex $argv 0] CIRCTTCL[info sharedlibextension]]

puts "loading library: $lib"
load $lib

puts "running:"

# CHECK: Hello, World!
puts [hello]

