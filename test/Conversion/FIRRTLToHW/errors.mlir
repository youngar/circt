// RUN: circt-opt -lower-firrtl-to-hw -verify-diagnostics %s

firrtl.circuit "UnknownWidth" {
  // COM: Unknown widths are unsupported
  // expected-error @+1 {{cannot lower this port type to HW}}
  firrtl.module @UnknownWidth(in %a: !firrtl.uint) {}
}
