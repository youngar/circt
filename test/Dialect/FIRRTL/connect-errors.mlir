
// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-note @+1 {{the left-hand-side was defined here}}
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{has invalid flow: the left-hand-side has source flow}}
  firrtl.connect %a, %b : !firrtl.uint<1>, !firrtl.uint<1>
}
}

/// Analog types cannot be connected and must be attached.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.analog, out %b : !firrtl.analog) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.analog, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<a: analog>, out %b : !firrtl.bundle<a: analog>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.bundle<a: analog>, !firrtl.bundle<a: analog>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.analog, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.analog) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.analog, !firrtl.uint<1>
}
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.reset, out %b : !firrtl.uint<2>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<2>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.reset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.reset, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.reset
}
}

// Reset destination.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<2>, out %b : !firrtl.reset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.uint<2>'}}
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.uint<2>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.reset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.sint<1>
}
}

/// Ground types can be connected if they are the same ground type.

// UInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {

firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.uint<1>
}

}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.uint<1>
}
}

// SInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.sint<1>
}
}

// Clock source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.clock
}
}

// AsyncReset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.asyncreset
}
}

/// Vector types can be connected if they have the same size and element type.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<uint<1>, 2>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<uint<1>, 2>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 3>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<sint<1>, 3>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<sint<1>, 3>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.vector<sint<1>, 3>, !firrtl.vector<uint<1>, 3>
}
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f2 flip: uint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f2 flip: uint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f2 flip: uint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f1 flip: sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1 flip: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1 flip: sint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, out %b : !firrtl.bundle<f1: uint<1>>) {
  // expected-note @+1 {{the left-hand-side was defined here}}
  %0 = firrtl.subfield %a(0) : (!firrtl.bundle<f1: uint<1>>) -> !firrtl.uint<1>
  %1 = firrtl.subfield %b(0) : (!firrtl.bundle<f1: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{op has invalid flow: the left-hand-side has source flow}}
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}
}

/// Destination bitwidth must be greater than or equal to source bitwidth.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<2>, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{destination '!firrtl.uint<1>' is not as wide as the source '!firrtl.uint<2>'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<2>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {a: {flip a: UInt<1>}}
///     wire   ax: {a: {flip a: UInt<1>}}
///     a.a.a <= ax.a.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a_a = firrtl.subfield %a_a(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  %ax_a_a = firrtl.subfield %ax_a(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a <= ax.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a.a <= ax.a.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a_a = firrtl.subfield %a_a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  %ax_a_a = firrtl.subfield %ax_a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {flip a: UInt<1>}}
///     wire   ax: {flip a: {flip a: UInt<1>}}
///     a.a <= ax.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----

/// Check that the following is an invalid sink flow source.  This has to use a
/// memory because all other sinks (module outputs or instance inputs) can
/// legally be used as sources.
///
///     output a: UInt<1>
///
///     mem memory:
///       data-type => UInt<1>
///       depth => 2
///       reader => r
///       read-latency => 0
///       write-latency => 1
///       read-under-write => undefined
///
///     a <= memory.r.en

firrtl.circuit "test" {
firrtl.module @test(out %a: !firrtl.uint<1>) {
  %memory_r = firrtl.mem Undefined  {depth = 2 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // expected-note @+1 {{the right-hand-side was defined here}}
  %memory_r_en = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the right-hand-side has sink flow}}
  firrtl.connect %a, %memory_r_en : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {a: {flip a: UInt<1>}}
///     wire   ax: {a: {flip a: UInt<1>}}
///     a.a.a <- ax.a.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a_a = firrtl.subfield %a_a(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  %ax_a_a = firrtl.subfield %ax_a(0) : (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.partialconnect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a <- ax.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.partialconnect %a_a, %ax_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a.a <- ax.a.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a_a = firrtl.subfield %a_a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint<1>>
  %ax_a_a = firrtl.subfield %ax_a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.partialconnect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {flip a: UInt<1>}}
///     wire   ax: {flip a: {flip a: UInt<1>}}
///     a.a <- ax.a

firrtl.circuit "test"  {
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  // expected-note @+1 {{the left-hand-side was defined here}}
  %a_a = firrtl.subfield %a(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  %ax_a = firrtl.subfield %ax(0) : (!firrtl.bundle<a flip: bundle<a flip: uint<1>>>) -> !firrtl.bundle<a flip: uint<1>>
  // expected-error @+1 {{invalid flow: the left-hand-side has source flow}}
  firrtl.partialconnect %a_a, %ax_a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----

/// Check that the following is an invalid sink flow source.  This has to use a
/// memory because all other sinks (module outputs or instance inputs) can
/// legally be used as sources.
///
///     output a: UInt<1>
///
///     mem memory:
///       data-type => UInt<1>
///       depth => 2
///       reader => r
///       read-latency => 0
///       write-latency => 1
///       read-under-write => undefined
///
///     a <- memory.r.en

firrtl.circuit "test" {
firrtl.module @test(out %a: !firrtl.uint<1>) {
  %memory_r = firrtl.mem Undefined  {depth = 2 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // expected-note @+1 {{the right-hand-side was defined here}}
  %memory_r_en = firrtl.subfield %memory_r(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>) -> !firrtl.uint<1>
  // expected-error @+1 {{invalid flow: the right-hand-side has sink flow}}
  firrtl.partialconnect %a, %memory_r_en : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<a: uint<1>>, out %b : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.partialconnect %b, %a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.partialconnect %b, %a : !firrtl.sint<1>, !firrtl.uint<1>
}
}
