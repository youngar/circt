
// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "reset0" {

// Reset destination.
firrtl.module @reset0(in %a : !firrtl.uint<1>, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.uint<1>
}

firrtl.module @reset1(in %a : !firrtl.asyncreset, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.asyncreset
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.
firrtl.module @reset2(in %a : !firrtl.reset, out %b : !firrtl.reset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.reset
}

firrtl.module @reset3(in %a : !firrtl.reset, out %b : !firrtl.uint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.reset
}

firrtl.module @reset4(in %a : !firrtl.reset, out %b : !firrtl.asyncreset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.reset
}

// AsyncReset source.
firrtl.module @asyncreset0(in %a : !firrtl.asyncreset, out %b : !firrtl.asyncreset) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.asyncreset
}

// Clock source.
firrtl.module @clock0(in %a : !firrtl.clock, out %b : !firrtl.clock) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.clock
}

/// Ground types can be connected if they are the same ground type.

// SInt<> source.
firrtl.module @sint0(in %a : !firrtl.sint<1>, out %b : !firrtl.sint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.sint<1>
}

// UInt<> source.
firrtl.module @uint0(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<1>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<1>
}
firrtl.module @uint1(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<2>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.uint<1>
}

/// Vector types can be connected if they have the same size and element type.
firrtl.module @vect0(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<uint<1>, 3>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.vector<uint<1>, 3>, !firrtl.vector<uint<1>, 3>
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

firrtl.module @bundle0(in %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>, out %b : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>, !firrtl.bundle<f1: uint<1>, f2 flip: sint<1>>
}

firrtl.module @bundle1(in %a : !firrtl.bundle<f1: uint<1>, f2 flip: sint<2>>, out %b : !firrtl.bundle<f1: uint<2>, f2 flip: sint<1>>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.bundle<f1: uint<2>, f2 flip: sint<1>>, !firrtl.bundle<f1: uint<1>, f2 flip: sint<2>>
}

/// Destination bitwidth must be greater than or equal to source bitwidth.
firrtl.module @bitwidth(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<2>) {
  // CHECK: firrtl.connect %b, %a
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.uint<1>
}

firrtl.module @wires0(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %w = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w, %in : !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %w : !firrtl.uint<1>
  firrtl.connect %w, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %w : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires1(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %wf, %in : !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %wf : !firrtl.uint<1>
  firrtl.connect %wf, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %wf : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires2() {
  %w0 = firrtl.wire : !firrtl.uint<1>
  %w1 = firrtl.wire : !firrtl.uint<1>
  // CHECK: firrtl.connect %w0, %w1
  firrtl.connect %w0, %w1 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires3(out %out : !firrtl.uint<1>) {
  %wf = firrtl.wire : !firrtl.uint<1>
  // check that we can read from an output port
  // CHECK: firrtl.connect %wf, %out
  firrtl.connect %wf, %out : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @wires4(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
  %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @registers0(in %clock : !firrtl.clock, in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %in : !firrtl.uint<1>
  // CHECK: firrtl.connect %out, %0 : !firrtl.uint<1>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
  firrtl.connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @registers1(in %clock : !firrtl.clock) {
  %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  %1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: firrtl.connect %0, %1
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Connections can occur within conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionConstAssign
firrtl.module @ConstConditionConstAssign(in %cond: !firrtl.uint<1>, in %in1: !firrtl.sint<2>, in %in2: !firrtl.sint<2>, out %out: !firrtl.sint<2>) {
  firrtl.when %cond : !firrtl.uint<1> {
    firrtl.matchingconnect %out, %in1 : !firrtl.sint<2>
  } else {
    firrtl.matchingconnect %out, %in2 : !firrtl.sint<2>
  }
}

// Connections can occur within conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionNonConstAssign
firrtl.module @ConstConditionNonConstAssign(in %cond: !firrtl.uint<1>, in %in1: !firrtl.sint<2>, in %in2: !firrtl.sint<2>, out %out: !firrtl.sint<2>) {
  firrtl.when %cond : !firrtl.uint<1> {
    firrtl.matchingconnect %out, %in1 : !firrtl.sint<2>
  } else {
    firrtl.matchingconnect %out, %in2 : !firrtl.sint<2>
  }
}

// Connections can occur when the destination is local to a conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstAssign
firrtl.module @NonConstWhenLocalConstAssign(in %cond: !firrtl.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.uint<9>
    %c = firrtl.constant 0 : !firrtl.uint<9>
    firrtl.matchingconnect %w, %c : !firrtl.uint<9>
  }
}

// Connections can occur when the destination is local to a conditioned when block
// and the connection is inside a nested conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign
firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign(in %cond: !firrtl.uint<1>, in %constCond: !firrtl.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.uint<9>
    firrtl.when %constCond : !firrtl.uint<1> {
      %c = firrtl.constant 0 : !firrtl.uint<9>
      firrtl.matchingconnect %w, %c : !firrtl.uint<9>
    } else {
      %c = firrtl.constant 1 : !firrtl.uint<9>
      firrtl.matchingconnect %w, %c : !firrtl.uint<9>
    }
  }
}

// Connections to flip destinations are allowed within when blocks
firrtl.module @NonConstWhenConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.bundle<a flip: uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

// Connections to nested flip destinations are allowed within when blocks
firrtl.module @NonConstWhenNestedConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.bundle<a flip: uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

// Connections to flip sources can occur when the source is local to a conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstFlipAssign
firrtl.module @NonConstWhenLocalConstFlipAssign(in %cond: !firrtl.uint<1>, out %out : !firrtl.bundle<a flip: uint<2>>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.bundle<a flip: uint<2>>
    firrtl.connect %out, %w : !firrtl.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

// Connections to nested flip sources can occur when the source is local to a conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalNestedConstFlipAssign
firrtl.module @NonConstWhenLocalNestedConstFlipAssign(in %cond: !firrtl.uint<1>, out %out : !firrtl.bundle<a flip: uint<2>>) {
  firrtl.when %cond : !firrtl.uint<1> {
    %w = firrtl.wire : !firrtl.bundle<a flip: uint<2>>
    firrtl.connect %out, %w : !firrtl.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}
}
