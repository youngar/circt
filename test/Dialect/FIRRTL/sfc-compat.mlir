// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-sfc-compat))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "SFCCompatTests" {

  firrtl.module @SFCCompatTests() {}

  // An invalidated regreset should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidValue
  firrtl.module @InvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %invalid_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through a wire should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidThroughWire
  firrtl.module @InvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %inv  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated via an output port should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidPort
  firrtl.module @InvalidPort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>, out %x: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %x, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %x  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidate via an instance input port should be converted to a
  // reg.
  //
  // CHECK-LABEL: @InvalidInstancePort
  firrtl.module @InvalidInstancePort_Submodule(in %inv: !firrtl.uint<1>) {}
  firrtl.module @InvalidInstancePort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = firrtl.wire  : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %submodule_inv = firrtl.instance submodule  @InvalidInstancePort_Submodule(in inv: !firrtl.uint<1>)
    firrtl.connect %submodule_inv, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: firrtl.reg %clock
    %r = firrtl.regreset %clock, %reset, %submodule_inv  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A primitive operation should block invalid propagation.
  firrtl.module @InvalidPrimop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.not %invalid_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %0  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalid value should NOT propagate through a node.
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    %inv = firrtl.wire  : !firrtl.uint<8>
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.connect %inv, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %_T = firrtl.node %inv  : !firrtl.uint<8>
    // CHECK: firrtl.regreset %clock
    %r = firrtl.regreset %clock, %reset, %_T  : !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %r, %d : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %q, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }

  firrtl.module @AggregateInvalid(out %q: !firrtl.bundle<a:uint<1>>) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.bundle<a:uint<1>>
    firrtl.connect %q, %invalid_ui1 : !firrtl.bundle<a:uint<1>>, !firrtl.bundle<a:uint<1>>
    // CHECK: %c0_ui1 = firrtl.constant 0
    // CHECK-NEXT: %[[CAST:.+]] = firrtl.bitcast %c0_ui1
    // CHECK-NEXT: %q, %[[CAST]]
  }

  // All of these should not error as the register is initialzed to a constant
  // reset value while looking through constructs that the SFC allows.  This is
  // testing the following cases:
  //
  //   1. A wire marked don't touch driven to a constant.
  //   2. A node driven to a constant.
  //   3. A wire driven to an invalid.
  //   4. A constant that passes through SFC-approved primops.
  //
  // CHECK-LABEL: firrtl.module @ConstantAsyncReset
  firrtl.module @ConstantAsyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.strictconnect %r0_init, %c0_ui1 : !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %r1_init = firrtl.node %c0_ui1 : !firrtl.uint<1>
    %r1 = firrtl.regreset %clock, %reset, %r1_init : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %inv_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %r2_init = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %r2_init, %inv_ui1 : !firrtl.uint<1>
    %r2 = firrtl.regreset %clock, %reset, %r2_init : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %c0_si1 = firrtl.asSInt %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
    %c0_clock = firrtl.asClock %c0_si1 : (!firrtl.sint<1>) -> !firrtl.clock
    %c0_asyncreset = firrtl.asAsyncReset %c0_clock : (!firrtl.clock) -> !firrtl.asyncreset
    %r3_init = firrtl.asUInt %c0_asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %r3 = firrtl.regreset %clock, %reset, %r3_init : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @TailPrimOp
  firrtl.module @TailPrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.pad %c0_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = firrtl.tail %0, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    %r0_init = firrtl.wire sym @r0_init : !firrtl.uint<1>
    firrtl.strictconnect %r0_init, %1: !firrtl.uint<1>
    %r0 = firrtl.regreset %clock, %reset, %r0_init : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Port" {
  // expected-note @+1 {{reset driver is here}}
  firrtl.module @NonConstantAsyncReset_Port(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.regreset' op has an async reset, but its reset value is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %x : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_PrimOp" {
  firrtl.module @NonConstantAsyncReset_PrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // expected-note @+1 {{reset driver is here}}
    %c1_ui1 = firrtl.not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.regreset' op has an async reset, but its reset value is not driven with a constant value through wires, nodes, or connects}}
    %r0 = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}
