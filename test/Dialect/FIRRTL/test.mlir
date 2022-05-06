// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "MyModule" {

firrtl.module @mod() { }
firrtl.extmodule @extmod()
firrtl.memmodule @memmod () attributes {
  depth = 16 : ui64, dataType = !firrtl.uint<1>, extraPorts = [], 
  maskBits = 0 : ui32, numReadPorts = 0 : ui32, numWritePorts = 0 : ui32,
  numReadWritePorts = 0 : ui32, readLatency = 0 : ui32,
  writeLatency = 1 : ui32}

// Constant op supports different return types.
firrtl.module @Constants() {
  // CHECK: %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
  firrtl.constant 0 : !firrtl.uint<0>
  // CHECK: %c0_si0 = firrtl.constant 0 : !firrtl.sint<0>
  firrtl.constant 0 : !firrtl.sint<0>
  // CHECK: %c4_ui8 = firrtl.constant 4 : !firrtl.uint<8>
  firrtl.constant 4 : !firrtl.uint<8>
  // CHECK: %c-4_si16 = firrtl.constant -4 : !firrtl.sint<16>
  firrtl.constant -4 : !firrtl.sint<16>
  // CHECK: %c1_clock = firrtl.specialconstant 1 : !firrtl.clock
  firrtl.specialconstant 1 : !firrtl.clock
  // CHECK: %c1_reset = firrtl.specialconstant 1 : !firrtl.reset
  firrtl.specialconstant 1 : !firrtl.reset
  // CHECK: %c1_asyncreset = firrtl.specialconstant 1 : !firrtl.asyncreset
  firrtl.specialconstant 1 : !firrtl.asyncreset
  // CHECK: firrtl.constant 4 : !firrtl.uint<8> {name = "test"}
  firrtl.constant 4 : !firrtl.uint<8> {name = "test"}
}

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(in %in : !firrtl.uint<8>,
                        out %out : !firrtl.uint<8>) {
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @MyModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>)
// CHECK-NEXT:    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input c:Analog<13>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  firrtl.module @Top(out %out: !firrtl.uint,
                     in %b: !firrtl.uint<32>,
                     in %c: !firrtl.analog<13>,
                     in %d: !firrtl.uint<16>) {
    %3 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>

    %4 = firrtl.invalidvalue : !firrtl.analog<13>
    firrtl.attach %c, %4 : !firrtl.analog<13>, !firrtl.analog<13>
    %5 = firrtl.add %3, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>

    firrtl.connect %out, %5 : !firrtl.uint, !firrtl.uint<34>
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(out %out: !firrtl.uint,
// CHECK:                            in %b: !firrtl.uint<32>, in %c: !firrtl.analog<13>, in %d: !firrtl.uint<16>) {
// CHECK-NEXT:      %0 = firrtl.add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>
// CHECK-NEXT:      %invalid_analog13 = firrtl.invalidvalue : !firrtl.analog<13>
// CHECK-NEXT:      firrtl.attach %c, %invalid_analog13 : !firrtl.analog<13>, !firrtl.analog<13>
// CHECK-NEXT:      %1 = firrtl.add %0, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>
// CHECK-NEXT:      firrtl.connect %out, %1 : !firrtl.uint, !firrtl.uint<34>
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// Test some hard cases of name handling.
firrtl.module @Mod2(in %in : !firrtl.uint<8>,
                    out %out : !firrtl.uint<8>) attributes {portNames = ["some_name", "out"]}{
  firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.module @Mod2(in %some_name: !firrtl.uint<8>,
// CHECK:                           out %out: !firrtl.uint<8>)
// CHECK-NEXT:    firrtl.connect %out, %some_name : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }

// Check that quotes port names are paresable and printed with quote only if needed.
// CHECK: firrtl.extmodule @TrickyNames(in "777": !firrtl.uint, in abc: !firrtl.uint)
firrtl.extmodule @TrickyNames(in "777": !firrtl.uint, in "abc": !firrtl.uint)

// Modules may be completely empty.
// CHECK-LABEL: firrtl.module @no_ports() {
firrtl.module @no_ports() {
}

// stdIntCast can work with clock inputs/outputs too.
// CHECK-LABEL: @ClockCast
firrtl.module @ClockCast(in %clock: !firrtl.clock) {
  // CHECK: %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1
  %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1

  // CHECK: %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
  %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
}


// CHECK-LABEL: @TestDshRL
firrtl.module @TestDshRL(in %in1 : !firrtl.uint<2>, in %in2: !firrtl.uint<3>) {
  // CHECK: %0 = firrtl.dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>
  %0 = firrtl.dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>

  // CHECK: %1 = firrtl.dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %1 = firrtl.dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>

  // CHECK: %2 = firrtl.dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %2 = firrtl.dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
}

// We allow implicit truncation of a register's reset value.
// CHECK-LABEL: @RegResetTruncation
firrtl.module @RegResetTruncation(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %value: !firrtl.bundle<a: uint<2>>, out %out: !firrtl.bundle<a: uint<1>>) {
  %r2 = firrtl.regreset %clock, %reset, %value  : !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<1>>
  firrtl.connect %out, %r2 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: @TestNodeName
firrtl.module @TestNodeName(in %in1 : !firrtl.uint<8>) {
  // CHECK: %n1 = firrtl.node %in1 : !firrtl.uint<8>
  %n1 = firrtl.node %in1 : !firrtl.uint<8>

  // CHECK: %n1_0 = firrtl.node %in1 {name = "n1"} : !firrtl.uint<8>
  %n2 = firrtl.node %in1 {name = "n1"} : !firrtl.uint<8>
}

// CHECK-LABEL: @TestInvalidAttr
firrtl.module @TestInvalidAttr() {
  // This just shows we can parse and print the InvalidAttr.

  // CHECK: firrtl.constant 42 : !firrtl.uint<8>
  %x = firrtl.constant 42 : !firrtl.uint<8> {
    // CHECK-SAME: {test.thing1 = #firrtl.invalidvalue<!firrtl.clock>,
    test.thing1 = #firrtl.invalidvalue<!firrtl.clock>,
    // CHECK-SAME: test.thing2 = #firrtl.invalidvalue<!firrtl.sint<3>>,
    test.thing2 = #firrtl.invalidvalue<!firrtl.sint<3>>,
    // CHECK-SAME: test.thing3 = #firrtl.invalidvalue<!firrtl.uint>}
    test.thing3 = #firrtl.invalidvalue<!firrtl.uint>
  }
}

// CHECK-LABEL: @VerbatimExpr
firrtl.module @VerbatimExpr() {
  // CHECK: %[[TMP:.+]] = firrtl.verbatim.expr "FOO" : () -> !firrtl.uint<42>
  // CHECK: %[[TMP2:.+]] = firrtl.verbatim.expr "$bits({{[{][{]0[}][}]}})"(%[[TMP]]) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  // CHECK: firrtl.add %[[TMP]], %[[TMP2]] : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
  %0 = firrtl.verbatim.expr "FOO" : () -> !firrtl.uint<42>
  %1 = firrtl.verbatim.expr "$bits({{0}})"(%0) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  %2 = firrtl.add %0, %1 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
}

// CHECK-LABL: @LowerToBind
// CHECK: firrtl.instance foo sym @s1 {lowerToBind = true} @InstanceLowerToBind()
firrtl.module @InstanceLowerToBind() {}
firrtl.module @LowerToBind() {
  firrtl.instance foo sym @s1 {lowerToBind = true} @InstanceLowerToBind() 
}

// CHECK-LABEL: @ProbeTest
firrtl.module @ProbeTest(in %in1 : !firrtl.uint<2>, in %in2 : !firrtl.uint<3>, out %out3: !firrtl.uint<3>) {
  %w1 = firrtl.wire  : !firrtl.uint<4>
  // CHECK: %[[TMP3:.+]] = firrtl.cat
  %w2 = firrtl.cat %in1, %in1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  firrtl.connect %w1, %w2 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out3, %in2 : !firrtl.uint<3>, !firrtl.uint<3>
  %someNode = firrtl.node %in1 : !firrtl.uint<2>
  // CHECK: firrtl.probe @foobar, %in1, %in2, %out3, %w1, %[[TMP3]], %someNode : !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<3>, !firrtl.uint<4>, !firrtl.uint<4>, !firrtl.uint<2>
  firrtl.probe @foobar, %in1, %in2, %out3, %w1, %w2, %someNode : !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<3>, !firrtl.uint<4>, !firrtl.uint<4>, !firrtl.uint<2>
}


}

