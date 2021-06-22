// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-translate  --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8) -> (%a: i8, %b: i8) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8
  hw.output %a, %b : i8, i8
}

// VERILOG: assign a = ~arg0 + arg1;
// VERILOG: assign b = ~arg0 + arg2;


/// The pass should sink constants in to the block where they are used.
// CHECK-LABEL: @sink_constants
hw.module @sink_constants(%clock :i1) -> (%out : i1){
  // CHECK: %false = hw.constant false
  %false = hw.constant false

  /// Constants not used should be removed.
  // CHECK-NOT: %true = hw.constant true
  %true = hw.constant true

  /// Simple procedural and graph region sinking.
  sv.ifdef.procedural "FOO" {
    // CHECK: [[TRUE:%.*]] = hw.constant true
    // CHECK: [[FALSE:%.*]] = hw.constant false
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    sv.fwrite "%x"(%true) : i1
    // CHECK: sv.fwrite "%x"([[FALSE]]) : i1
    sv.fwrite "%x"(%false) : i1
  }
  
  /// Multiple uses in the same block should use the same constant.
  sv.ifdef.procedural "FOO" {
    // CHECK: [[TRUE:%.*]] = hw.constant true
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    // CHECK: sv.fwrite "%x"([[TRUE]]) : i1
    sv.fwrite "%x"(%true) : i1
    sv.fwrite "%x"(%true) : i1
  }

  //CHECK: hw.output %false : i1
  hw.output %false : i1
}

// VERILOG: `ifdef FOO	// <stdin>:12:5
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);	// <stdin>:13:15, :15:7
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h0);	// <stdin>:14:18, :16:7
// VERILOG: `endif
// VERILOG: `ifdef FOO	// <stdin>:18:5
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);	// <stdin>:19:15, :20:7
// VERILOG:   $fwrite(32'h80000002, "%x", 1'h1);	// <stdin>:19:15, :21:7
// VERILOG: `endif

