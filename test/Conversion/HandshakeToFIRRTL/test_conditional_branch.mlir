// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_conditional_branch_in_ui1_ui64_out_ui64_ui64(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %arg2(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.subfield %arg3(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %10 = firrtl.subfield %arg3(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %11 = firrtl.subfield %arg3(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %12 = firrtl.and %0, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %13 = firrtl.not %2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %14 = firrtl.and %2, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %6, %14 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %15 = firrtl.and %13, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %9, %15 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %8, %5 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   firrtl.connect %11, %5 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   %16 = firrtl.mux(%2, %7, %10) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %17 = firrtl.and %16, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_conditional_branch(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg5: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_conditional_branch(%arg0: i1, %arg1: index, %arg2: none, ...) -> (index, index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_arg2, %inst_arg3 = firrtl.instance "" @handshake_conditional_branch_in_ui1_ui64_out_ui64_ui64(in arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  %0:2 = "handshake.conditional_branch"(%arg0, %arg1) {control = false}: (i1, index) -> (index, index)
  handshake.return %0#0, %0#1, %arg2 : index, index, none
}
