// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: out %out0: !firrtl.uint<42>
  // CHECK-SAME: out %out1: !firrtl.sint<42>
  firrtl.module @InferConstant(out %out0: !firrtl.uint, out %out1: !firrtl.sint) {
    %0 = firrtl.constant 1 : !firrtl.uint<42>
    %1 = firrtl.constant 2 : !firrtl.sint<42>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: {{.+}} = firrtl.constant 0 : !firrtl.sint<1>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.uint<8>
    // CHECK: {{.+}} = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: {{.+}} = firrtl.constant -200 : !firrtl.sint<9>
    %2 = firrtl.constant 0 : !firrtl.uint
    %3 = firrtl.constant 0 : !firrtl.sint
    %4 = firrtl.constant 200 : !firrtl.uint
    %5 = firrtl.constant 200 : !firrtl.sint
    %6 = firrtl.constant -200 : !firrtl.sint
    firrtl.connect %out0, %0 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out1, %1 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @InferSpecialConstant
  firrtl.module @InferSpecialConstant() {
    // CHECK: %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
  }

  // CHECK-LABEL: @InferOutput
  // CHECK-SAME: out %out: !firrtl.uint<2>
  firrtl.module @InferOutput(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  // CHECK-LABEL: @InferOutput2
  // CHECK-SAME: out %out: !firrtl.uint<2>
  firrtl.module @InferOutput2(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    firrtl.partialconnect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  firrtl.module @InferNode() {
    %w = firrtl.wire : !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    // CHECK: %node = firrtl.node %w : !firrtl.uint<3>
    %node = firrtl.node %w : !firrtl.uint
  }

  firrtl.module @InferNode2() {
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %w = firrtl.wire : !firrtl.uint
    firrtl.connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>

    %node2 = firrtl.node %w : !firrtl.uint

    %w1 = firrtl.wire : !firrtl.uint
    firrtl.connect %w1, %node2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @AddSubOp
  firrtl.module @AddSubOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.add {{.*}} -> !firrtl.uint<4>
    // CHECK: %3 = firrtl.sub {{.*}} -> !firrtl.uint<5>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.add %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.sub %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @MulDivRemOp
  firrtl.module @MulDivRemOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.mul {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.div {{.*}} -> !firrtl.uint<3>
    // CHECK: %6 = firrtl.div {{.*}} -> !firrtl.sint<4>
    // CHECK: %7 = firrtl.rem {{.*}} -> !firrtl.uint<2>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.mul %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.div %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %6 = firrtl.div %3, %2 : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
    %7 = firrtl.rem %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_si2 = firrtl.constant 1 : !firrtl.sint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorOp
  firrtl.module @AndOrXorOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.and {{.*}} -> !firrtl.uint<3>
    // CHECK: %3 = firrtl.or {{.*}} -> !firrtl.uint<3>
    // CHECK: %4 = firrtl.xor {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.and %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = firrtl.or %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %4 = firrtl.xor %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @ComparisonOp
  firrtl.module @ComparisonOp(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>) {
    // CHECK: %6 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %7 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %8 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %9 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %10 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %11 = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.leq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %1 = firrtl.lt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %2 = firrtl.geq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = firrtl.gt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %4 = firrtl.eq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %5 = firrtl.neq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %6 = firrtl.wire : !firrtl.uint
    %7 = firrtl.wire : !firrtl.uint
    %8 = firrtl.wire : !firrtl.uint
    %9 = firrtl.wire : !firrtl.uint
    %10 = firrtl.wire : !firrtl.uint
    %11 = firrtl.wire : !firrtl.uint
    firrtl.connect %6, %0 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %7, %1 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %8, %2 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %9, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %10, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %11, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @CatDynShiftOp
  firrtl.module @CatDynShiftOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.sint<2>
    // CHECK: %3 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = firrtl.cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %6 = firrtl.dshl {{.*}} -> !firrtl.uint<10>
    // CHECK: %7 = firrtl.dshl {{.*}} -> !firrtl.sint<10>
    // CHECK: %8 = firrtl.dshlw {{.*}} -> !firrtl.uint<3>
    // CHECK: %9 = firrtl.dshlw {{.*}} -> !firrtl.sint<3>
    // CHECK: %10 = firrtl.dshr {{.*}} -> !firrtl.uint<3>
    // CHECK: %11 = firrtl.dshr {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.sint
    %3 = firrtl.wire : !firrtl.sint
    %4 = firrtl.cat %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = firrtl.cat %2, %3 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
    %6 = firrtl.dshl %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %7 = firrtl.dshl %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %8 = firrtl.dshlw %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %9 = firrtl.dshlw %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %10 = firrtl.dshr %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %11 = firrtl.dshr %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_si2 = firrtl.constant 1 : !firrtl.sint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    firrtl.connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CastOp
  firrtl.module @CastOp() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %4 = firrtl.asSInt {{.*}} -> !firrtl.sint<2>
    // CHECK: %5 = firrtl.asUInt {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.wire : !firrtl.clock
    %3 = firrtl.wire : !firrtl.asyncreset
    %4 = firrtl.asSInt %0 : (!firrtl.uint) -> !firrtl.sint
    %5 = firrtl.asUInt %1 : (!firrtl.sint) -> !firrtl.uint
    %6 = firrtl.asUInt %2 : (!firrtl.clock) -> !firrtl.uint<1>
    %7 = firrtl.asUInt %3 : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %8 = firrtl.asClock %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
    %9 = firrtl.asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CvtOp
  firrtl.module @CvtOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.cvt {{.*}} -> !firrtl.sint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.cvt %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.cvt %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NegOp
  firrtl.module @NegOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.neg {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = firrtl.neg {{.*}} -> !firrtl.sint<4>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.neg %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = firrtl.neg %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NotOp
  firrtl.module @NotOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.sint<3>
    // CHECK: %2 = firrtl.not {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.not {{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.sint
    %2 = firrtl.not %0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.not %1 : (!firrtl.sint) -> !firrtl.uint
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorReductionOp
  firrtl.module @AndOrXorReductionOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.andr {{.*}} -> !firrtl.uint<1>
    // CHECK: %4 = firrtl.orr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.xorr {{.*}} -> !firrtl.uint<1>
    %c0_ui16 = firrtl.constant 0 : !firrtl.uint<16>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.andr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %4 = firrtl.orr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %5 = firrtl.xorr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %1, %4 : !firrtl.uint, !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @BitsHeadTailPadOp
  firrtl.module @BitsHeadTailPadOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %3 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %8 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %9 = firrtl.tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %10 = firrtl.pad {{.*}} -> !firrtl.uint<42>
    // CHECK: %11 = firrtl.pad {{.*}} -> !firrtl.sint<42>
    // CHECK: %12 = firrtl.pad {{.*}} -> !firrtl.uint<99>
    // CHECK: %13 = firrtl.pad {{.*}} -> !firrtl.sint<99>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.wire : !firrtl.uint

    %4 = firrtl.bits %ui 3 to 1 : (!firrtl.uint) -> !firrtl.uint<3>
    %5 = firrtl.bits %si 3 to 1 : (!firrtl.sint) -> !firrtl.uint<3>
    %6 = firrtl.head %ui, 5 : (!firrtl.uint) -> !firrtl.uint<5>
    %7 = firrtl.head %si, 5 : (!firrtl.sint) -> !firrtl.uint<5>
    %8 = firrtl.tail %ui, 30 : (!firrtl.uint) -> !firrtl.uint
    %9 = firrtl.tail %si, 30 : (!firrtl.sint) -> !firrtl.uint
    %10 = firrtl.pad %ui, 13 : (!firrtl.uint) -> !firrtl.uint
    %11 = firrtl.pad %si, 13 : (!firrtl.sint) -> !firrtl.sint
    %12 = firrtl.pad %ui, 99 : (!firrtl.uint) -> !firrtl.uint
    %13 = firrtl.pad %si, 99 : (!firrtl.sint) -> !firrtl.sint

    firrtl.connect %0, %4 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %2, %6 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %3, %7 : !firrtl.uint, !firrtl.uint<5>

    %c0_ui42 = firrtl.constant 0 : !firrtl.uint<42>
    %c0_si42 = firrtl.constant 0 : !firrtl.sint<42>
    firrtl.connect %ui, %c0_ui42 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %si, %c0_si42 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @MuxOp
  firrtl.module @MuxOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<2>
    // CHECK: %1 = firrtl.wire : !firrtl.uint<3>
    // CHECK: %2 = firrtl.wire : !firrtl.uint<1>
    // CHECK: %3 = firrtl.mux{{.*}} -> !firrtl.uint<3>
    %0 = firrtl.wire : !firrtl.uint
    %1 = firrtl.wire : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.mux(%2, %0, %1) : (!firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // CHECK: %4 = firrtl.wire : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %4 = firrtl.wire : !firrtl.uint
    %5 = firrtl.mux(%4, %c1_ui1, %c1_ui1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @ShlShrOp
  firrtl.module @ShlShrOp() {
    // CHECK: %0 = firrtl.shl {{.*}} -> !firrtl.uint<8>
    // CHECK: %1 = firrtl.shl {{.*}} -> !firrtl.sint<8>
    // CHECK: %2 = firrtl.shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = firrtl.shr {{.*}} -> !firrtl.sint<2>
    // CHECK: %4 = firrtl.shr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = firrtl.shr {{.*}} -> !firrtl.sint<1>
    %ui = firrtl.wire : !firrtl.uint
    %si = firrtl.wire : !firrtl.sint

    %0 = firrtl.shl %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %1 = firrtl.shl %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %2 = firrtl.shr %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %4 = firrtl.shr %ui, 9 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %si, 9 : (!firrtl.sint) -> !firrtl.sint

    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_si5 = firrtl.constant 0 : !firrtl.sint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %si, %c0_si5 : !firrtl.sint, !firrtl.sint<5>
  }

  // CHECK-LABEL: @PassiveCastOp
  firrtl.module @PassiveCastOp() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<5>
    // CHECK: %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint<5> to !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint
    %0 = firrtl.wire : !firrtl.uint
    %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint to !firrtl.uint
    firrtl.connect %0, %1 : !firrtl.uint, !firrtl.uint
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
  }

  // CHECK-LABEL: @TransparentOps
  firrtl.module @TransparentOps(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    %false = firrtl.constant 0 : !firrtl.uint<1>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>

    // CHECK: %ui = firrtl.wire : !firrtl.uint<5>
    %ui = firrtl.wire : !firrtl.uint

    firrtl.printf %clk, %false, "foo"
    firrtl.skip
    firrtl.stop %clk, %false, 0
    firrtl.when %a  {
      firrtl.connect %ui, %c0_ui4 : !firrtl.uint, !firrtl.uint<4>
    } else  {
      firrtl.connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    }
    firrtl.assert %clk, %true, %true, "foo"
    firrtl.assume %clk, %true, %true, "foo"
    firrtl.cover %clk, %true, %true, "foo"
  }

  // Issue #1088
  // CHECK-LABEL: @Issue1088
  firrtl.module @Issue1088(out %y: !firrtl.sint<4>) {
    // CHECK: %x = firrtl.wire : !firrtl.sint<9>
    // CHECK: %c200_si9 = firrtl.constant 200 : !firrtl.sint<9>
    // CHECK: %0 = firrtl.tail %x, 5 : (!firrtl.sint<9>) -> !firrtl.uint<4>
    // CHECK: %1 = firrtl.asSInt %0 : (!firrtl.uint<4>) -> !firrtl.sint<4>
    // CHECK: firrtl.connect %y, %1 : !firrtl.sint<4>, !firrtl.sint<4>
    // CHECK: firrtl.connect %x, %c200_si9 : !firrtl.sint<9>, !firrtl.sint<9>
    %x = firrtl.wire : !firrtl.sint
    %c200_si = firrtl.constant 200 : !firrtl.sint
    firrtl.connect %y, %x : !firrtl.sint<4>, !firrtl.sint
    firrtl.connect %x, %c200_si : !firrtl.sint, !firrtl.sint
  }

  // Should truncate all the way to 0 bits if its has to.
  // CHECK-LABEL: @TruncateConnect
  firrtl.module @TruncateConnect() {
    %w = firrtl.wire  : !firrtl.uint
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %w, %c1_ui1 : !firrtl.uint, !firrtl.uint<1>
    %w1 = firrtl.wire  : !firrtl.uint<0>
    // CHECK: %0 = firrtl.tail %w, 1 : (!firrtl.uint<1>) -> !firrtl.uint<0>
    // CHECK: firrtl.connect %w1, %0 : !firrtl.uint<0>, !firrtl.uint<0>
    firrtl.connect %w1, %w : !firrtl.uint<0>, !firrtl.uint
  }

  // Issue #1110: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1110
  // CHECK-SAME: out %y: !firrtl.uint<0>
  firrtl.module @Issue1110(in %x: !firrtl.uint<0>, out %y: !firrtl.uint) {
    firrtl.connect %y, %x : !firrtl.uint, !firrtl.uint<0>
  }

  // Issue #1118: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1118
  // CHECK-SAME: out %x: !firrtl.sint<13>
  firrtl.module @Issue1118(out %x: !firrtl.sint) {
    %c4232_ui = firrtl.constant 4232 : !firrtl.uint
    %0 = firrtl.asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
    firrtl.connect %x, %0 : !firrtl.sint, !firrtl.sint
  }

  // CHECK-LABEL: @RegSimple
  firrtl.module @RegSimple(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.uint<6>
    // CHECK: %1 = firrtl.reg %clk : !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.uint
    %2 = firrtl.wire : !firrtl.uint
    %3 = firrtl.xor %1, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %3 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // CHECK-LABEL: @RegShr
  firrtl.module @RegShr(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.uint<6>
    // CHECK: %1 = firrtl.reg %clk : !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.uint
    %2 = firrtl.shr %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %3 = firrtl.shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    firrtl.connect %1, %3 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegShl
  firrtl.module @RegShl(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = firrtl.reg %clk : !firrtl.uint<6>
    %0 = firrtl.reg %clk : !firrtl.uint
    %1 = firrtl.reg %clk : !firrtl.uint
    %2 = firrtl.reg %clk : !firrtl.uint
    %3 = firrtl.shl %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %4 = firrtl.shl %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %5 = firrtl.shr %4, 3 : (!firrtl.uint) -> !firrtl.uint
    %6 = firrtl.shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %7 = firrtl.shl %6, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %0, %2 : !firrtl.uint, !firrtl.uint
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %7 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegResetSimple
  firrtl.module @RegResetSimple(
    in %clk: !firrtl.clock,
    in %rst: !firrtl.asyncreset,
    in %x: !firrtl.uint<6>
  ) {
    // CHECK: %0 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %1 = firrtl.regreset %clk, %rst, %c0_ui1 : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %2 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>
    // CHECK: %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>
    %c0_ui = firrtl.constant 0 : !firrtl.uint
    %c0_ui17 = firrtl.constant 0 : !firrtl.uint<17>
    %0 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %1 = firrtl.regreset %clk, %rst, %c0_ui : !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %2 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint
    %3 = firrtl.regreset %clk, %rst, %c0_ui17 : !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint
    %4 = firrtl.wire : !firrtl.uint
    %5 = firrtl.xor %1, %4 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %1, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    firrtl.connect %3, %5 : !firrtl.uint, !firrtl.uint
    firrtl.connect %4, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // Don't infer a width for `firrtl.invalidvalue`, and don't complain about the
  // op not having a type inferred.
  // CHECK-LABEL: @IgnoreInvalidValue
  firrtl.module @IgnoreInvalidValue(out %out: !firrtl.uint) {
    // CHECK: %invalid_ui = firrtl.invalidvalue : !firrtl.uint
    %invalid_ui = firrtl.invalidvalue : !firrtl.uint
    %c42_ui = firrtl.constant 42 : !firrtl.uint
    firrtl.connect %out, %invalid_ui : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %c42_ui : !firrtl.uint, !firrtl.uint
  }

  // Inter-module width inference for one-to-one module-instance correspondence.
  // CHECK-LABEL: @InterModuleSimpleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleSimpleBar
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<44>
  firrtl.module @InterModuleSimpleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleSimpleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst @InterModuleSimpleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.add %inst_out, %inst_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // Inter-module width inference for multiple instances per module.
  // CHECK-LABEL: @InterModuleMultipleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleMultipleBar
  // CHECK-SAME: in %in1: !firrtl.uint<17>
  // CHECK-SAME: in %in2: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  firrtl.module @InterModuleMultipleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  firrtl.module @InterModuleMultipleBar(in %in1: !firrtl.uint<17>, in %in2: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst1_in, %inst1_out = firrtl.instance inst1 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %inst2_in, %inst2_out = firrtl.instance inst2 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = firrtl.xor %inst1_out, %inst2_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %inst1_in, %in1 : !firrtl.uint, !firrtl.uint<17>
    firrtl.connect %inst2_in, %in2 : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @InferBundle
  firrtl.module @InferBundle(in %in : !firrtl.uint<3>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<3>>
    // CHECK: firrtl.reg %clk : !firrtl.bundle<a: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: uint>
    %r = firrtl.reg %clk : !firrtl.bundle<a: uint>
    %w_a = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint>) -> !firrtl.uint
    %r_a = firrtl.subfield %r(0) : (!firrtl.bundle<a: uint>) -> !firrtl.uint
    firrtl.connect %w_a, %in : !firrtl.uint, !firrtl.uint<3>
    firrtl.connect %r_a, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferEmptyBundle
  firrtl.module @InferEmptyBundle(in %in : !firrtl.uint<3>) {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<>, b: uint>
    %w_a = firrtl.subfield %w(0) : (!firrtl.bundle<a: bundle<>, b: uint>) -> !firrtl.bundle<>
    %w_b = firrtl.subfield %w(1) : (!firrtl.bundle<a: bundle<>, b: uint>) -> !firrtl.uint
    firrtl.connect %w_b, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferBundlePort
  firrtl.module @InferBundlePort(in %in: !firrtl.bundle<a: uint<2>, b: uint<3>>, out %out: !firrtl.bundle<a: uint, b: uint>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.bundle<a: uint<2>, b: uint<3>>, !firrtl.bundle<a: uint<2>, b: uint<3>>
    firrtl.connect %out, %in : !firrtl.bundle<a: uint, b: uint>, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }

  // CHECK-LABEL: @InferVectorSubindex
  firrtl.module @InferVectorSubindex(in %in : !firrtl.uint<4>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.vector<uint, 10>
    %w_5 = firrtl.subindex %w[5] : !firrtl.vector<uint, 10>
    %r_5 = firrtl.subindex %r[5] : !firrtl.vector<uint, 10>
    firrtl.connect %w_5, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_5, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorSubaccess
  firrtl.module @InferVectorSubaccess(in %in : !firrtl.uint<4>, in %addr : !firrtl.uint<32>, in %clk : !firrtl.clock) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    // CHECK: firrtl.reg %clk : !firrtl.vector<uint<4>, 10>
    %w = firrtl.wire : !firrtl.vector<uint, 10>
    %r = firrtl.reg %clk : !firrtl.vector<uint, 10>
    %w_addr = firrtl.subaccess %w[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    %r_addr = firrtl.subaccess %r[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    firrtl.connect %w_addr, %in : !firrtl.uint, !firrtl.uint<4>
    firrtl.connect %r_addr, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorPort
  firrtl.module @InferVectorPort(in %in: !firrtl.vector<uint<4>, 2>, out %out: !firrtl.vector<uint, 2>) {
    // CHECK: firrtl.connect %out, %in : !firrtl.vector<uint<4>, 2>, !firrtl.vector<uint<4>, 2>
    firrtl.connect %out, %in : !firrtl.vector<uint, 2>, !firrtl.vector<uint<4>, 2>
  }

  // CHECK-LABEL: @InferVectorFancy
  firrtl.module @InferVectorFancy(in %in : !firrtl.uint<4>) {
    // CHECK: firrtl.wire : !firrtl.vector<uint<4>, 10>
    %wv = firrtl.wire : !firrtl.vector<uint, 10>
    %wv_5 = firrtl.subindex %wv[5] : !firrtl.vector<uint, 10>
    firrtl.connect %wv_5, %in : !firrtl.uint, !firrtl.uint<4>

    // CHECK: firrtl.wire : !firrtl.bundle<a: uint<4>>
    %wb = firrtl.wire : !firrtl.bundle<a: uint>
    %wb_a = firrtl.subfield %wb(0) : (!firrtl.bundle<a: uint>) -> !firrtl.uint

    %wv_2 = firrtl.subindex %wv[2] : !firrtl.vector<uint, 10>
    firrtl.connect %wb_a, %wv_2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: InferElementAfterVector
  firrtl.module @InferElementAfterVector() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint<3>>
    %w = firrtl.wire : !firrtl.bundle<a: vector<uint<10>, 10>, b :uint>
    %w_a = firrtl.subfield %w(1) : (!firrtl.bundle<a: vector<uint<10>, 10>, b: uint>) -> !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: InferComplexBundles
  firrtl.module @InferComplexBundles() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint<3>, 10>>, b: bundle<v: vector<uint<3>, 10>>>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a = firrtl.subfield %w(0) : (!firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>) -> !firrtl.bundle<v : vector<uint, 10>>
    %w_a_v = firrtl.subfield %w_a(0) : (!firrtl.bundle<v : vector<uint, 10>>) -> !firrtl.vector<uint, 10>
    %w_b = firrtl.subfield %w(1) : (!firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>) -> !firrtl.bundle<v : vector<uint, 10>>
    %w_b_v = firrtl.subfield %w_b(0) : (!firrtl.bundle<v : vector<uint, 10>>) -> !firrtl.vector<uint, 10>
    firrtl.connect %w_a_v, %w_b_v : !firrtl.vector<uint, 10>, !firrtl.vector<uint, 10>
    %w_b_v_2 = firrtl.subindex %w_b_v[2] : !firrtl.vector<uint, 10>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_b_v_2, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: InferComplexVectors
  firrtl.module @InferComplexVectors() {
    // CHECK: %w = firrtl.wire : !firrtl.vector<bundle<a: uint<3>, b: uint<3>>, 10>
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2 = firrtl.subindex %w[2] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2_a = firrtl.subfield %w_2(0) : (!firrtl.bundle<a: uint, b: uint>) -> !firrtl.uint
    %w_4 = firrtl.subindex %w[4] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_4_b = firrtl.subfield %w_4(1) : (!firrtl.bundle<a: uint, b: uint>) -> !firrtl.uint
    firrtl.connect %w_4_b, %w_2_a : !firrtl.uint, !firrtl.uint
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    firrtl.connect %w_2_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @AttachOne
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  firrtl.module @AttachOne(in %a0: !firrtl.analog<8>) {
    firrtl.attach %a0 : !firrtl.analog<8>
  }

  // CHECK-LABEL: @AttachTwo
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  firrtl.module @AttachTwo(in %a0: !firrtl.analog<8>, in %a1: !firrtl.analog) {
    firrtl.attach %a0, %a1 : !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @AttachMany
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  // CHECK-SAME: in %a2: !firrtl.analog<8>
  // CHECK-SAME: in %a3: !firrtl.analog<8>
  firrtl.module @AttachMany(
    in %a0: !firrtl.analog<8>,
    in %a1: !firrtl.analog,
    in %a2: !firrtl.analog<8>,
    in %a3: !firrtl.analog) {
    firrtl.attach %a0, %a1, %a2, %a3 : !firrtl.analog<8>, !firrtl.analog, !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @MemScalar
  // CHECK-SAME: out %out: !firrtl.uint<7>
  firrtl.module @MemScalar(out %out: !firrtl.uint) {
    // CHECK: firrtl.mem
    // CHECK-SAME: data flip: uint<7>
    // CHECK-SAME: data: uint<7>
    // CHECK-SAME: data: uint<7>
    %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %m_p0_data = firrtl.subfield %m_p0(3) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>) -> !firrtl.uint
    %m_p1_data = firrtl.subfield %m_p1(3) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>) -> !firrtl.uint
    %m_p2_wdata = firrtl.subfield %m_p2(5) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>) -> !firrtl.uint
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @MemBundle
  // CHECK-SAME: out %out: !firrtl.bundle<a: uint<7>>
  firrtl.module @MemBundle(out %out: !firrtl.bundle<a: uint>) {
    // CHECK: firrtl.mem
    // CHECK-SAME: data flip: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    %m_p0, %m_p1, %m_p2 = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p0_data = firrtl.subfield %m_p0(3) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>) -> !firrtl.bundle<a: uint>
    %m_p1_data = firrtl.subfield %m_p1(3) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint>
    %m_p2_wdata = firrtl.subfield %m_p2(5) : (!firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>) -> !firrtl.bundle<a: uint>
    %m_p1_data_a = firrtl.subfield %m_p1_data(0) : (!firrtl.bundle<a: uint>) -> !firrtl.uint
    %m_p2_wdata_a = firrtl.subfield %m_p2_wdata(0) : (!firrtl.bundle<a: uint>) -> !firrtl.uint
    %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
    %c0_ui7 = firrtl.constant 0 : !firrtl.uint<7>
    firrtl.connect %m_p1_data_a, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    firrtl.connect %m_p2_wdata_a, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    firrtl.connect %out, %m_p0_data : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }

  // Only matching fields are connected.
  // CHECK-LABEL: @PartialConnectBundle
  firrtl.module @PartialConnectBundle() {
    // CHECK: %a = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<2>, c: uint<3>>
    %a = firrtl.wire : !firrtl.bundle<a: uint, b: uint, c: uint>
    %b = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %c = firrtl.wire : !firrtl.bundle<c: uint<3>>
    firrtl.partialconnect %a, %b : !firrtl.bundle<a: uint, b: uint, c: uint>, !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.partialconnect %a, %c : !firrtl.bundle<a: uint, b: uint, c: uint>, !firrtl.bundle<c: uint<3>>

    // CHECK: %d = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %d = firrtl.wire : !firrtl.bundle<a: uint, b: uint>
    %e = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<2>, c: uint<3>>
    firrtl.partialconnect %d, %e : !firrtl.bundle<a: uint, b: uint>, !firrtl.bundle<a: uint<1>, b: uint<2>, c: uint<3>>
  }

  // Only the first 'n' elements in a vector are connected.
  // CHECK-LABEL: @PartialConnectVector
  firrtl.module @PartialConnectVector() {
    // CHECK: %a = firrtl.wire : !firrtl.vector<uint<42>, 2>
    %a = firrtl.wire : !firrtl.vector<uint, 2>
    %b = firrtl.wire : !firrtl.vector<uint<42>, 3>
    firrtl.partialconnect %a, %b : !firrtl.vector<uint, 2>, !firrtl.vector<uint<42>, 3>

    // CHECK: %c = firrtl.wire : !firrtl.vector<uint<9001>, 3>
    %c = firrtl.wire : !firrtl.vector<uint, 3>
    %d = firrtl.wire : !firrtl.vector<uint<9001>, 2>
    firrtl.partialconnect %c, %d : !firrtl.vector<uint, 3>, !firrtl.vector<uint<9001>, 2>
  }

  // CHECK-LABEL: @PartialConnectDepth0
  firrtl.module @PartialConnectDepth0() {
    // CHECK: %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    // CHECK: %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    // CHECK: %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    // CHECK: %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>

    // wire a0: {b: {c: UInt}}
    // wire b0: {b: {c: UInt<1>}}
    // a0 <- b0
    %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint>>
    %b0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    firrtl.partialconnect %a0, %b0 : !firrtl.bundle<b: bundle<c: uint>>, !firrtl.bundle<b: bundle<c: uint<1>>>

    // wire a1: {b: {flip c: UInt}}
    // wire b1: {b: {flip c: UInt<1>}}
    // b1 <- a1
    %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint>>
    %b1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    firrtl.partialconnect %b1, %a1 : !firrtl.bundle<b: bundle<c flip: uint<1>>>, !firrtl.bundle<b: bundle<c flip: uint>>

    // wire a2: {flip b: {c: UInt}}
    // wire b2: {flip b: {c: UInt<1>}}
    // b2 <- a2
    %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint>>
    %b2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    firrtl.partialconnect %b2, %a2 : !firrtl.bundle<b flip: bundle<c: uint<1>>>, !firrtl.bundle<b flip: bundle<c: uint>>

    // wire a3: {flip b: {flip c: UInt}}
    // wire b3: {flip b: {flip c: UInt<1>}}
    // a3 <- b3
    %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint>>
    %b3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>
    firrtl.partialconnect %a3, %b3 : !firrtl.bundle<b flip: bundle<c flip: uint>>, !firrtl.bundle<b flip: bundle<c flip: uint<1>>>
  }

  // CHECK-LABEL: @PartialConnectDepth1
  firrtl.module @PartialConnectDepth1() {
    // CHECK: %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    // CHECK: %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    // CHECK: %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    // CHECK: %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>

    // wire a0: {b: {c: UInt}}
    // wire b0: {b: {c: UInt<1>}}
    // a0.b <- b0.b
    %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint>>
    %b0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    %0 = firrtl.subfield %a0(0) : (!firrtl.bundle<b: bundle<c: uint>>) -> !firrtl.bundle<c: uint>
    %1 = firrtl.subfield %b0(0) : (!firrtl.bundle<b: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    firrtl.partialconnect %0, %1 : !firrtl.bundle<c: uint>, !firrtl.bundle<c: uint<1>>

    // wire a1: {b: {flip c: UInt}}
    // wire b1: {b: {flip c: UInt<1>}}
    // b1.b <- a1.b
    %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint>>
    %b1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    %2 = firrtl.subfield %b1(0) : (!firrtl.bundle<b: bundle<c flip: uint<1>>>) -> !firrtl.bundle<c flip: uint<1>>
    %3 = firrtl.subfield %a1(0) : (!firrtl.bundle<b: bundle<c flip: uint>>) -> !firrtl.bundle<c flip: uint>
    firrtl.partialconnect %2, %3 : !firrtl.bundle<c flip: uint<1>>, !firrtl.bundle<c flip: uint>

    // wire a2: {flip b: {c: UInt}}
    // wire b2: {flip b: {c: UInt<1>}}
    // a2.b <- b2.b
    %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint>>
    %b2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    %4 = firrtl.subfield %a2(0) : (!firrtl.bundle<b flip: bundle<c: uint>>) -> !firrtl.bundle<c: uint>
    %5 = firrtl.subfield %b2(0) : (!firrtl.bundle<b flip: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    firrtl.partialconnect %4, %5 : !firrtl.bundle<c: uint>, !firrtl.bundle<c: uint<1>>

    // wire a3: {flip b: {flip c: UInt}}
    // wire b3: {flip b: {flip c: UInt<1>}}
    // b3.b <- a3.b
    %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint>>
    %b3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>
    %6 = firrtl.subfield %b3(0) : (!firrtl.bundle<b flip: bundle<c flip: uint<1>>>) -> !firrtl.bundle<c flip: uint<1>>
    %7 = firrtl.subfield %a3(0) : (!firrtl.bundle<b flip: bundle<c flip: uint>>) -> !firrtl.bundle<c flip: uint>
    firrtl.partialconnect %6, %7 : !firrtl.bundle<c flip: uint<1>>, !firrtl.bundle<c flip: uint>
  }

  // CHECK-LABEL: @PartialConnectDepth2
  firrtl.module @PartialConnectDepth2() {
    // CHECK: %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    // CHECK: %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    // CHECK: %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    // CHECK: %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>

    // wire a0: {b: {c: UInt}}
    // wire b0: {b: {c: UInt<1>}}
    // a0.b.c <- b0.b.c
    %a0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint>>
    %b0 = firrtl.wire : !firrtl.bundle<b: bundle<c: uint<1>>>
    %0 = firrtl.subfield %a0(0) : (!firrtl.bundle<b: bundle<c: uint>>) -> !firrtl.bundle<c: uint>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<c: uint>) -> !firrtl.uint
    %2 = firrtl.subfield %b0(0) : (!firrtl.bundle<b: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    %3 = firrtl.subfield %2(0) : (!firrtl.bundle<c: uint<1>>) -> !firrtl.uint<1>
    firrtl.partialconnect %1, %3 : !firrtl.uint, !firrtl.uint<1>

    // wire a1: {b: {flip c: UInt}}
    // wire b1: {b: {flip c: UInt<1>}}
    // a1.b.c <- b1.b.c
    %a1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint>>
    %b1 = firrtl.wire : !firrtl.bundle<b: bundle<c flip: uint<1>>>
    %4 = firrtl.subfield %a1(0) : (!firrtl.bundle<b: bundle<c flip: uint>>) -> !firrtl.bundle<c flip: uint>
    %5 = firrtl.subfield %4(0) : (!firrtl.bundle<c flip: uint>) -> !firrtl.uint
    %6 = firrtl.subfield %b1(0) : (!firrtl.bundle<b: bundle<c flip: uint<1>>>) -> !firrtl.bundle<c flip: uint<1>>
    %7 = firrtl.subfield %6(0) : (!firrtl.bundle<c flip: uint<1>>) -> !firrtl.uint<1>
    firrtl.partialconnect %5, %7 : !firrtl.uint, !firrtl.uint<1>

    // wire a2: {flip b: {c: UInt}}
    // wire b2: {flip b: {c: UInt<1>}}
    // a2.b.c <- b2.b.c
    %a2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint>>
    %b2 = firrtl.wire : !firrtl.bundle<b flip: bundle<c: uint<1>>>
    %8 = firrtl.subfield %a2(0) : (!firrtl.bundle<b flip: bundle<c: uint>>) -> !firrtl.bundle<c: uint>
    %9 = firrtl.subfield %8(0) : (!firrtl.bundle<c: uint>) -> !firrtl.uint
    %10 = firrtl.subfield %b2(0) : (!firrtl.bundle<b flip: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    %11 = firrtl.subfield %10(0) : (!firrtl.bundle<c: uint<1>>) -> !firrtl.uint<1>
    firrtl.partialconnect %9, %11 : !firrtl.uint, !firrtl.uint<1>

    // wire a3: {flip b: {flip c: UInt}}
    // wire b3: {flip b: {flip c: UInt<1>}}
    // a3.b.c <- b3.b.c
    %a3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint>>
    %b3 = firrtl.wire : !firrtl.bundle<b flip: bundle<c flip: uint<1>>>
    %12 = firrtl.subfield %a3(0) : (!firrtl.bundle<b flip: bundle<c flip: uint>>) -> !firrtl.bundle<c flip: uint>
    %13 = firrtl.subfield %12(0) : (!firrtl.bundle<c flip: uint>) -> !firrtl.uint
    %14 = firrtl.subfield %b3(0) : (!firrtl.bundle<b flip: bundle<c flip: uint<1>>>) -> !firrtl.bundle<c flip: uint<1>>
    %15 = firrtl.subfield %14(0) : (!firrtl.bundle<c flip: uint<1>>) -> !firrtl.uint<1>
    firrtl.partialconnect %13, %15 : !firrtl.uint, !firrtl.uint<1>
  }

  // Breakable cycles in inter-module width inference.
  // CHECK-LABEL: @InterModuleGoodCycleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = firrtl.shr %in, 3 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  // CHECK-LABEL: @InterModuleGoodCycleBar
  // CHECK-SAME: out %out: !firrtl.uint<39>
  firrtl.module @InterModuleGoodCycleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = firrtl.instance inst  @InterModuleGoodCycleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %inst_in, %inst_out : !firrtl.uint, !firrtl.uint
    firrtl.connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @Issue1271
  firrtl.module @Issue1271(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    // CHECK: %a = firrtl.reg %clock  : !firrtl.uint<2>
    // CHECK: %b = firrtl.node %0  : !firrtl.uint<3>
    // CHECK: %c = firrtl.node %1  : !firrtl.uint<2>
    %a = firrtl.reg %clock  : !firrtl.uint
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.add %a, %c0_ui1 : (!firrtl.uint, !firrtl.uint<1>) -> !firrtl.uint
    %b = firrtl.node %0  : !firrtl.uint
    %1 = firrtl.tail %b, 1 : (!firrtl.uint) -> !firrtl.uint
    %c = firrtl.node %1  : !firrtl.uint
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %2 = firrtl.mux(%cond, %c0_ui2, %c) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %a, %2 : !firrtl.uint, !firrtl.uint
  }

  firrtl.module @Foo() {}
}
