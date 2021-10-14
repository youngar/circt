// RUN: circt-opt -lower-firrtl-to-hw  %s | FileCheck %s

firrtl.circuit "Simple"   attributes {annotations = [{class =
"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation", directory = "dir1",  filename = "./dir1/filename1" }, {class =
"sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "dir2",  filename = "./dir2/filename2" }, {class =
"sifive.enterprise.firrtl.ExtractAssertionsAnnotation", directory = "dir3",  filename = "./dir3/filename3" }]}
{

  //These come from MemSimple, IncompleteRead, and MemDepth1
  // CHECK-LABEL: hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs"]
  //
  // This memory has two write ports where both write ports are driven by the
  // same clock.
  //
  // CHECK-NEXT:  hw.module.generated @FIRRTLMem_0_2_0_8_16_1_1_1_0_1_aa,
  // CHECK-SAME: @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W0_mask: i1, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8, %W1_mask: i1)
  // CHECK-SAME: attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32,
  // CHECK-SAME: numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32,
  // CHECK-SAME: readLatency = 1 : ui32, readUnderWrite = 0 : ui32,
  // CHECK-SAME: width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32],
  // CHECK-SAME: writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  //
  // This memory is the same as the above memory, but each write port is driven
  // by a different clock.
  //
  // CHECK-NEXT:  hw.module.generated @FIRRTLMem_0_2_0_8_16_1_1_1_0_1_ab,
  // CHECK-SAME: @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W0_mask: i1, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8, %W1_mask: i1)
  // CHECK-SAME: attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 0 : ui32,
  // CHECK-SAME: numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32,
  // CHECK-SAME: readLatency = 1 : ui32, readUnderWrite = 0 : ui32,
  // CHECK-SAME: width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32],
  // CHECK-SAME: writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  //
  // CHECK-NEXT:  hw.module.generated @FIRRTLMem_1_0_0_32_1_0_1_0_1_1,
  // CHECK-SAME: @FIRRTLMem(%R0_addr: i1, %R0_en: i1, %R0_clk: i1) -> (R0_data: i32)
  // CHECK-SAME: attributes {depth = 1 : i64, maskGran = 32 : ui32, numReadPorts = 1 : ui32,
  // CHECK-SAME: numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32,
  // CHECK-SAME: readLatency = 0 : ui32, readUnderWrite = 1 : ui32,
  // CHECK-SAME: width = 32 : ui32, writeClockIDs = [],
  // CHECK-SAME: writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK-NEXT:  hw.module.generated @FIRRTLMem_1_0_0_42_12_0_1_0_0_1,
  // CHECK-SAME: @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42)
  // CHECK-SAME: attributes {depth = 12 : i64, maskGran = 42 : ui32, numReadPorts = 1 : ui32,
  // CHECK-SAME: numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32,
  // CHECK-SAME: readLatency = 0 : ui32, readUnderWrite = 0 : ui32,
  // CHECK-SAME: width = 42 : ui32, writeClockIDs = [],
  // CHECK-SAME: writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK-NEXT: hw.module.generated @FIRRTLMem_1_1_1_40_1022_1_1_4_0_1_a, 
  // CHECK-SAME:  @FIRRTLMem(%R0_addr: i10, %R0_en: i1, %R0_clk: i1, %RW0_addr: i10, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i40, %RW0_wmask: i4, %W0_addr: i10, %W0_en: i1, %W0_clk: i1, %W0_data: i40, %W0_mask: i4) -> (R0_data: i40, RW0_rdata: i40)
  // CHECK-NEXT:  hw.module.generated @FIRRTLMem_1_1_1_42_12_0_1_1_0_1_a,
  // CHECK-SAME: @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i42, %RW0_wmask: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i42, %W0_mask: i1) -> (R0_data: i42, RW0_rdata: i42)
  // CHECK-SAME: attributes {depth = 12 : i64, maskGran = 42 : ui32, numReadPorts = 1 : ui32,
  // CHECK-SAME: numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32,
  // CHECK-SAME: readLatency = 0 : ui32, readUnderWrite = 0 : ui32,
  // CHECK-SAME: width = 42 : ui32, writeClockIDs = [0 : i32],
  // CHECK-SAME: writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

  // CHECK-LABEL: hw.module @Simple
  firrtl.module @Simple(in %in1: !firrtl.uint<4>,
                        in %in2: !firrtl.uint<2>,
                        in %in3: !firrtl.sint<8>,
                        in %in4: !firrtl.uint<0>,
                        in %in5: !firrtl.sint<0>,
                        out %out1: !firrtl.sint<1>,
                        out %out2: !firrtl.sint<1>  ) {
    // Issue #364: https://github.com/llvm/circt/issues/364
    // CHECK: = hw.constant -1175 : i12
    // CHECK-DAG: hw.constant -4 : i4
    %c12_ui4 = firrtl.constant 12 : !firrtl.uint<4>

    // CHECK-DAG: hw.constant 2 : i3
    %c2_si3 = firrtl.constant 2 : !firrtl.sint<3>


    // CHECK: %out4 = sv.wire sym @__Simple__out4 : !hw.inout<i4>
    // CHECK: %out5 = sv.wire : !hw.inout<i4>
    %out4 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    %out5 = firrtl.wire : !firrtl.uint<4>
    // CHECK: sv.wire sym @__Simple{{.*}}
    // CHECK: sv.wire sym @__Simple{{.*}}
    %500 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    %501 = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<5>

    // CHECK: sv.wire sym @__Simple__dntnode
    %dntnode = firrtl.node %in1 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK: %clockWire = sv.wire  : !hw.inout<i1>
    // CHECK: sv.assign %clockWire, %false : i1
    %c0_clock = firrtl.specialconstant 0 : !firrtl.clock
    %clockWire = firrtl.wire : !firrtl.clock
    firrtl.connect %clockWire, %c0_clock : !firrtl.clock, !firrtl.clock

    // CHECK: sv.assign %out5, %c0_i4 : i4
    %tmp1 = firrtl.invalidvalue : !firrtl.uint<4>
    firrtl.connect %out5, %tmp1 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: [[ZEXT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK: [[ADD:%.+]] = comb.add %c12_i5, [[ZEXT]] : i5
    %0 = firrtl.add %c12_ui4, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[ZEXT1:%.+]] = comb.concat %false, [[ADD]] : i1, i5
    // CHECK: [[ZEXT2:%.+]] = comb.concat %c0_i2, %in1 : i2, i4
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub [[ZEXT1]], [[ZEXT2]] : i6
    %2 = firrtl.sub %0, %1 : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<6>

    %in2s = firrtl.asSInt %in2 : (!firrtl.uint<2>) -> !firrtl.sint<2>

    // CHECK: [[PADRES:%.+]] = comb.sext %in2 : (i2) -> i3
    %3 = firrtl.pad %in2s, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    %4 = firrtl.pad %in2, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    // CHECK: [[XOR:%.+]] = comb.xor [[IN2EXT]], [[PADRES2]] : i4
    %5 = firrtl.xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.and [[XOR]]
    %and = firrtl.and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.or [[XOR]]
    %or = firrtl.or %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = comb.concat [[PADRES2]], [[XOR]] : i4, i4
    %6 = firrtl.cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: comb.concat %in1, %in2
    %7 = firrtl.cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK-NEXT: sv.assign %out5, [[PADRES2]] : i4
    firrtl.connect %out5, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: sv.assign %out4, [[XOR]] : i4
    firrtl.connect %out4, %5 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: [[ZEXT:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    // CHECK-NEXT: sv.assign %out4, [[ZEXT]] : i4
    firrtl.connect %out4, %in2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK-NEXT: %test-name = sv.wire sym @"__Simple__test-name" : !hw.inout<i4>
    firrtl.wire {name = "test-name", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK-NEXT: = sv.wire : !hw.inout<i2>
    %_t_1 = firrtl.wire : !firrtl.uint<2>

    // CHECK-NEXT: = sv.wire  : !hw.inout<array<13xi1>>
    %_t_2 = firrtl.wire : !firrtl.vector<uint<1>, 13>

    // CHECK-NEXT: = sv.wire  : !hw.inout<array<13xi2>>
    %_t_3 = firrtl.wire : !firrtl.vector<uint<2>, 13>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %8 = firrtl.bits %6 7 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 5 : (i8) -> i3
    %9 = firrtl.head %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 0 : (i8) -> i5
    %10 = firrtl.tail %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %11 = firrtl.shr %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    %12 = firrtl.shr %6, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.extract %in3 from 7 : (i8) -> i1
    %13 = firrtl.shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: = comb.concat [[CONCAT1]], %c0_i3 : i8, i3
    %14 = firrtl.shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = comb.parity [[CONCAT1]] : i8
    %15 = firrtl.xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp eq  {{.*}}, %c-1_i8 : i8
    %16 = firrtl.andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp ne {{.*}}, %c0_i8 : i8
    %17 = firrtl.orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[ZEXTC1:%.+]] = comb.concat %c0_i6, [[CONCAT1]] : i6, i8
    // CHECK-NEXT: [[ZEXT2:%.+]] = comb.concat %c0_i8, [[SUB]] : i8, i6
    // CHECK-NEXT: [[VAL18:%.+]] = comb.mul  [[ZEXTC1]], [[ZEXT2]] : i14
    %18 = firrtl.mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<6>) -> !firrtl.uint<14>

    // CHECK-NEXT: [[IN3SEXT:%.+]] = comb.sext %in3 : (i8) -> i9
    // CHECK-NEXT: [[PADRESSEXT:%.+]] = comb.sext [[PADRES]] : (i3) -> i9
    // CHECK-NEXT: = comb.divs [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK-NEXT: [[IN3EX:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD1:%.+]] = comb.mods %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = comb.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = firrtl.rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[IN4EX:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK-NEXT: [[MOD2:%.+]] = comb.mods [[IN4EX]], %in3 : i8
    // CHECK-NEXT: = comb.extract [[MOD2]] from 0 : (i8) -> i3
    %21 = firrtl.rem %3, %in3 : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // Nodes with names but no attribute are just dropped.
    %n1 = firrtl.node %in2  {name = "n1"} : !firrtl.uint<2>

    // CHECK-NEXT: [[WIRE:%n2]] = sv.wire sym @__Simple__n2 : !hw.inout<i2>
    // CHECK-NEXT: sv.assign [[WIRE]], %in2 : i2
    %n2 = firrtl.node %in2  {name = "n2", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = firrtl.node %in2 {name = ""} : !firrtl.uint<2>

    // CHECK-NEXT: [[CVT:%.+]] = comb.concat %false, %in2 : i1, i2
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = firrtl.cvt %in3 : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: [[XOR:%.+]] = comb.xor [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK-NEXT: [[SEXT:%.+]] = comb.sext [[XOR]] : (i3) -> i4
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub %c0_i4, [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK-NEXT: [[CVT4:%.+]] = comb.sext [[CVT]] : (i3) -> i4
    // CHECK-NEXT: comb.mux {{.*}}, [[CVT4]], [[SUB]] : i4
    %26 = firrtl.mux(%17, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>

    // CHECK-NEXT: = comb.icmp eq  {{.*}}, %c-1_i14 : i14
    %28 = firrtl.andr %18 : (!firrtl.uint<14>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[XOREXT:%.+]] = comb.concat %c0_i11, [[XOR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shru [[XOREXT]], [[VAL18]] : i14
    // CHECK-NEXT: [[DSHR:%.+]] = comb.extract [[SHIFT]] from 0 : (i14) -> i3
    %29 = firrtl.dshr %24, %18 : (!firrtl.uint<3>, !firrtl.uint<14>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.concat %c0_i5, {{.*}} : i5, i3
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs %in3, {{.*}} : i8
    %a29 = firrtl.dshr %in3, %9 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<8>

    // CHECK-NEXT: = comb.sext %in3 : (i8) -> i15
    // CHECK-NEXT: = comb.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shl {{.*}}, {{.*}} : i15
    %30 = firrtl.dshl %in3, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = comb.shl [[DSHR]], [[DSHR]] : i3
    %dshlw = firrtl.dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK-NEXT: = comb.sext {{.*}} : (i4) -> i14
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs {{.*}}, {{.*}} : i14
    // CHECK-NEXT: = comb.extract [[SHIFT]] from 0 : (i14) -> i4
    %31 = firrtl.dshr %25, %18 : (!firrtl.sint<4>, !firrtl.uint<14>) -> !firrtl.sint<4>

    // CHECK-NEXT: comb.icmp ule {{.*}}, {{.*}} : i4
    %41 = firrtl.leq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ult {{.*}}, {{.*}} : i4
    %42 = firrtl.lt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp uge {{.*}}, {{.*}} : i4
    %43 = firrtl.geq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ugt {{.*}}, {{.*}} : i4
    %44 = firrtl.gt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp eq {{.*}}, {{.*}} : i4
    %45 = firrtl.eq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp ne {{.*}}, {{.*}} : i4
    %46 = firrtl.neq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>

    // Noop
    %47 = firrtl.asClock %44 : (!firrtl.uint<1>) -> !firrtl.clock
    %48 = firrtl.asAsyncReset %44 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK: [[VERB1:%.+]] = sv.verbatim.expr "MAGIC_CONSTANT" : () -> i42
    // CHECK: [[VERB2:%.+]] = sv.verbatim.expr "$bits({{[{][{]0[}][}]}})"([[VERB1]]) : (i42) -> i32
    // CHECK: [[VERB1EXT:%.+]] = comb.concat {{%.+}}, [[VERB1]] : i1, i42
    // CHECK: [[VERB2EXT:%.+]] = comb.concat {{%.+}}, [[VERB2]] : i11, i32
    // CHECK: = comb.add [[VERB1EXT]], [[VERB2EXT]] : i43
    %56 = firrtl.verbatim.expr "MAGIC_CONSTANT" : () -> !firrtl.uint<42>
    %57 = firrtl.verbatim.expr "$bits({{0}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32>
    %58 = firrtl.add %56, %57 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = comb.sext [[PADRES]] : (i3) -> i8
    // CHECK: = comb.and %in3, [[PADRES_EXT]] : i8
    %49 = firrtl.and %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.uint<8>

    // Issue #355: https://github.com/llvm/circt/issues/355
    // CHECK: [[IN1:%.+]] = comb.concat %c0_i6, %in1 : i6, i4
    // CHECK: [[DIV:%.+]] = comb.divu [[IN1]], %c306_i10 : i10
    // CHECK: = comb.extract [[DIV]] from 0 : (i10) -> i4
    %c306_ui10 = firrtl.constant 306 : !firrtl.uint<10>
    %50 = firrtl.div %in1, %c306_ui10 : (!firrtl.uint<4>, !firrtl.uint<10>) -> !firrtl.uint<4>

    %c1175_ui11 = firrtl.constant 1175 : !firrtl.uint<11>
    %51 = firrtl.neg %c1175_ui11 : (!firrtl.uint<11>) -> !firrtl.sint<12>
    // https://github.com/llvm/circt/issues/821
    // CHECK: [[CONCAT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK:  = comb.sub %c0_i5, [[CONCAT]] : i5
    %52 = firrtl.neg %in1 : (!firrtl.uint<4>) -> !firrtl.sint<5>
    %53 = firrtl.neg %in4 : (!firrtl.uint<0>) -> !firrtl.sint<1>
    // CHECK: [[SEXT:%.+]] = comb.sext %in3 : (i8) -> i9
    // CHECK: = comb.sub %c0_i9, [[SEXT]] : i9
    %54 = firrtl.neg %in3 : (!firrtl.sint<8>) -> !firrtl.sint<9>
    // CHECK: hw.output %false, %false : i1, i1
    firrtl.connect %out1, %53 : !firrtl.sint<1>, !firrtl.sint<1>
    %55 = firrtl.neg %in5 : (!firrtl.sint<0>) -> !firrtl.sint<1>
    firrtl.connect %out2, %55 : !firrtl.sint<1>, !firrtl.sint<1>
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: hw.module @Print
  firrtl.module @Print(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                       in %a: !firrtl.uint<4>, in %b: !firrtl.uint<4>) {
    // CHECK: [[ADD:%.+]] = comb.add

    // CHECK: sv.always posedge %clock {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:     %PRINTF_COND_ = sv.verbatim.expr "`PRINTF_COND_" : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND_, %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:     %PRINTF_COND__0 = sv.verbatim.expr "`PRINTF_COND_" : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND__0, %reset : i1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       sv.fwrite "Hi %x %x\0A"(%2, %b) : i5, i4
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   firrtl.printf %clock, %reset, "No operands!\0A"

    %0 = firrtl.add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    firrtl.printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.uint<5>, !firrtl.uint<4>

    firrtl.skip

    // CHECK: hw.output
   }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: hw.module @Stop
  firrtl.module @Stop(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock, in %reset: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.always posedge %clock1 {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     %STOP_COND_ = sv.verbatim.expr "`STOP_COND_" : () -> i1
    // CHECK-NEXT:     %0 = comb.and %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT: sv.always posedge %clock2 {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     %STOP_COND_ = sv.verbatim.expr "`STOP_COND_" : () -> i1
    // CHECK-NEXT:     %0 = comb.and %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.stop %clock2, %reset, 0
  }

  // circuit Verification:
  //   module Verification:
  //     input clock: Clock
  //     input aCond: UInt<8>
  //     input aEn: UInt<8>
  //     input bCond: UInt<1>
  //     input bEn: UInt<1>
  //     input cCond: UInt<1>
  //     input cEn: UInt<1>
  //     assert(clock, bCond, bEn, "assert0")
  //     assert(clock, bCond, bEn, "assert0") : assert_0
  //     assume(clock, aCond, aEn, "assume0")
  //     assume(clock, aCond, aEn, "assume0") : assume_0
  //     cover(clock,  cCond, cEn, "cover0)"
  //     cover(clock,  cCond, cEn, "cover0)" : cover_0

  // CHECK-LABEL: hw.module @Verification
  firrtl.module @Verification(in %clock: !firrtl.clock, in %aCond: !firrtl.uint<1>,
    in %aEn: !firrtl.uint<1>, in %bCond: !firrtl.uint<1>, in %bEn: !firrtl.uint<1>,
    in %cCond: !firrtl.uint<1>, in %cEn: !firrtl.uint<1>, in %value: !firrtl.uint<42>) {

    firrtl.assert %clock, %aCond, %aEn, "assert0" {isConcurrent = true}
    firrtl.assert %clock, %aCond, %aEn, "assert0" {isConcurrent = true, name = "assert_0"}
    firrtl.assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP3:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP4:%.+]] = comb.or [[TMP3]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP4]] label "assert__assert_0" message "assert0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP5:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP6:%.+]] = comb.or [[TMP5]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP6]] message "assert0"(%value) : i42
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP4]] label "assume__assert_0"
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP6]]
    // CHECK-NEXT: }
    firrtl.assume %clock, %bCond, %bEn, "assume0" {isConcurrent = true}
    firrtl.assume %clock, %bCond, %bEn, "assume0" {isConcurrent = true, name = "assume_0"}
    firrtl.assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] label "assume__assume_0" message "assume0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"(%value) : i42
    firrtl.cover %clock, %cCond, %cEn, "cover0" {isConcurrent = true}
    firrtl.cover %clock, %cCond, %cEn, "cover0" {isConcurrent = true, name = "cover_0"}
    firrtl.cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP2]] label "cover__cover_0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP2]]
    firrtl.cover %clock, %cCond, %cEn, "cover1" {eventControl = 1 : i32, isConcurrent = true, name = "cover_1"}
    firrtl.cover %clock, %cCond, %cEn, "cover2" {eventControl = 2 : i32, isConcurrent = true, name = "cover_2"}
    // CHECK: sv.cover.concurrent negedge %clock, {{%.+}} label "cover__cover_1"
    // CHECK: sv.cover.concurrent edge %clock, {{%.+}} label "cover__cover_2"

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %aEn {
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate label "assert__assert_0" message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %bEn {
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate label "assume__assume_0" message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %cEn {
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:     sv.cover %cCond, immediate label "cover__cover_0"
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assert %clock, %aCond, %aEn, "assert0"
    firrtl.assert %clock, %aCond, %aEn, "assert0" {name = "assert_0"}
    firrtl.assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.uint<42>
    firrtl.assume %clock, %bCond, %bEn, "assume0"
    firrtl.assume %clock, %bCond, %bEn, "assume0" {name = "assume_0"}
    firrtl.assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.uint<42>
    firrtl.cover %clock, %cCond, %cEn, "cover0"
    firrtl.cover %clock, %cCond, %cEn, "cover0" {name = "cover_0"}
    firrtl.cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.uint<42>
    // CHECK-NEXT: hw.output
  }

  // CHECK-LABEL: hw.module @VerificationGuards
  firrtl.module @VerificationGuards(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    // CHECK-NEXT: sv.ifdef "HELLO" {
    // CHECK-NEXT:   sv.ifdef "WORLD" {
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or [[TMP1]], %cond
    // CHECK-NEXT:     sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT:     sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:       sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.assume %clock, %cond, %enable, "assume0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    firrtl.cover %clock, %cond, %enable, "cover0" {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    // CHECK-NEXT: sv.ifdef "HELLO" {
    // CHECK-NEXT:   sv.ifdef "WORLD" {
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or [[TMP1]], %cond
    // CHECK-NEXT:     sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or [[TMP1]], %cond
    // CHECK-NEXT:     sv.cover.concurrent posedge %clock, [[TMP2]]
    // CHECK-NOT:      label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: hw.module @VerificationAssertFormat
  firrtl.module @VerificationAssertFormat(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>,
    in %value: !firrtl.uint<42>
  ) {
    firrtl.assert %clock, %cond, %enable, "assert0" {isConcurrent = true, format = "sva"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %cond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: }
    firrtl.assert %clock, %cond, %enable, "assert1"(%value) : !firrtl.uint<42> {isConcurrent = true, format = "ifElseFatal"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cond, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.and %enable, [[TMP1]]
    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.if [[TMP2]] {
    // CHECK-NEXT:       [[ASSERT_VERBOSE_COND:%.+]] = sv.verbatim.expr "`ASSERT_VERBOSE_COND_"
    // CHECK-NEXT:       sv.if [[ASSERT_VERBOSE_COND]] {
    // CHECK-NEXT:         sv.error "assert1"(%value) : i42
    // CHECK-NEXT:       }
    // CHECK-NEXT:       [[STOP_COND:%.+]] = sv.verbatim.expr "`STOP_COND_"
    // CHECK-NEXT:       sv.if [[STOP_COND]] {
    // CHECK-NEXT:         sv.fatal
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  firrtl.module @bar(in %io_cpu_flush: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: hw.module @foo
  firrtl.module @foo() {
    // CHECK-NEXT:  %io_cpu_flush.wire = sv.wire sym @__foo__io_cpu_flush.wire : !hw.inout<i1>
    // CHECK-NEXT:  [[IO:%[0-9]+]] = sv.read_inout %io_cpu_flush.wire
    %io_cpu_flush.wire = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT: hw.instance "fetch" @bar(io_cpu_flush: [[IO]]: i1)
    %i = firrtl.instance fetch @bar(in io_cpu_flush: !firrtl.uint<1>)
    firrtl.connect %i, %io_cpu_flush.wire : !firrtl.uint<1>, !firrtl.uint<1>

    %hits_1_7 = firrtl.node %io_cpu_flush.wire {name = "hits_1_7", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT:  %hits_1_7 = sv.wire sym @__foo__hits_1_7
    // CHECK-NEXT:  sv.assign %hits_1_7, [[IO]] : i1
    %1455 = builtin.unrealized_conversion_cast %hits_1_7 : !firrtl.uint<1> to !firrtl.uint<1>
  }

  // CHECK: sv.bind @[[bazSymbol:.+]] in @bindTest
  // CHECK-NOT: output_file
  // CHECK-NEXT: sv.bind @[[quxSymbol:.+]] in @bindTest {
  // CHECK-SAME: output_file = #hw.output_file<"outputDir/bindings.sv", excludeFromFileList>
  // CHECK-NEXT: hw.module @bindTest()
  firrtl.module @bindTest() {
    // CHECK: hw.instance "baz" sym @[[bazSymbol]] @bar
    %baz = firrtl.instance baz {lowerToBind = true} @bar(in io_cpu_flush: !firrtl.uint<1>)
    // CHECK: hw.instance "qux" sym @[[quxSymbol]] @bar
    %qux = firrtl.instance qux {lowerToBind = true, output_file = #hw.output_file<"outputDir/bindings.sv", excludeFromFileList>} @bar(in io_cpu_flush: !firrtl.uint<1>)
  }


  // CHECK-LABEL: hw.module @output_fileTest
  // CHECK-SAME: output_file = #hw.output_file<"output_fileTest/dir/output_fileTest.sv", excludeFromFileList>
  firrtl.module @output_fileTest() attributes {output_file = #hw.output_file<
    "output_fileTest/dir/output_fileTest.sv", excludeFromFileList
  >} {
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: hw.module @issue314
  firrtl.module @issue314(in %inp_2: !firrtl.uint<27>, in %inpi: !firrtl.uint<65>) {
    // CHECK: %c0_i38 = hw.constant 0 : i38
    // CHECK: %tmp48 = sv.wire : !hw.inout<i27>
    %tmp48 = firrtl.wire : !firrtl.uint<27>

    // CHECK-NEXT: %0 = comb.concat %c0_i38, %inp_2 : i38, i27
    // CHECK-NEXT: %1 = comb.divu %0, %inpi : i65
    %0 = firrtl.div %inp_2, %inpi : (!firrtl.uint<27>, !firrtl.uint<65>) -> !firrtl.uint<27>
    // CHECK-NEXT: %2 = comb.extract %1 from 0 : (i65) -> i27
    // CHECK-NEXT: sv.assign %tmp48, %2 : i27
    firrtl.connect %tmp48, %0 : !firrtl.uint<27>, !firrtl.uint<27>
  }

  // https://github.com/llvm/circt/issues/318
  // CHECK-LABEL: hw.module @test_rem
  // CHECK-NEXT:     %0 = comb.modu
  // CHECK-NEXT:     hw.output %0
  firrtl.module @test_rem(in %tmp85: !firrtl.uint<1>, in %tmp79: !firrtl.uint<1>,
       out %out: !firrtl.uint<1>) {
    %2 = firrtl.rem %tmp79, %tmp85 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @Analog(%a1: !hw.inout<i1>, %b1: !hw.inout<i1>,
  // CHECK:                          %c1: !hw.inout<i1>) -> (outClock: i1) {
  // CHECK-NEXT:   %0 = sv.read_inout %c1 : !hw.inout<i1>
  // CHECK-NEXT:   %1 = sv.read_inout %b1 : !hw.inout<i1>
  // CHECK-NEXT:   %2 = sv.read_inout %a1 : !hw.inout<i1>
  // CHECK-NEXT:   sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT:     sv.assign %a1, %1 : i1
  // CHECK-NEXT:     sv.assign %a1, %0 : i1
  // CHECK-NEXT:     sv.assign %b1, %2 : i1
  // CHECK-NEXT:     sv.assign %b1, %0 : i1
  // CHECK-NEXT:     sv.assign %c1, %2 : i1
  // CHECK-NEXT:     sv.assign %c1, %1 : i1
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:     sv.ifdef "verilator" {
  // CHECK-NEXT:       sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.alias %a1, %b1, %c1 : !hw.inout<i1>
  // CHECK-NEXT:     }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hw.output %2 : i1
  firrtl.module @Analog(in %a1: !firrtl.analog<1>, in %b1: !firrtl.analog<1>,
                        in %c1: !firrtl.analog<1>, out %outClock: !firrtl.clock) {
    firrtl.attach %a1, %b1, %c1 : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>

    %1 = firrtl.asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    firrtl.connect %outClock, %1 : !firrtl.clock, !firrtl.clock
  }


 // module UninitReg1 :
 //   input clock: Clock
 //   input reset : UInt<1>
 //   input cond: UInt<1>
 //   input value: UInt<2>
 //   reg count : UInt<2>, clock with :
 //     reset => (UInt<1>("h0"), count)
 //   node x = count
 //   node _GEN_0 = mux(cond, value, count)
 //   count <= mux(reset, UInt<2>("h0"), _GEN_0)

  // CHECK-LABEL: hw.module @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {

  firrtl.module @UninitReg1(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    // CHECK-NEXT: %count = sv.reg sym @count : !hw.inout<i2>
    // CHECK-NEXT: %0 = sv.read_inout %count : !hw.inout<i2>
    %count = firrtl.reg %clock {name = "count", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %RANDOM = sv.verbatim.expr.se "`RANDOM" : () -> i32
    // CHECK-NEXT:       %3 = comb.extract %RANDOM from 0 : (i32) -> i2
    // CHECK-NEXT:       sv.bpassign %count, %3 : i2
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    // CHECK-NEXT: %1 = comb.mux %cond, %value, %0 : i2
    // CHECK-NEXT: %2 = comb.mux %reset, %c0_i2, %1 : i2
    %4 = firrtl.mux(%cond, %value, %count) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %5 = firrtl.mux(%reset, %c0_ui2, %4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.alwaysff(posedge %clock)  {
    // CHECK-NEXT:   sv.passign %count, %2 : i2
    // CHECK-NEXT: }
    firrtl.connect %count, %5 : !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK-NEXT: hw.output
  }

  // module InitReg1 :
  //     input clock : Clock
  //     input reset : UInt<1>
  //     input io_d : UInt<32>
  //     output io_q : UInt<32>
  //     input io_en : UInt<1>
  //
  //     node _T = asAsyncReset(reset)
  //     reg reg : UInt<32>, clock with :
  //       reset => (_T, UInt<32>("h0"))
  //     io_q <= reg
  //     reg <= mux(io_en, io_d, reg)

  // CHECK-LABEL: hw.module @InitReg1(
  firrtl.module @InitReg1(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                          in %io_d: !firrtl.uint<32>, in %io_en: !firrtl.uint<1>,
                          out %io_q: !firrtl.uint<32>) {
    // CHECK: %c0_i32 = hw.constant 0 : i32
    %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>

    %4 = firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK-NEXT: %reg = sv.reg : !hw.inout<i32>
    // CHECK-NEXT: %0 = sv.read_inout %reg : !hw.inout<i32>
    // CHECK-NEXT: %reg2 = sv.reg : !hw.inout<i32>
    // CHECK-NEXT: %1 = sv.read_inout %reg2 : !hw.inout<i32>
    // CHECK-NEXT: sv.alwaysff(posedge %clock) {
    // CHECK-NEXT: }(syncreset : posedge %reset) {
    // CHECK-NEXT:    sv.passign %reg2, %c0_i32 : i32
    // CHECK-NEXT: }
    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.if %reset  {
    // CHECK-NEXT:       } else {
    // CHECK-NEXT:         %RANDOM = sv.verbatim.expr.se "`RANDOM" : () -> i32
    // CHECK-NEXT:         sv.bpassign %reg, %RANDOM : i32
    // CHECK-NEXT:         %RANDOM_0 = sv.verbatim.expr.se "`RANDOM" : () -> i32
    // CHECK-NEXT:         sv.bpassign %reg2, %RANDOM_0 : i32
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: %2 = comb.concat %false, %0 : i1, i32
    // CHECK-NEXT: %3 = comb.concat %false, %1 : i1, i32
    // CHECK-NEXT: %4 = comb.add %2, %3 : i33
    // CHECK-NEXT: %5 = comb.extract %4 from 1 : (i33) -> i32
    // CHECK-NEXT: %6 = comb.mux %io_en, %io_d, %5 : i32
    // CHECK-NEXT: sv.alwaysff(posedge %clock) {
    // CHECK-NEXT:   sv.passign %reg, %6 : i32
    // CHECK-NEXT: }(asyncreset : posedge %reset) {
    // CHECK-NEXT:   sv.passign %reg, %c0_i32 : i32
    // CHECK-NEXT: }
    %reg = firrtl.regreset %clock, %4, %c0_ui32 {name = "reg"} : !firrtl.asyncreset, !firrtl.uint<32>, !firrtl.uint<32>
    %reg2 = firrtl.regreset %clock, %reset, %c0_ui32 {name = "reg2"} : !firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>

    %sum = firrtl.add %reg, %reg2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %shorten = firrtl.head %sum, 32 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    %5 = firrtl.mux(%io_en, %io_d, %shorten) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>

    firrtl.connect %reg, %5 : !firrtl.uint<32>, !firrtl.uint<32>
    firrtl.connect %io_q, %reg: !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK-NEXT: hw.output %0 : i32
  }

  //  module MemSimple :
  //     input clock1  : Clock
  //     input clock2  : Clock
  //     input inpred  : UInt<1>
  //     input indata  : SInt<42>
  //     output result : SInt<42>
  //     output result2 : SInt<42>
  //
  //     mem _M : @[Decoupled.scala 209:27]
  //           data-type => SInt<42>
  //           depth => 12
  //           read-latency => 0
  //           write-latency => 1
  //           reader => read
  //           writer => write
  //           readwriter => rw
  //           read-under-write => undefined
  //
  //     result <= _M.read.data
  //     result2 <= _M.rw.rdata
  //
  //     _M.read.addr <= UInt<1>("h0")
  //     _M.read.en <= UInt<1>("h1")
  //     _M.read.clk <= clock1
  //     _M.rw.addr <= UInt<1>("h0")
  //     _M.rw.en <= UInt<1>("h1")
  //     _M.rw.clk <= clock1
  //     _M.rw.wmask <= UInt<1>("h1")
  //     _M.rw.wmode <= UInt<1>("h1")
  //     _M.write.addr <= validif(inpred, UInt<3>("h0"))
  //     _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
  //     _M.write.clk <= clock2
  //     _M.write.data <= validif(inpred, indata)
  //     _M.write.mask <= validif(inpred, UInt<1>("h1"))

  // CHECK-LABEL: hw.module @MemSimple(
  firrtl.module @MemSimple(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<42>,
                           out %result: !firrtl.sint<42>,
                           out %result2: !firrtl.sint<42>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
  // CHECK: %_M.R0_data, %_M.RW0_rdata = hw.instance "_M" @FIRRTLMem_1_1_1_42_12_0_1_1_0_1_a(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i4_0: i4, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %true: i1, RW0_wdata: %0: i42, RW0_wmask: %true: i1, W0_addr: %c0_i4_1: i4, W0_en: %inpred: i1, W0_clk: %clock2: i1, W0_data: %indata: i42, W0_mask: %true: i1) -> (R0_data: i42, RW0_rdata: i42)
  // CHECK: hw.output %_M.R0_data, %_M.RW0_rdata : i42, i42

      %0 = firrtl.subfield %_M_read(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.sint<42>
      firrtl.connect %result, %0 : !firrtl.sint<42>, !firrtl.sint<42>
      %1 = firrtl.subfield %_M_rw(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.sint<42>
      firrtl.connect %result2, %1 : !firrtl.sint<42>, !firrtl.sint<42>
      %2 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<4>
      firrtl.connect %2, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %3 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<1>
      firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.clock
      firrtl.connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = firrtl.subfield %_M_rw(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<4>
      firrtl.connect %5, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %6 = firrtl.subfield %_M_rw(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = firrtl.subfield %_M_rw(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.clock
      firrtl.connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = firrtl.subfield %_M_rw(6) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %8, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %9 = firrtl.subfield %_M_rw(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

      %10 = firrtl.subfield %_M_write(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<4>
      firrtl.connect %10, %c0_ui3 : !firrtl.uint<4>, !firrtl.uint<3>
      %11 = firrtl.subfield %_M_write(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %11, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %12 = firrtl.subfield %_M_write(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.clock
      firrtl.connect %12, %clock2 : !firrtl.clock, !firrtl.clock
      %13 = firrtl.subfield %_M_write(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.sint<42>
      firrtl.connect %13, %indata : !firrtl.sint<42>, !firrtl.sint<42>
      %14 = firrtl.subfield %_M_write(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>) -> !firrtl.uint<1>
      firrtl.connect %14, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @MemSimple_mask(
  firrtl.module @MemSimple_mask(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<40>,
                           out %result: !firrtl.sint<40>,
                           out %result2: !firrtl.sint<40>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui10 = firrtl.constant 0 : !firrtl.uint<10>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c1_ui5 = firrtl.constant 1 : !firrtl.uint<5>
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 1022 : i64, name = "_M_mask", portNames = ["read", "rw", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
    // CHECK: %_M_mask.R0_data, %_M_mask.RW0_rdata = hw.instance "_M_mask" @FIRRTLMem_1_1_1_40_1022_1_1_4_0_1_a(R0_addr: %c0_i10: i10, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i10: i10, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %true: i1, RW0_wdata: %0: i40, RW0_wmask: %c0_i4: i4, W0_addr: %c0_i10: i10, W0_en: %inpred: i1, W0_clk: %clock2: i1, W0_data: %indata: i40, W0_mask: %c0_i4: i4) -> (R0_data: i40, RW0_rdata: i40)
    // CHECK: hw.output %_M_mask.R0_data, %_M_mask.RW0_rdata : i40, i40

      %0 = firrtl.subfield %_M_read(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.sint<40>
      firrtl.connect %result, %0 : !firrtl.sint<40>, !firrtl.sint<40>
      %1 = firrtl.subfield %_M_rw(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.sint<40>
      firrtl.connect %result2, %1 : !firrtl.sint<40>, !firrtl.sint<40>
      %2 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>)      -> !firrtl.uint<10>
      firrtl.connect %2, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %3 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.uint<1>
      firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>) -> !firrtl.clock
      firrtl.connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = firrtl.subfield %_M_rw(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>,  wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<10>
      firrtl.connect %5, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %6 = firrtl.subfield %_M_rw(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = firrtl.subfield %_M_rw(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.clock
      firrtl.connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = firrtl.subfield %_M_rw(6) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<4>
      firrtl.connect %8, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
      %9 = firrtl.subfield %_M_rw(4) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

      %10 = firrtl.subfield %_M_write(0) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>,
      mask: uint<4>>) -> !firrtl.uint<10>
      firrtl.connect %10, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %11 = firrtl.subfield %_M_write(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.uint<1>
      firrtl.connect %11, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %12 = firrtl.subfield %_M_write(2) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.clock
      firrtl.connect %12, %clock2 : !firrtl.clock, !firrtl.clock
      %13 = firrtl.subfield %_M_write(3) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.sint<40>
      firrtl.connect %13, %indata : !firrtl.sint<40>, !firrtl.sint<40>
      %14 = firrtl.subfield %_M_write(4) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>) -> !firrtl.uint<4>
      firrtl.connect %14, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
  }
  // CHECK-LABEL: hw.module @IncompleteRead(
  // The read port has no use of the data field.
  firrtl.module @IncompleteRead(in %clock1: !firrtl.clock) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK:  %_M.R0_data = hw.instance "_M" @FIRRTLMem_1_0_0_42_12_0_1_0_0_1(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1) -> (R0_data: i42)
    %_M_read = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    // Read port.
    %6 = firrtl.subfield %_M_read(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<4>
    firrtl.connect %6, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
    %7 = firrtl.subfield %_M_read(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.uint<1>
    firrtl.connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = firrtl.subfield %_M_read(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>) -> !firrtl.clock
    firrtl.connect %8, %clock1 : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module @top_modx() -> (tmp27: i23) {
  // CHECK-NEXT:    %c0_i23 = hw.constant 0 : i23
  // CHECK-NEXT:    %c42_i23 = hw.constant 42 : i23
  // CHECK-NEXT:    hw.output %c0_i23 : i23
  // CHECK-NEXT:  }
  firrtl.module @top_modx(out %tmp27: !firrtl.uint<23>) {
    %0 = firrtl.wire : !firrtl.uint<0>
    %c42_ui23 = firrtl.constant 42 : !firrtl.uint<23>
    %1 = firrtl.tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    firrtl.connect %0, %1 : !firrtl.uint<0>, !firrtl.uint<0>
    %2 = firrtl.head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = firrtl.pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    firrtl.connect %tmp27, %3 : !firrtl.uint<23>, !firrtl.uint<23>
  }

  //CHECK-LABEL: hw.module @test_partialconnect(%clock: i1) {
  //CHECK: sv.alwaysff(posedge %clock)
  firrtl.module @test_partialconnect(in %clock : !firrtl.clock) {
    %b = firrtl.reg %clock {name = "pcon"} : !firrtl.uint<1>
    %a = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.partialconnect %b, %a : !firrtl.uint<1>, !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module @SimpleStruct(%source: !hw.struct<valid: i1, ready: i1, data: i64>) -> (fldout: i64) {
  // CHECK-NEXT:    %0 = hw.struct_extract %source["data"] : !hw.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:    hw.output %0 : i64
  // CHECK-NEXT:  }
  firrtl.module @SimpleStruct(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              out %fldout: !firrtl.uint<64>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.connect %fldout, %2 : !firrtl.uint<64>, !firrtl.uint<64>
  }

  // CHECK-LABEL: IsInvalidIssue572
  // https://github.com/llvm/circt/issues/572
  firrtl.module @IsInvalidIssue572(in %a: !firrtl.analog<1>) {
    // CHECK-NEXT: %0 = sv.read_inout %a : !hw.inout<i1>

    // CHECK-NEXT: %.invalid_analog = sv.wire : !hw.inout<i1>
    // CHECK-NEXT: %1 = sv.read_inout %.invalid_analog : !hw.inout<i1>
    %0 = firrtl.invalidvalue : !firrtl.analog<1>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   sv.assign %a, %1 : i1
    // CHECK-NEXT:   sv.assign %.invalid_analog, %0 : i1
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "verilator" {
    // CHECK-NEXT:     sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.alias %a, %.invalid_analog : !hw.inout<i1>, !hw.inout<i1>
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.attach %a, %0 : !firrtl.analog<1>, !firrtl.analog<1>
  }

  // CHECK-LABEL: IsInvalidIssue654
  // https://github.com/llvm/circt/issues/654
  firrtl.module @IsInvalidIssue654() {
    %w = firrtl.wire : !firrtl.uint<0>
    %0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.connect %w, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: ASQ
  // https://github.com/llvm/circt/issues/699
  firrtl.module @ASQ(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %widx_widx_bin = firrtl.regreset %clock, %reset, %c0_ui1 {name = "widx_widx_bin"} : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module @Struct0bits(%source: !hw.struct<valid: i1, ready: i1, data: i0>) {
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }
  firrtl.module @Struct0bits(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) -> !firrtl.uint<0>
  }

  // CHECK-LABEL: hw.module @MemDepth1
  firrtl.module @MemDepth1(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %addr: !firrtl.uint<1>, out %data: !firrtl.uint<32>) {
    // CHECK: %mem0.R0_data = hw.instance "mem0" @FIRRTLMem_1_0_0_32_1_0_1_0_1_1(R0_addr: %addr: i1, R0_en: %en: i1, R0_clk: %clock: i1) -> (R0_data: i32)
    // CHECK: hw.output %mem0.R0_data : i32
    %mem0_load0 = firrtl.mem Old {depth = 1 : i64, name = "mem0", portNames = ["load0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    %0 = firrtl.subfield %mem0_load0(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %mem0_load0(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<1>
    firrtl.connect %1, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %mem0_load0(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<32>
    firrtl.connect %data, %2 : !firrtl.uint<32>, !firrtl.uint<32>
    %3 = firrtl.subfield %mem0_load0(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>) -> !firrtl.uint<1>
    firrtl.connect %3, %en : !firrtl.uint<1>, !firrtl.uint<1>
}

  // https://github.com/llvm/circt/issues/1115
  // CHECK-LABEL: hw.module @issue1115
  firrtl.module @issue1115(in %a: !firrtl.sint<20>, out %tmp59: !firrtl.sint<2>) {
    %0 = firrtl.shr %a, 21 : (!firrtl.sint<20>) -> !firrtl.sint<1>
    firrtl.connect %tmp59, %0 : !firrtl.sint<2>, !firrtl.sint<1>
  }

   // CHECK-LABEL: hw.module @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {

  firrtl.module @UninitReg42(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<42>) {
    %c0_ui42 = firrtl.constant 0 : !firrtl.uint<42>
    // CHECK: %count = sv.reg sym @count : !hw.inout<i42>
    %count = firrtl.reg %clock {name = "count", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<42>

    // CHECK: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       %RANDOM = sv.verbatim.expr.se "`RANDOM" : () -> i32
    // CHECK-NEXT:       %RANDOM_0 = sv.verbatim.expr.se "`RANDOM" : () -> i32
    // CHECK-NEXT:       %3 = comb.extract %RANDOM_0 from 0 : (i32) -> i10
    // CHECK-NEXT:       %4 = comb.concat %RANDOM, %3 : i32, i10
    // CHECK-NEXT:       sv.bpassign %count, %4 : i42
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    %4 = firrtl.mux(%cond, %value, %count) : (!firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>) -> !firrtl.uint<42>
    %5 = firrtl.mux(%reset, %c0_ui42, %4) : (!firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>) -> !firrtl.uint<42>

    firrtl.connect %count, %5 : !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: issue1303
  firrtl.module @issue1303(out %out: !firrtl.reset) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %out, %c1_ui : !firrtl.reset, !firrtl.uint<1>
    // CHECK-NEXT: %true = hw.constant true
    // CHECK-NEXT: hw.output %true
  }

  // CHECK-LABEL: issue1594
  // Make sure LowerToHW's merging of always blocks kicks in for this example.
  firrtl.module @issue1594(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %reset_n = firrtl.wire  : !firrtl.uint<1>
    %0 = firrtl.not %reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %reset_n, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset_n, %c0_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %r : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: sv.alwaysff(posedge %clock)
    // CHECK-NOT: sv.alwaysff
    // CHECK: hw.output
  }

  // CHECK-LABEL: hw.module @Force
  firrtl.module @Force(in %in: !firrtl.uint<42>) {
    // CHECK: %out = sv.wire : !hw.inout<i42>
    // CHECK: sv.initial {
    // CHECK:   sv.force %out, %in : i42
    // CHECK: }
    %out = firrtl.wire : !firrtl.uint<42>
    firrtl.force %out, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: hw.module @FooDUT
  // CHECK: attributes {firrtl.moduleHierarchyFile
  firrtl.module @FooDUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}

  // CHECK-LABEL: hw.module @MemoryWritePortBehavior
  firrtl.module @MemoryWritePortBehavior(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock) {
    // This memory has both write ports driven by the same clock.  It should be
    // lowered to an "aa" memory.
    //
    // CHECK: hw.instance "aa" @FIRRTLMem_0_2_0_8_16_1_1_1_0_1_aa
    %memory_aa_w0, %memory_aa_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "aa", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_aa_w0 = firrtl.subfield %memory_aa_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_aa_w1 = firrtl.subfield %memory_aa_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %clk_aa_w0, %clock1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %clk_aa_w1, %clock1 : !firrtl.clock, !firrtl.clock

    // This memory has different clocks for each write port.  It should be
    // lowered to an "ab" memory.
    //
    // CHECK: hw.instance "ab" @FIRRTLMem_0_2_0_8_16_1_1_1_0_1_ab
    %memory_ab_w0, %memory_ab_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "ab", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_w0 = firrtl.subfield %memory_ab_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_ab_w1 = firrtl.subfield %memory_ab_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %clk_ab_w0, %clock1 : !firrtl.clock, !firrtl.clock
    firrtl.connect %clk_ab_w1, %clock2 : !firrtl.clock, !firrtl.clock

    // This memory is the same as the first memory, but a node is used to alias
    // the second write port clock (e.g., this could be due to a dont touch
    // annotation blocking this from being optimized away).  This should be
    // lowered to an "ab" memory even though it is trivially convertible to an
    // "aa" memory.
    //
    // CHECK: hw.instance "ab_node" @FIRRTLMem_0_2_0_8_16_1_1_1_0_1_ab
    %memory_ab_node_w0, %memory_ab_node_w1 = firrtl.mem Undefined {depth = 16 : i64, name = "ab_node", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_node_w0 = firrtl.subfield %memory_ab_node_w0(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %clk_ab_node_w1 = firrtl.subfield %memory_ab_node_w1(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %clk_ab_node_w0, %clock1 : !firrtl.clock, !firrtl.clock
    %tmp = firrtl.node %clock1 : !firrtl.clock
    firrtl.connect %clk_ab_node_w1, %tmp : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module @AsyncResetBasic(
  firrtl.module @AsyncResetBasic(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset, in %srst: !firrtl.uint<1>) {
    %c9_ui42 = firrtl.constant 9 : !firrtl.uint<42>
    %c-9_si42 = firrtl.constant -9 : !firrtl.sint<42>
    // The following should not error because the reset values are constant.
    %r0 = firrtl.regreset %clock, %arst, %c9_ui42 : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %r1 = firrtl.regreset %clock, %srst, %c9_ui42 : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    %r2 = firrtl.regreset %clock, %arst, %c-9_si42 : !firrtl.asyncreset, !firrtl.sint<42>, !firrtl.sint<42>
    %r3 = firrtl.regreset %clock, %srst, %c-9_si42 : !firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>
  }

  // CHECK-LABEL: hw.module @AsyncResetThroughWires(
  firrtl.module @AsyncResetThroughWires(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset) {
    %c9000_ui42 = firrtl.constant 9000 : !firrtl.uint<42>
    %c9001_ui42 = firrtl.constant 9001 : !firrtl.uint<42>
    %constWire = firrtl.wire : !firrtl.uint<42>
    firrtl.connect %constWire, %c9000_ui42 : !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %constWire, %c9001_ui42 : !firrtl.uint<42>, !firrtl.uint<42>
    // The following should not error because the reset values are constant.
    %r0 = firrtl.regreset %clock, %arst, %constWire : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: hw.module @AsyncResetThroughNodes(
  firrtl.module @AsyncResetThroughNodes(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset) {
    %c1337_ui42 = firrtl.constant 1337 : !firrtl.uint<42>
    %constNode = firrtl.node %c1337_ui42 : !firrtl.uint<42>
    // The following should not error because the reset values are constant.
    %r0 = firrtl.regreset %clock, %arst, %constNode : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: hw.module @BitCast1
  firrtl.module @BitCast1() {
    %a = firrtl.wire : !firrtl.vector<uint<2>, 13>
    %b = firrtl.bitcast %a : (!firrtl.vector<uint<2>, 13>) -> (!firrtl.uint<26>)
    // CHECK: hw.bitcast %0 : (!hw.array<13xi2>) -> i26 
  }

  // CHECK-LABEL: hw.module @BitCast2
  firrtl.module @BitCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i1, ready: i1, data: i1>) -> i3

  }
  // CHECK-LABEL: hw.module @BitCast3
  firrtl.module @BitCast3() {
    %a = firrtl.wire : !firrtl.uint<26>
    %b = firrtl.bitcast %a : (!firrtl.uint<26>) -> (!firrtl.vector<uint<2>, 13>)
    // CHECK: hw.bitcast %0 : (i26) -> !hw.array<13xi2>
  }

  // CHECK-LABEL: hw.module @BitCast4
  firrtl.module @BitCast4() {
    %a = firrtl.wire : !firrtl.uint<3>
    %b = firrtl.bitcast %a : (!firrtl.uint<3>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
    // CHECK: hw.bitcast %0 : (i3) -> !hw.struct<valid: i1, ready: i1, data: i1>

  }
  // CHECK-LABEL: hw.module @BitCast5
  firrtl.module @BitCast5() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>) -> (!firrtl.vector<uint<2>, 3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i2, ready: i1, data: i3>) -> !hw.array<3xi2>
  }

  firrtl.extmodule @chkcoverAnno(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
}
