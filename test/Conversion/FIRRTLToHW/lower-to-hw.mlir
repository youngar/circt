// RUN: circt-opt -lower-firrtl-to-hw -verify-diagnostics %s | FileCheck %s

firrtl.circuit "Simple"   attributes {annotations = [{class =
"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation", directory = "dir1",  filename = "./dir1/filename1" }, {class =
"sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "dir2",  filename = "./dir2/filename2" }, {class =
"sifive.enterprise.firrtl.ExtractAssertionsAnnotation", directory = "dir3",  filename = "./dir3/filename3" }]}
{

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

    // CHECK: [[PADRES_SIGN:%.+]] = comb.extract %in2 from 1 : (i2) -> i1
    // CHECK: [[PADRES:%.+]] = comb.concat  [[PADRES_SIGN]], %in2 : i1, i2
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

    // CHECK: [[IN3SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: [[PADRESSEXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i6, i3
    // CHECK-NEXT: = comb.divs [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = firrtl.div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK: [[IN3EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK-NEXT: [[MOD1:%.+]] = comb.mods %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = comb.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = firrtl.rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK: [[IN4EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
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

    // CHECK-NEXT: [[WIRE:%n3]] = sv.wire sym @nodeSym : !hw.inout<i2>
    // CHECK-NEXT: sv.assign [[WIRE]], %in2 : i2
    %n3 = firrtl.node sym @nodeSym %in2 : !firrtl.uint<2>

    // CHECK-NEXT: [[CVT:%.+]] = comb.concat %false, %in2 : i1, i2
    %23 = firrtl.cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = firrtl.cvt %in3 : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: [[XOR:%.+]] = comb.xor [[CVT]], %c-1_i3 : i3
    %24 = firrtl.not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = firrtl.asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, [[XOR]] : i1, i3
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub %c0_i4, [[SEXT]] : i4
    %25 = firrtl.neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK: [[CVT4:%.+]] = comb.concat {{.*}}, [[CVT]] : i1, i3
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

    // CHECK: = comb.concat {{.*}}, %in3 : i7, i8
    // CHECK-NEXT: = comb.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shl {{.*}}, {{.*}} : i15
    %30 = firrtl.dshl %in3, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = comb.shl [[DSHR]], [[DSHR]] : i3
    %dshlw = firrtl.dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK: = comb.concat {{.*}} : i10, i4
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
    // CHECK: [[VERB2:%.+]] = sv.verbatim.expr "$bits({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> i32 {symbols = [@Simple]}
    // CHECK: [[VERB3:%.+]] = sv.verbatim.expr.se "$size({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> !hw.inout<i32> {symbols = [@Simple]}
    // CHECK: [[VERB3READ:%.+]] = sv.read_inout [[VERB3]]
    // CHECK: [[VERB1EXT:%.+]] = comb.concat {{%.+}}, [[VERB1]] : i1, i42
    // CHECK: [[VERB2EXT:%.+]] = comb.concat {{%.+}}, [[VERB2]] : i11, i32
    // CHECK: [[ADD:%.+]] = comb.add [[VERB1EXT]], [[VERB2EXT]] : i43
    // CHECK: [[VERB3EXT:%.+]] = comb.concat {{%.+}}, [[VERB3READ]] : i12, i32
    // CHECK: [[ADDEXT:%.+]] = comb.concat {{%.+}}, [[ADD]] : i1, i43
    // CHECK: = comb.add [[VERB3EXT]], [[ADDEXT]] : i44
    %56 = firrtl.verbatim.expr "MAGIC_CONSTANT" : () -> !firrtl.uint<42>
    %57 = firrtl.verbatim.expr "$bits({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %58 = firrtl.verbatim.wire "$size({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %59 = firrtl.add %56, %57 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
    %60 = firrtl.add %58, %59 : (!firrtl.uint<32>, !firrtl.uint<43>) -> !firrtl.uint<44>

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
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
    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: = comb.sub %c0_i9, [[SEXT]] : i9
    %54 = firrtl.neg %in3 : (!firrtl.sint<8>) -> !firrtl.sint<9>
    firrtl.connect %out1, %53 : !firrtl.sint<1>, !firrtl.sint<1>
    %55 = firrtl.neg %in5 : (!firrtl.sint<0>) -> !firrtl.sint<1>

    %61 = firrtl.multibit_mux %17, %55, %55, %55 : !firrtl.uint<1>, !firrtl.sint<1>
    // CHECK:      %[[ZEXT_INDEX:.+]] = comb.concat %false, {{.*}} : i1, i1
    // CHECK-NEXT: %[[ARRAY:.+]] = hw.array_create %false, %false, %false
    // CHECK-NEXT: %[[ARRAY_GET:.+]] = hw.array_get %[[ARRAY]][%[[ZEXT_INDEX]]]
    // CHECK-NEXT: %[[ARRAY_ZEROTH:.+]] = hw.array_get %[[ARRAY]][%c0_i2]
    // CHECK-NEXT: %[[IS_OOB:.+]] = comb.icmp uge %[[ZEXT_INDEX]], %c-1_i2
    // CHECK-NEXT: %[[GUARDED:.+]] = comb.mux %[[IS_OOB]], %[[ARRAY_ZEROTH]], %[[ARRAY_GET]]
    // CHECK: hw.output %false, %[[GUARDED]] : i1, i1
    firrtl.connect %out2, %61 : !firrtl.sint<1>, !firrtl.sint<1>
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: hw.module private @Print
  firrtl.module private @Print(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                       in %a: !firrtl.uint<4>, in %b: !firrtl.uint<4>) {
    // CHECK: [[ADD:%.+]] = comb.add

    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else  {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     %PRINTF_COND_ = sv.macro.ref< "PRINTF_COND_"> : i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND_, %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:     %PRINTF_COND__0 = sv.macro.ref< "PRINTF_COND_"> : i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and %PRINTF_COND__0, %reset : i1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "Hi %x %x\0A"(%2, %b) : i5, i4
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

  // CHECK-LABEL: hw.module private @Stop
  firrtl.module private @Stop(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock, in %reset: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock1 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref< "STOP_COND_"> : i1
    // CHECK-NEXT:     %0 = comb.and %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    firrtl.stop %clock1, %reset, 42

    // CHECK-NEXT:   sv.always posedge %clock2 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref< "STOP_COND_"> : i1
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

  // CHECK-LABEL: hw.module private @Verification
  firrtl.module private @Verification(in %clock: !firrtl.clock, in %aCond: !firrtl.uint<1>,
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
    // CHECK-NEXT: [[SAMPLED:%.+]] =  sv.system.sampled %value : i42
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP5:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP6:%.+]] = comb.or [[TMP5]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP6]] message "assert0"([[SAMPLED]]) : i42
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
    // CHECK-NEXT: [[SAMPLED:%.+]] = sv.system.sampled %value
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"([[SAMPLED]]) : i42
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

  // CHECK-LABEL: hw.module private @VerificationGuards
  firrtl.module private @VerificationGuards(
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

  // CHECK-LABEL: hw.module private @VerificationAssertFormat
  firrtl.module private @VerificationAssertFormat(
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
    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     sv.if [[TMP2]] {
    // CHECK-NEXT:       [[ASSERT_VERBOSE_COND:%.+]] = sv.macro.ref< "ASSERT_VERBOSE_COND_"> : i1
    // CHECK-NEXT:       sv.if [[ASSERT_VERBOSE_COND]] {
    // CHECK-NEXT:         sv.error "assert1"(%value) : i42
    // CHECK-NEXT:       }
    // CHECK-NEXT:       [[STOP_COND:%.+]] = sv.macro.ref< "STOP_COND_"> : i1
    // CHECK-NEXT:       sv.if [[STOP_COND]] {
    // CHECK-NEXT:         sv.fatal
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  firrtl.module private @bar(in %io_cpu_flush: !firrtl.uint<1>) {
    // CHECK: hw.probe @baz, %io_cpu_flush, %io_cpu_flush : i1, i1
    firrtl.probe @baz, %io_cpu_flush, %io_cpu_flush  : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @foo
  firrtl.module private @foo() {
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

  // CHECK: sv.bind <@bindTest::@[[bazSymbol:.+]]>
  // CHECK-NOT: output_file
  // CHECK-NEXT: sv.bind <@bindTest::@[[quxSymbol:.+]]> {
  // CHECK-SAME: output_file = #hw.output_file<"outputDir/bindings.sv", excludeFromFileList>
  // CHECK-NEXT: hw.module private @bindTest()
  firrtl.module private @bindTest() {
    // CHECK: hw.instance "baz" sym @[[bazSymbol]] @bar
    %baz = firrtl.instance baz {lowerToBind = true} @bar(in io_cpu_flush: !firrtl.uint<1>)
    // CHECK: hw.instance "qux" sym @[[quxSymbol]] @bar
    %qux = firrtl.instance qux {lowerToBind = true, output_file = #hw.output_file<"outputDir/bindings.sv", excludeFromFileList>} @bar(in io_cpu_flush: !firrtl.uint<1>)
  }


  // CHECK-LABEL: hw.module private @output_fileTest
  // CHECK-SAME: output_file = #hw.output_file<"output_fileTest/dir/output_fileTest.sv", excludeFromFileList>
  firrtl.module private @output_fileTest() attributes {output_file = #hw.output_file<
    "output_fileTest/dir/output_fileTest.sv", excludeFromFileList
  >} {
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: hw.module private @issue314
  firrtl.module private @issue314(in %inp_2: !firrtl.uint<27>, in %inpi: !firrtl.uint<65>) {
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
  // CHECK-LABEL: hw.module private @test_rem
  // CHECK-NEXT:     %0 = comb.modu
  // CHECK-NEXT:     hw.output %0
  firrtl.module private @test_rem(in %tmp85: !firrtl.uint<1>, in %tmp79: !firrtl.uint<1>,
       out %out: !firrtl.uint<1>) {
    %2 = firrtl.rem %tmp79, %tmp85 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @Analog(%a1: !hw.inout<i1>, %b1: !hw.inout<i1>,
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
  firrtl.module private @Analog(in %a1: !firrtl.analog<1>, in %b1: !firrtl.analog<1>,
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

  // CHECK-LABEL: hw.module private @UninitReg1(%clock: i1, %reset: i1, %cond: i1, %value: i2) {

  firrtl.module private @UninitReg1(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    // CHECK-NEXT: %count = sv.reg sym @count : !hw.inout<i2>
    // CHECK-NEXT: %0 = sv.read_inout %count : !hw.inout<i2>
    %count = firrtl.reg %clock {name = "count", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:    sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:      %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:    }
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg1::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[1:0];" {symbols = [#hw.innerNameRef<@UninitReg1::@count>, #hw.innerNameRef<@UninitReg1::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    // CHECK-NEXT: %1 = comb.mux %cond, %value, %0 : i2
    // CHECK-NEXT: %2 = comb.mux %reset, %c0_i2, %1 : i2
    %4 = firrtl.mux(%cond, %value, %count) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    %5 = firrtl.mux(%reset, %c0_ui2, %4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>

    // CHECK-NEXT: sv.always posedge %clock {
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

  // CHECK-LABEL: hw.module private @InitReg1(
  firrtl.module private @InitReg1(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                          in %io_d: !firrtl.uint<32>, in %io_en: !firrtl.uint<1>,
                          out %io_q: !firrtl.uint<32>) {
    // CHECK:      %c1_i32 = hw.constant 1 : i32
    // CHECK-NEXT: %c0_i32 = hw.constant 0 : i32
    %c0_ui32 = firrtl.constant 0 : !firrtl.uint<32>
    %c1_ui32 = firrtl.constant 1 : !firrtl.uint<32>

    %4 = firrtl.asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK-NEXT: %reg = sv.reg sym @[[reg_sym:.+]] : !hw.inout<i32>
    // CHECK-NEXT: %0 = sv.read_inout %reg : !hw.inout<i32>
    // CHECK-NEXT: %reg2 = sv.reg sym @[[reg2_sym:.+]] : !hw.inout<i32>
    // CHECK-NEXT: %1 = sv.read_inout %reg2 : !hw.inout<i32>
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:     sv.passign %reg2, %c0_i32 : i32
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: %reg3 = sv.reg sym @[[reg3_sym:.+]] : !hw.inout<i32
    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:     %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:     %[[RANDOM_2:.+]] = sv.reg sym @[[RANDOM_2_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:     %[[RANDOM_3:.+]] = sv.reg sym @[[RANDOM_3_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_2_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg2_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_2_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@InitReg1::@[[RANDOM_3_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}};" {symbols = [#hw.innerNameRef<@InitReg1::@[[reg3_sym]]>, #hw.innerNameRef<@InitReg1::@[[RANDOM_3_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:     sv.if %reset {
    // CHECK-NEXT:       sv.bpassign %reg, %c0_i32 : i32
    // CHECK-NEXT:       sv.bpassign %reg3, %c1_i32 : i32
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: %2 = comb.concat %false, %0 : i1, i32
    // CHECK-NEXT: %3 = comb.concat %false, %1 : i1, i32
    // CHECK-NEXT: %4 = comb.add %2, %3 : i33
    // CHECK-NEXT: %5 = comb.extract %4 from 1 : (i33) -> i32
    // CHECK-NEXT: %6 = comb.mux %io_en, %io_d, %5 : i32
    // CHECK-NEXT: sv.always posedge %clock, posedge %reset  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:     sv.passign %reg, %c0_i32 : i32
    // CHECK-NEXT:     sv.passign %reg3, %c1_i32 : i32
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:     sv.passign %reg, %6 : i32
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    %reg = firrtl.regreset %clock, %4, %c0_ui32 {name = "reg"} : !firrtl.asyncreset, !firrtl.uint<32>, !firrtl.uint<32>
    %reg2 = firrtl.regreset %clock, %reset, %c0_ui32 {name = "reg2"} : !firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>
    %reg3 = firrtl.regreset %clock, %4, %c1_ui32 {name = "reg3"} : !firrtl.asyncreset, !firrtl.uint<32>, !firrtl.uint<32>

    %sum = firrtl.add %reg, %reg2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>
    %shorten = firrtl.head %sum, 32 : (!firrtl.uint<33>) -> !firrtl.uint<32>
    %5 = firrtl.mux(%io_en, %io_d, %shorten) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>

    firrtl.connect %reg, %5 : !firrtl.uint<32>, !firrtl.uint<32>
    firrtl.connect %io_q, %reg: !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK-NEXT: hw.output %0 : i32
  }


  // CHECK-LABEL: hw.module private @top_modx() -> (tmp27: i23) {
  // CHECK-NEXT:    %c0_i23 = hw.constant 0 : i23
  // CHECK-NEXT:    %c42_i23 = hw.constant 42 : i23
  // CHECK-NEXT:    hw.output %c0_i23 : i23
  // CHECK-NEXT:  }
  firrtl.module private @top_modx(out %tmp27: !firrtl.uint<23>) {
    %0 = firrtl.wire : !firrtl.uint<0>
    %c42_ui23 = firrtl.constant 42 : !firrtl.uint<23>
    %1 = firrtl.tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    firrtl.connect %0, %1 : !firrtl.uint<0>, !firrtl.uint<0>
    %2 = firrtl.head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = firrtl.pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    firrtl.connect %tmp27, %3 : !firrtl.uint<23>, !firrtl.uint<23>
  }

  // CHECK-LABEL: hw.module private @SimpleStruct(%source: !hw.struct<valid: i1, ready: i1, data: i64>) -> (fldout: i64) {
  // CHECK-NEXT:    %0 = hw.struct_extract %source["data"] : !hw.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:    hw.output %0 : i64
  // CHECK-NEXT:  }
  firrtl.module private @SimpleStruct(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              out %fldout: !firrtl.uint<64>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
    firrtl.connect %fldout, %2 : !firrtl.uint<64>, !firrtl.uint<64>
  }

  // CHECK-LABEL: IsInvalidIssue572
  // https://github.com/llvm/circt/issues/572
  firrtl.module private @IsInvalidIssue572(in %a: !firrtl.analog<1>) {
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
  firrtl.module private @IsInvalidIssue654() {
    %w = firrtl.wire : !firrtl.uint<0>
    %0 = firrtl.invalidvalue : !firrtl.uint<0>
    firrtl.connect %w, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: ASQ
  // https://github.com/llvm/circt/issues/699
  firrtl.module private @ASQ(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %widx_widx_bin = firrtl.regreset %clock, %reset, %c0_ui1 {name = "widx_widx_bin"} : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module private @Struct0bits(%source: !hw.struct<valid: i1, ready: i1, data: i0>) {
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }
  firrtl.module private @Struct0bits(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) {
    %2 = firrtl.subfield %source (2) : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) -> !firrtl.uint<0>
  }

  // https://github.com/llvm/circt/issues/1115
  // CHECK-LABEL: hw.module private @issue1115
  firrtl.module private @issue1115(in %a: !firrtl.sint<20>, out %tmp59: !firrtl.sint<2>) {
    %0 = firrtl.shr %a, 21 : (!firrtl.sint<20>) -> !firrtl.sint<1>
    firrtl.connect %tmp59, %0 : !firrtl.sint<2>, !firrtl.sint<1>
  }

   // CHECK-LABEL: hw.module private @UninitReg42(%clock: i1, %reset: i1, %cond: i1, %value: i42) {

  firrtl.module private @UninitReg42(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<42>) {
    %c0_ui42 = firrtl.constant 0 : !firrtl.uint<42>
    // CHECK: %count = sv.reg sym @count : !hw.inout<i42>
    %count = firrtl.reg %clock {name = "count", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<42>

    // CHECK: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:    sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:      %[[RANDOM_0:.+]] = sv.reg sym @[[RANDOM_0_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:      %[[RANDOM_1:.+]] = sv.reg sym @[[RANDOM_1_SYM:[_A-Za-z0-9]+]] {{.+}}
    // CHECK-NEXT:    }
    // CHECK-NEXT:    sv.initial {
    // CHECK-NEXT:    sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:    sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg42::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@UninitReg42::@[[RANDOM_1_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{][{]1[}][}]}}[9:0], {{[{][{]2[}][}][}]}};" {symbols = [#hw.innerNameRef<@UninitReg42::@count>, #hw.innerNameRef<@UninitReg42::@[[RANDOM_1_SYM]]>, #hw.innerNameRef<@UninitReg42::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }

    %4 = firrtl.mux(%cond, %value, %count) : (!firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>) -> !firrtl.uint<42>
    %5 = firrtl.mux(%reset, %c0_ui42, %4) : (!firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>) -> !firrtl.uint<42>

    firrtl.connect %count, %5 : !firrtl.uint<42>, !firrtl.uint<42>
  }

  // CHECK-LABEL: issue1303
  firrtl.module private @issue1303(out %out: !firrtl.reset) {
    %c1_ui = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %out, %c1_ui : !firrtl.reset, !firrtl.uint<1>
    // CHECK-NEXT: %true = hw.constant true
    // CHECK-NEXT: hw.output %true
  }

  // CHECK-LABEL: issue1594
  // Make sure LowerToHW's merging of always blocks kicks in for this example.
  firrtl.module private @issue1594(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %reset_n = firrtl.wire  : !firrtl.uint<1>
    %0 = firrtl.not %reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %reset_n, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = firrtl.regreset %clock, %reset_n, %c0_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %b, %r : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: sv.always posedge %clock
    // CHECK-NOT: sv.always
    // CHECK: hw.output
  }

  // CHECK-LABEL: hw.module private @Force
  firrtl.module private @Force(in %in: !firrtl.uint<42>) {
    // CHECK: %out = sv.wire : !hw.inout<i42>
    // CHECK: sv.initial {
    // CHECK:   sv.force %out, %in : i42
    // CHECK: }
    %out = firrtl.wire : !firrtl.uint<42>
    firrtl.force %out, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }

  firrtl.extmodule @chkcoverAnno(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // chckcoverAnno is extracted because it is instantiated inside the DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno(%clock: i1)
  // CHECK-SAME: attributes {firrtl.extract.cover.extra}

  firrtl.extmodule @chkcoverAnno2(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // checkcoverAnno2 is NOT extracted because it is not instantiated under the
  // DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno2(%clock: i1)
  // CHECK-NOT: attributes {firrtl.extract.cover.extra}

  // CHECK-LABEL: hw.module.extern @InnerNamesExt
  // CHECK-SAME:  (
  // CHECK-SAME:    clockIn: i1 {hw.exportPort = @extClockInSym}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    clockOut: i1 {hw.exportPort = @extClockOutSym}
  // CHECK-SAME:  )
  firrtl.extmodule @InnerNamesExt(
    in clockIn: !firrtl.clock sym @extClockInSym,
    out clockOut: !firrtl.clock sym @extClockOutSym
  )
  attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}

  // CHECK-LABEL: hw.module private @FooDUT
  firrtl.module private @FooDUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %chckcoverAnno_clock = firrtl.instance chkcoverAnno @chkcoverAnno(in clock: !firrtl.clock)
  }

  // CHECK-LABEL: hw.module private @AsyncResetBasic(
  firrtl.module private @AsyncResetBasic(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset, in %srst: !firrtl.uint<1>) {
    %c9_ui42 = firrtl.constant 9 : !firrtl.uint<42>
    %c-9_si42 = firrtl.constant -9 : !firrtl.sint<42>
    // The following should not error because the reset values are constant.
    %r0 = firrtl.regreset %clock, %arst, %c9_ui42 : !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %r1 = firrtl.regreset %clock, %srst, %c9_ui42 : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    %r2 = firrtl.regreset %clock, %arst, %c-9_si42 : !firrtl.asyncreset, !firrtl.sint<42>, !firrtl.sint<42>
    %r3 = firrtl.regreset %clock, %srst, %c-9_si42 : !firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>
  }

  // CHECK-LABEL: hw.module private @BitCast1
  firrtl.module private @BitCast1() {
    %a = firrtl.wire : !firrtl.vector<uint<2>, 13>
    %b = firrtl.bitcast %a : (!firrtl.vector<uint<2>, 13>) -> (!firrtl.uint<26>)
    // CHECK: hw.bitcast %0 : (!hw.array<13xi2>) -> i26
  }

  // CHECK-LABEL: hw.module private @BitCast2
  firrtl.module private @BitCast2() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i1, ready: i1, data: i1>) -> i3

  }
  // CHECK-LABEL: hw.module private @BitCast3
  firrtl.module private @BitCast3() {
    %a = firrtl.wire : !firrtl.uint<26>
    %b = firrtl.bitcast %a : (!firrtl.uint<26>) -> (!firrtl.vector<uint<2>, 13>)
    // CHECK: hw.bitcast %0 : (i26) -> !hw.array<13xi2>
  }

  // CHECK-LABEL: hw.module private @BitCast4
  firrtl.module private @BitCast4() {
    %a = firrtl.wire : !firrtl.uint<3>
    %b = firrtl.bitcast %a : (!firrtl.uint<3>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
    // CHECK: hw.bitcast %0 : (i3) -> !hw.struct<valid: i1, ready: i1, data: i1>

  }
  // CHECK-LABEL: hw.module private @BitCast5
  firrtl.module private @BitCast5() {
    %a = firrtl.wire : !firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>
    %b = firrtl.bitcast %a : (!firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>) -> (!firrtl.vector<uint<2>, 3>)
    // CHECK: hw.bitcast %0 : (!hw.struct<valid: i2, ready: i1, data: i3>) -> !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module private @InnerNames
  // CHECK-SAME:  (
  // CHECK-SAME:    %value: i42 {hw.exportPort = @portValueSym}
  // CHECK-SAME:    %clock: i1 {hw.exportPort = @portClockSym}
  // CHECK-SAME:    %reset: i1 {hw.exportPort = @portResetSym}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    out: i1 {hw.exportPort = @portOutSym}
  // CHECK-SAME:  )
  firrtl.module private @InnerNames(
    in %value: !firrtl.uint<42> sym @portValueSym,
    in %clock: !firrtl.clock sym @portClockSym,
    in %reset: !firrtl.uint<1> sym @portResetSym,
    out %out: !firrtl.uint<1> sym @portOutSym
  ) {
    firrtl.instance instName sym @instSym @BitCast1()
    // CHECK: hw.instance "instName" sym @instSym @BitCast1
    %nodeName = firrtl.node sym @nodeSym %value : !firrtl.uint<42>
    // CHECK: [[WIRE:%nodeName]] = sv.wire sym @nodeSym : !hw.inout<i42>
    // CHECK-NEXT: sv.assign [[WIRE]], %value
    %wireName = firrtl.wire sym @wireSym : !firrtl.uint<42>
    // CHECK: %wireName = sv.wire sym @wireSym : !hw.inout<i42>
    %regName = firrtl.reg sym @regSym %clock : !firrtl.uint<42>
    // CHECK: %regName = sv.reg sym @regSym : !hw.inout<i42>
    %regResetName = firrtl.regreset sym @regResetSym %clock, %reset, %value : !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    firrtl.connect %out, %reset : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @regInitRandomReuse
  firrtl.module private @regInitRandomReuse(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, out %o1: !firrtl.uint<2>, out %o2: !firrtl.uint<4>, out %o3: !firrtl.uint<32>, out %o4: !firrtl.uint<100>) {
    %r1 = firrtl.reg %clock  : !firrtl.uint<2>
    %r2 = firrtl.reg %clock  : !firrtl.uint<4>
    %r3 = firrtl.reg %clock  : !firrtl.uint<32>
    %r4 = firrtl.reg %clock  : !firrtl.uint<100>
    // CHECK:      %r1 = sv.reg sym @[[r1_sym:[_A-Za-z0-9]+]]
    // CHECK:      %r2 = sv.reg sym @[[r2_sym:[_A-Za-z0-9]+]]
    // CHECK:      %r3 = sv.reg sym @[[r3_sym:[_A-Za-z0-9]+]]
    // CHECK:      %r4 = sv.reg sym @[[r4_sym:[_A-Za-z0-9]+]]
    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:     %[[RANDOM_0:.+]] = sv.reg sym @[[RANDOM_0_SYM:[_A-Za-z0-9]+]]
    // CHECK-NEXT:     %[[RANDOM_1:.+]] = sv.reg sym @[[RANDOM_1_SYM:[_A-Za-z0-9]+]]
    // CHECK-NEXT:     %[[RANDOM_2:.+]] = sv.reg sym @[[RANDOM_2_SYM:[_A-Za-z0-9]+]]
    // CHECK-NEXT:     %[[RANDOM_3:.+]] = sv.reg sym @[[RANDOM_3_SYM:[_A-Za-z0-9]+]]
    // CHECK-NEXT:     %[[RANDOM_4:.+]] = sv.reg sym @[[RANDOM_4_SYM:[_A-Za-z0-9]+]]{{.+}}
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[1:0];" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r1_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{][{]1[}][}]}}[5:2];" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r2_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{]}}{{[{][{]1[}][}]}}[5:0], {{[{][{]2[}][}]}}[31:6]{{[}]}};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r3_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_0_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_2_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_3_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_4_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {{[{]}}{{[{][{]1[}][}]}}[9:0], {{[{][{]2[}][}]}}, {{[{][{]3[}][}]}}, {{[{][{]4[}][}]}}[31:6]{{[}]}};" {symbols = [#hw.innerNameRef<@regInitRandomReuse::@[[r4_sym]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_4_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_3_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_2_SYM]]>, #hw.innerNameRef<@regInitRandomReuse::@[[RANDOM_1_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.connect %r1, %a : !firrtl.uint<2>, !firrtl.uint<1>
    firrtl.connect %r2, %a : !firrtl.uint<4>, !firrtl.uint<1>
    firrtl.connect %r3, %a : !firrtl.uint<32>, !firrtl.uint<1>
    firrtl.connect %r4, %a : !firrtl.uint<100>, !firrtl.uint<1>
    firrtl.connect %o1, %r1 : !firrtl.uint<2>, !firrtl.uint<2>
    firrtl.connect %o2, %r2 : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %o3, %r3 : !firrtl.uint<32>, !firrtl.uint<32>
    firrtl.connect %o4, %r4 : !firrtl.uint<100>, !firrtl.uint<100>
  }

  // CHECK-LABEL: hw.module private @init1DVector
  firrtl.module private @init1DVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    %r = firrtl.reg %clock  : !firrtl.vector<uint<1>, 2>
    // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]
    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:     %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]]{{.+}}
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}}[0] = {{[{][{]1[}][}]}}[0];" {symbols = [#hw.innerNameRef<@init1DVector::@[[r_sym]]>, #hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}}[1] = {{[{][{]1[}][}]}}[1];" {symbols = [#hw.innerNameRef<@init1DVector::@[[r_sym]]>, #hw.innerNameRef<@init1DVector::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }

    firrtl.connect %r, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    firrtl.connect %b, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    // CHECK:      sv.always posedge %clock  {
    // CHECK-NEXT:   sv.passign %r, %a : !hw.array<2xi1>
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output %0 : !hw.array<2xi1>
  }

  // CHECK-LABEL: hw.module private @init2DVector
  firrtl.module private @init2DVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, out %b: !firrtl.vector<vector<uint<1>, 1>, 1>) {
    %r = firrtl.reg %clock  : !firrtl.vector<vector<uint<1>, 1>, 1>
    // CKECK-NEXT: sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CKECK-NEXT:   %1 = sv.array_index_inout %r[%false] : !hw.inout<array<1xarray<1xi1>>>, i1
    // CKECK-NEXT:   %2 = sv.array_index_inout %1[%false] : !hw.inout<array<1xi1>>, i1
    // CKECK-NEXT:   %RANDOM = sv.verbatim.expr.se "`RANDOM" : () -> i32 {symbols = []}
    // CKECK-NEXT:   %3 = comb.extract %RANDOM from 0 : (i32) -> i1
    // CKECK-NEXT:   sv.bpassign %2, %3 : i1
    // CKECK-NEXT: }

    firrtl.connect %r, %a : !firrtl.vector<vector<uint<1>, 1>, 1>, !firrtl.vector<vector<uint<1>, 1>, 1>
    firrtl.connect %b, %r : !firrtl.vector<vector<uint<1>, 1>, 1>, !firrtl.vector<vector<uint<1>, 1>, 1>

    // CHECK:      sv.always posedge %clock  {
    // CHECK-NEXT:   sv.passign %r, %a : !hw.array<1xarray<1xi1>>
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output %0 : !hw.array<1xarray<1xi1>>
  }

  // CHECK-LABEL: hw.module private @connectNarrowUIntVector
  firrtl.module private @connectNarrowUIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 1>, out %b: !firrtl.vector<uint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<uint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<uint<2>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<uint<3>, 1>, !firrtl.vector<uint<2>, 1>
    // CHECK:      %2 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %3 = comb.concat %false, %2 : i1, i1
    // CHECK-NEXT: %4 = hw.array_create %3 : i2
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.passign %r1, %4 : !hw.array<1xi2>
    // CHECK-NEXT: }
    // CHECK-NEXT: %5 = hw.array_get %1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %6 = comb.concat %false, %5 : i1, i2
    // CHECK-NEXT: %7 = hw.array_create %6 : i3
    // CHECK-NEXT: sv.assign %.b.output, %7 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @connectNarrowSIntVector
  firrtl.module private @connectNarrowSIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<sint<1>, 1>, out %b: !firrtl.vector<sint<3>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.vector<sint<2>, 1>
    firrtl.connect %r1, %a : !firrtl.vector<sint<2>, 1>, !firrtl.vector<sint<1>, 1>
    firrtl.connect %b, %r1 : !firrtl.vector<sint<3>, 1>, !firrtl.vector<sint<2>, 1>
    // CHECK:      %2 = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: %3 = comb.concat %2, %2 : i1, i1
    // CHECK-NEXT: %4 = hw.array_create %3 : i2
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.passign %r1, %4 : !hw.array<1xi2>
    // CHECK-NEXT: }
    // CHECK-NEXT: %5 = hw.array_get %1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: %6 = comb.extract %5 from 1 : (i2) -> i1
    // CHECK-NEXT: %7 = comb.concat %6, %5 : i1, i2
    // CHECK-NEXT: %8 = hw.array_create %7 : i3
    // CHECK-NEXT: sv.assign %.b.output, %8 : !hw.array<1xi3>
    // CHECK-NEXT: hw.output %0 : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @SubIndex
  firrtl.module private @SubIndex(in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, in %clock: !firrtl.clock, out %o1: !firrtl.uint<1>, out %o2: !firrtl.vector<uint<1>, 1>) {
    %r1 = firrtl.reg %clock  : !firrtl.uint<1>
    %r2 = firrtl.reg %clock  : !firrtl.vector<uint<1>, 1>
    %0 = firrtl.subindex %a[0] : !firrtl.vector<vector<uint<1>, 1>, 1>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %r1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %r2, %0 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    firrtl.connect %o1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %o2, %r2 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    // CHECK:      %2 = hw.array_get %a[%false] : !hw.array<1xarray<1xi1>>
    // CHECK-NEXT: %3 = hw.array_get %2[%false] : !hw.array<1xi1>
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.passign %r1, %3 : i1
    // CHECK-NEXT:   sv.passign %r2, %2 : !hw.array<1xi1>
    // CHECK-NEXT: }
    // CHECK-NEXT: hw.output %0, %1 : i1, !hw.array<1xi1>
  }

  // CHECK-LABEL: hw.module private @SubindexDestination
  firrtl.module private @SubindexDestination(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 3>, out %b: !firrtl.vector<uint<1>, 3>) {
    %0 = firrtl.subindex %b[2] : !firrtl.vector<uint<1>, 3>
    %1 = firrtl.subindex %a[2] : !firrtl.vector<uint<1>, 3>
    firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %c-2_i2 = hw.constant -2 : i2
    // CHECK-NEXT: %.b.output = sv.wire  : !hw.inout<array<3xi1>>
    // CHECK-NEXT: %0 = sv.read_inout %.b.output : !hw.inout<array<3xi1>>
    // CHECK-NEXT: %1 = sv.array_index_inout %.b.output[%c-2_i2] : !hw.inout<array<3xi1>>, i2
    // CHECK-NEXT: %2 = hw.array_get %a[%c-2_i2] : !hw.array<3xi1>
    // CHECK-NEXT: sv.assign %1, %2 : i1
    // CHECK-NEXT: hw.output %0 : !hw.array<3xi1>
  }

  // CHECK-LABEL: hw.module private @zero_width_constant()
  // https://github.com/llvm/circt/issues/2269
  firrtl.module private @zero_width_constant(out %a: !firrtl.uint<0>) {
    // CHECK-NEXT: hw.output
    %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    firrtl.connect %a, %c0_ui0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: @subfield_write1(
  firrtl.module private @subfield_write1(out %a: !firrtl.bundle<a: uint<1>>) {
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %true = hw.constant true
    // CHECK-NEXT: %.a.output = sv.wire  : !hw.inout<struct<a: i1>>
    // CHECK-NEXT: %0 = sv.read_inout %.a.output : !hw.inout<struct<a: i1>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.a.output["a"] : !hw.inout<struct<a: i1>>
    // CHECK-NEXT: sv.assign %1, %true : i1
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: i1>
  }

  // CHECK-LABEL: @subfield_write2(
  firrtl.module private @subfield_write2(in %in: !firrtl.uint<1>, out %sink: !firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>) {
    %0 = firrtl.subfield %sink(0) : (!firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>) -> !firrtl.bundle<b: bundle<c: uint<1>>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<b: bundle<c: uint<1>>>) -> !firrtl.bundle<c: uint<1>>
    %2 = firrtl.subfield %1(0) : (!firrtl.bundle<c: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %2, %in : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %.sink.output = sv.wire  : !hw.inout<struct<a: !hw.struct<b: !hw.struct<c: i1>>>>
    // CHECK-NEXT: %0 = sv.read_inout %.sink.output : !hw.inout<struct<a: !hw.struct<b: !hw.struct<c: i1>>>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.sink.output["a"] : !hw.inout<struct<a: !hw.struct<b: !hw.struct<c: i1>>>>
    // CHECK-NEXT: %2 = sv.struct_field_inout %1["b"] : !hw.inout<struct<b: !hw.struct<c: i1>>>
    // CHECK-NEXT: %3 = sv.struct_field_inout %2["c"] : !hw.inout<struct<c: i1>>
    // CHECK-NEXT: sv.assign %3, %in : i1
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: !hw.struct<b: !hw.struct<c: i1>>>
  }

  // CHECK-LABEL: hw.module private @initStruct
  firrtl.module private @initStruct(in %clock: !firrtl.clock) {
    %r = firrtl.reg %clock  : !firrtl.bundle<a: uint<1>>
    // CHECK:      %r = sv.reg sym @[[r_sym:[_A-Za-z0-9]+]]
    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "RANDOMIZE_REG_INIT" {
    // CHECK-NEXT:     %[[RANDOM:.+]] = sv.reg sym @[[RANDOM_SYM:[_A-Za-z0-9]+]]{{.+}}
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.initial {
    // CHECK-NEXT:     sv.verbatim "`INIT_RANDOM_PROLOG_"
    // CHECK-NEXT:     sv.ifdef.procedural "RANDOMIZE_REG_INIT"  {
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}} = {`RANDOM};" {symbols = [#hw.innerNameRef<@initStruct::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:       sv.verbatim "{{[{][{]0[}][}]}}.a = {{[{][{]1[}][}]}}[0];" {symbols = [#hw.innerNameRef<@initStruct::@[[r_sym]]>, #hw.innerNameRef<@initStruct::@[[RANDOM_SYM]]>]}
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: hw.module private @RegResetStructWiden
  firrtl.module private @RegResetStructWiden(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %init: !firrtl.bundle<a: uint<2>>) {
    // CHECK:      [[FALSE:%.*]] = hw.constant false
    // CHECK-NEXT: [[A:%.*]] = hw.struct_extract %init["a"] : !hw.struct<a: i2>
    // CHECK-NEXT: [[PADDED:%.*]] = comb.concat [[FALSE]], [[A]] : i1, i2
    // CHECK-NEXT: [[STRUCT:%.*]] = hw.struct_create ([[PADDED]]) : !hw.struct<a: i3>
    // CHECK-NEXT: %reg = sv.reg {{.+}}  : !hw.inout<struct<a: i3>>
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:     sv.passign %reg, [[STRUCT]] : !hw.struct<a: i3>
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<3>>
  }

  // CHECK-LABEL: hw.module private @BundleConnection
  firrtl.module private @BundleConnection(in %source: !firrtl.bundle<a: bundle<b: uint<1>>>, out %sink: !firrtl.bundle<a: bundle<b: uint<1>>>) {
    %0 = firrtl.subfield %sink(0) : (!firrtl.bundle<a: bundle<b: uint<1>>>) -> !firrtl.bundle<b: uint<1>>
    %1 = firrtl.subfield %source(0) : (!firrtl.bundle<a: bundle<b: uint<1>>>) -> !firrtl.bundle<b: uint<1>>
    firrtl.connect %0, %1 : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
    // CHECK:      %.sink.output = sv.wire  : !hw.inout<struct<a: !hw.struct<b: i1>>>
    // CHECK-NEXT: %0 = sv.read_inout %.sink.output : !hw.inout<struct<a: !hw.struct<b: i1>>>
    // CHECK-NEXT: %1 = sv.struct_field_inout %.sink.output["a"] : !hw.inout<struct<a: !hw.struct<b: i1>>>
    // CHECK-NEXT: %2 = hw.struct_extract %source["a"] : !hw.struct<a: !hw.struct<b: i1>>
    // CHECK-NEXT: sv.assign %1, %2 : !hw.struct<b: i1>
    // CHECK-NEXT: hw.output %0 : !hw.struct<a: !hw.struct<b: i1>>
  }

  // CHECK-LABEL: hw.module private @AggregateInvalidValue
  firrtl.module private @AggregateInvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    %reg = firrtl.regreset %clock, %reset, %invalid : !firrtl.uint<1>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    // CHECK:      %c0_i101 = hw.constant 0 : i101
    // CHECK-NEXT: %0 = hw.bitcast %c0_i101 : (i101) -> !hw.struct<a: i1, b: !hw.array<10xi10>>
    // CHECK-NEXT: %reg = sv.reg {{.+}} : !hw.inout<struct<a: i1, b: !hw.array<10xi10>>>
    // CHECK-NEXT: sv.always posedge %clock  {
    // CHECK-NEXT:   sv.if %reset  {
    // CHECK-NEXT:     sv.passign %reg, %0 : !hw.struct<a: i1, b: !hw.array<10xi10>>
    // CHECK-NEXT:   } else  {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: hw.module private @AggregateRegAssign
  firrtl.module private @AggregateRegAssign(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.vector<uint<1>, 1>
    %reg_0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %reg_0, %value : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %0 = sv.array_index_inout %reg[%false] : !hw.inout<array<1xi1>>, i1
    // CHECK:  sv.passign %0, %value : i1
  }

  // CHECK-LABEL: hw.module private @AggregateRegResetAssign
  firrtl.module private @AggregateRegResetAssign(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                                         in %init: !firrtl.vector<uint<1>, 1>, in %value: !firrtl.uint<1>) {
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    %reg_0 = firrtl.subindex %reg[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %reg_0, %value : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %0 = sv.array_index_inout %reg[%false] : !hw.inout<array<1xi1>>, i1
    // CHECK:  sv.passign %0, %value : i1
  }

  // CHECK-LABEL: hw.module private @ForceNameSubmodule
  firrtl.nla @nla_1 [#hw.innerNameRef<@ForceNameTop::@sym_foo>,@ForceNameSubmodule]
  firrtl.nla @nla_2 [#hw.innerNameRef<@ForceNameTop::@sym_bar>,@ForceNameSubmodule]
  firrtl.module private @ForceNameSubmodule() attributes {annotations = [
    {circt.nonlocal = @nla_2,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Bar"},
    {circt.nonlocal = @nla_1,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Foo"}]} {}
  // CHECK: hw.module private @ForceNameTop
  firrtl.module private @ForceNameTop() {
    firrtl.instance foo sym @sym_foo
      {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    firrtl.instance bar sym @sym_bar
      {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    // CHECK:      hw.instance "foo" sym @sym_foo {{.+}} {hw.verilogName = "Foo"}
    // CHECK-NEXT: hw.instance "bar" sym @sym_bar {{.+}} {hw.verilogName = "Bar"}
  }

  // CHECK-LABEL: hw.module private @PreserveName
  firrtl.module private @PreserveName(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>, out %c : !firrtl.uint<1>) {
    //CHECK comb.or %a, %b {sv.namehint = "myname"}
    %foo = firrtl.or %a, %b {name = "myname"} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %c, %foo : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @MutlibitMux(%source_0: i1, %source_1: i1, %source_2: i1, %index: i2) -> (sink: i1) {
  firrtl.module private @MutlibitMux(in %source_0: !firrtl.uint<1>, in %source_1: !firrtl.uint<1>, in %source_2: !firrtl.uint<1>, out %sink: !firrtl.uint<1>, in %index: !firrtl.uint<2>) {
    %0 = firrtl.multibit_mux %index, %source_2, %source_1, %source_0 : !firrtl.uint<2>, !firrtl.uint<1>
    firrtl.connect %sink, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %0 = hw.array_create %source_2, %source_1, %source_0 : i1
    // CHECK-NEXT: %1 = hw.array_get %0[%index] : !hw.array<3xi1>
    // CHECK-NEXT: %2 = hw.array_get %0[%c0_i2]
    // CHECK-NEXT: %3 = comb.icmp uge %index, %c-1_i2
    // CHECK-NEXT: %4 = comb.mux %3, %2, %1
    // CHECK-NEXT: hw.output %4 : i1
  }

  // CHECK-LABEL: hw.module private @eliminateSingleOutputConnects
  // CHECK-NOT:     [[WIRE:%.+]] = sv.wire
  // CHECK-NEXT:    hw.output %a : i1
  firrtl.module private @eliminateSingleOutputConnects(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.strictconnect %b, %a : !firrtl.uint<1>
  }
}
