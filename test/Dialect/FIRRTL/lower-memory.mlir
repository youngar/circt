// RUN: circt-opt -firrtl-lower-memory %s | FileCheck %s

firrtl.circuit "Simple" {
  %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
}

firrtl.circuit ""
  firrtl.module private @inferUnmaskedMemory(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.uint<8>, in %wMask: !firrtl.uint<1>, in %wData: !firrtl.uint<8>) {
    %tbMemoryKind1_r, %tbMemoryKind1_w = firrtl.mem Undefined  {depth = 16 : i64, modName = "tbMemoryKind1_ext", name = "tbMemoryKind1", portNames = ["r", "w"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = firrtl.subfield %tbMemoryKind1_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<8>
    %1 = firrtl.subfield %tbMemoryKind1_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %tbMemoryKind1_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<4>
    %3 = firrtl.subfield %tbMemoryKind1_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %tbMemoryKind1_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>) -> !firrtl.clock
    %5 = firrtl.subfield %tbMemoryKind1_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<8>
    %6 = firrtl.subfield %tbMemoryKind1_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<4>
    %7 = firrtl.subfield %tbMemoryKind1_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.uint<1>
    %8 = firrtl.subfield %tbMemoryKind1_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>) -> !firrtl.clock
    firrtl.connect %8, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %7, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %rData, %5 : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %3, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    firrtl.connect %1, %wMask : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %wData : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: hw.module private @inferUnmaskedMemory
  // CHECK-NEXT:   %[[v0:.+]] = comb.and %rEn, %wMask : i1
  // CHECK-NEXT:   %tbMemoryKind1_ext.R0_data = hw.instance "tbMemoryKind1_ext" @tbMemoryKind1_ext(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, W0_addr: %rAddr: i4, W0_en: %[[v0]]: i1, W0_clk: %clock: i1, W0_data: %wData: i8) -> (R0_data: i8)
  // CHECK-NEXT:   hw.output %tbMemoryKind1_ext.R0_data : i8
