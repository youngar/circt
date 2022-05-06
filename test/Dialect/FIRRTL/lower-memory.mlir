// RUN: circt-opt -firrtl-lower-memory %s | FileCheck %s

// Test basic lowering of the three port types.
// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
firrtl.module @Simple() {
  %MWrite_write = firrtl.mem Undefined {depth = 12 : i64, name = "MWrite", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  %MReadWrite_readwrite = firrtl.mem Undefined {depth = 12 : i64, name = "MReadWrite", portNames = ["readwrite"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
  // CHECK: firrtl.instance MReadWrite_ext  @MReadWrite_ext(in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<42>, out RW0_rdata: !firrtl.uint<42>)
  // SeqMems need at least 1 write port, but this is primarily testing the Read port.
  %MRead_read, %MRead_write = firrtl.mem Undefined {depth = 12 : i64, name = "MRead", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance MRead_ext  @MRead_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
}

// Check that the attached metadata is correct:

// CHECK: firrtl.memmodule @MWrite_ext
// CHECK-SAME: {dataType = !firrtl.uint<42>, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}

// CHECK: firrtl.memmodule @MReadWrite_ext
// CHECK-SAME: {dataType = !firrtl.uint<42>, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}

// CHECK: firrtl.memmodule @MRead_ext
// CHECK-SAME: {dataType = !firrtl.uint<42>, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
}

// Test that the memory module is renamed when there is a collision.
// CHECK-LABEL: firrtl.circuit "Collision"
firrtl.circuit "Collision" {
firrtl.extmodule @test_ext()
firrtl.module @Collision() {
  %MWrite_write = firrtl.mem Undefined {depth = 12 : i64, name = "test", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance test_0_ext
}
// CHECK: firrtl.memmodule @test_0_ext
}

// Test that the memory modules are deduplicated.
// CHECK-LABEL: firrtl.circuit "Dedup"
firrtl.circuit "Dedup" {
firrtl.module @Dedup() {
  %mem0_write = firrtl.mem Undefined {depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  %mem1_write = firrtl.mem Undefined {depth = 12 : i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance mem0_ext  @mem0_ext(
  // CHECK: firrtl.instance mem0_ext  @mem0_ext(
}
// CHECK: firrtl.memmodule @mem0_ext
// CHECK-NOT: firrtl.memmodule @mem1_ext
}

// Test that memories in the testharness are not deduped with other memories.
// CHECK-LABEL: firrtl.circuit "NoTestharnessDedup"
firrtl.circuit "NoTestharnessDedup" {
firrtl.module @NoTestharnessDedup() {
  %mem0_write = firrtl.mem Undefined {depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance mem0_ext  @mem0_ext
  firrtl.instance dut @DUT()
}
firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  %mem1_write = firrtl.mem Undefined {depth = 12 : i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: firrtl.instance mem1_ext  @mem1_ext
}
// CHECK: firrtl.memmodule @mem0_ext
// CHECK: firrtl.memmodule @mem1_ext
}

// Check that when the mask is 1-bit, it is removed from the memory and the
// enable signal is and'd with the mask signal.
// CHECK-LABEL: firrtl.circuit "NoMask"
firrtl.circuit "NoMask" {
  firrtl.module @NoMask(in %en: !firrtl.uint<1>, in %mask: !firrtl.uint<1>) {
    %MemSimple_read, %MemSimple_write = firrtl.mem Undefined {depth = 12 : i64, name = "MemSimple", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // Enable:
    %0 = firrtl.subfield %MemSimple_write(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %0, %en : !firrtl.uint<1>, !firrtl.uint<1>

    // Mask:
    %1 = firrtl.subfield %MemSimple_write(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %mask : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: [[AND:%.+]] = firrtl.and %mask, %en
    // CHECK: firrtl.connect %MemSimple_ext_W0_en, [[AND]]
  }
}

// Check a memory with a mask gets lowered properly.
// CHECK-LABEL: firrtl.circuit "YesMask"
firrtl.circuit "YesMask" {
  firrtl.module @YesMask(in %en: !firrtl.uint<1>, in %mask: !firrtl.uint<4>) {
    %MemSimple_read, %MemSimple_write = firrtl.mem Undefined {depth = 1022 : i64, name = "MemSimple", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: uint<40>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>

    // Enable:
    %0 = firrtl.subfield %MemSimple_write(1) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>) -> !firrtl.uint<1>
    firrtl.connect %0, %en : !firrtl.uint<1>, !firrtl.uint<1>

    // Mask:
    %1 = firrtl.subfield %MemSimple_write(4) : (!firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>) -> !firrtl.uint<4>
    firrtl.connect %1, %mask : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: firrtl.connect %MemSimple_ext_W0_en, %en
    // CHECK: firrtl.connect %MemSimple_ext_W0_mask, %mask
  }
}

// CHECK-LABEL: firrtl.circuit "MemDepth1"
firrtl.circuit "MemDepth1" {
  firrtl.module @MemDepth1(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %addr: !firrtl.uint<1>, in %data: !firrtl.uint<32>) {
    // CHECK: firrtl.instance mem0_ext  @mem0_ext(in W0_addr: !firrtl.uint<1>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<32>, in W0_mask: !firrtl.uint<4>)
    // CHECK: firrtl.connect %mem0_ext_W0_data, %data : !firrtl.uint<32>, !firrtl.uint<32>
    %mem0_write = firrtl.mem Old {depth = 1 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    %1 = firrtl.subfield %mem0_write(0) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>) -> !firrtl.uint<1>
    firrtl.connect %1, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = firrtl.subfield %mem0_write(1) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>) -> !firrtl.uint<1>
    firrtl.connect %3, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.subfield %mem0_write(2) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %2 = firrtl.subfield %mem0_write(3) : (!firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>) -> !firrtl.uint<32>
    firrtl.connect %2, %data : !firrtl.uint<32>, !firrtl.uint<32>
}
// CHECK: firrtl.memmodule @mem0_ext
// CHECK-SAME: depth = 1
}

// CHECK-LABEL: firrtl.circuit "inferUnmaskedMemory"
firrtl.circuit "inferUnmaskedMemory" {
  firrtl.module @inferUnmaskedMemory(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.uint<8>, in %wMask: !firrtl.uint<1>, in %wData: !firrtl.uint<8>) {
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
    // CHECK: [[AND:%.+]] = firrtl.and %wMask, %rEn
    // CHECK: firrtl.connect %tbMemoryKind1_ext_W0_en, %0
  }
}