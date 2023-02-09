// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-aos-to-soa))' %s | FileCheck %s

firrtl.circuit "Test" {
  firrtl.module @Test() {}
  //===--------------------------------------------------------------------===//
  // Port Type-Change Tests
  //===--------------------------------------------------------------------===//

  // CHECK:     @VG(in %port: !firrtl.vector<uint<8>, 4>)
  firrtl.module @VG(in %port: !firrtl.vector<uint<8>, 4>) {}

  // CHECK:      BG(in %port: !firrtl.bundle<a: uint<8>>)
  firrtl.module @BG(in %port: !firrtl.bundle<a: uint<8>>) {}

  // CHECK:     @VB(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  firrtl.module @VB(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VB2(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  firrtl.module @VB2(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VBB(in %port: !firrtl.bundle<nested: bundle<field: vector<uint<1>, 4>>>)
  firrtl.module @VBB(in %port: !firrtl.vector<bundle<nested: bundle<field: uint<1>>>, 4>) {}

  // CHECK:     @VVB(in %port: !firrtl.bundle<field: vector<vector<uint<1>, 6>, 4>>)
  firrtl.module @VVB(in %port: !firrtl.vector<vector<bundle<field: uint<1>>, 6>, 4>) {}

  // CHECK:     @VBVB(in %port: !firrtl.bundle<field_a: bundle<field_b: vector<vector<uint<1>, 4>, 8>>>)    
  firrtl.module @VBVB(in %port: !firrtl.vector<bundle<field_a: vector<bundle<field_b: uint<1>>, 4>>, 8>) {}

  //===--------------------------------------------------------------------===//
  // Aggregate Create/Constant Ops
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestAggregateConstants
  firrtl.module @TestAggregateConstants() {
    // CHECK{LITERAL}: firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    // CHECK{LITERAL}: firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[[1, 3], [2, 4]], [[5, 7], [6, 8]]] : !firrtl.bundle<a: bundle<c: vector<uint<8>, 2>, d: vector<uint<5>, 2>>, b: bundle<e: vector<uint<8>, 2>, f: vector<uint<5>, 2>>>
    firrtl.aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] : !firrtl.bundle<a: vector<bundle<c: uint<8>, d: uint<5>>, 2>, b: vector<bundle<e: uint<8>, f: uint<5>>, 2>>
    // CHECK{LITERAL}: firrtl.aggregateconstant [[[1, 3], [5, 7], [9, 11]], [[2, 4], [6, 8], [10, 12]]] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 3>, b: vector<vector<uint<8>, 2>, 3>>
    firrtl.aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<8>>, 2>, 3>
  }

  // CHECK-LABEL: @TestBundleCreate
  firrtl.module @TestBundleCreate() {
    // CHECK: %0 = firrtl.bundlecreate  : () -> !firrtl.bundle<>
    %be = firrtl.bundlecreate : () -> !firrtl.bundle<>

    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    // CHECK: %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
    // %1 = firrtl.bundlecreate %c0_ui8, %c1_ui4 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>
    %c0 = firrtl.constant 0 : !firrtl.uint<8>
    %c1 = firrtl.constant 1 : !firrtl.uint<4>
    %bc = firrtl.bundlecreate %c0, %c1 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>

    // %2 = firrtl.aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    // %3 = firrtl.aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    // %4 = firrtl.bundlecreate %2, %3 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>
    %v0 = firrtl.aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    %v1 = firrtl.aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    %bv = firrtl.bundlecreate %v0, %v1 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>

    // %5 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %6 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %7 = firrtl.bundlecreate %5, %6 : (!firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>) -> !firrtl.bundle<a: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, b: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>>
    %vb0 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %vb1 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %bvb = firrtl.bundlecreate %vb0, %vb1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<5>>, 2>, b: vector<bundle<a: uint<8>, b: uint<5>>, 2>>
  }

  // CHECK-LABEL: @TestVectorCreate
  firrtl.module @TestVectorCreate() {
    %v0 = firrtl.vectorcreate : () -> !firrtl.vector<uint<8>, 0>

    %c0 = firrtl.constant 1 : !firrtl.uint<8>
    %v1 = firrtl.vectorcreate %c0, %c0: (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>

    %b0 = firrtl.bundlecreate %c0 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    %v2 = firrtl.vectorcreate %b0 : (!firrtl.bundle<a: uint<8>>) -> !firrtl.vector<bundle<a: uint<8>>, 1>

    %b1 = firrtl.bundlecreate %v1 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>>
    %v3 = firrtl.vectorcreate %b1 : (!firrtl.bundle<a: vector<uint<8>, 2>>) -> !firrtl.vector<bundle<a: vector<uint<8>, 2>>, 1>
  }

  //===--------------------------------------------------------------------===//
  // Declaration Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestWire
  firrtl.module @TestWire() {
    // CHECK: %0 = firrtl.wire : !firrtl.uint<8>
    %0 = firrtl.wire : !firrtl.uint<8>
    // CHECK: %1 = firrtl.wire : !firrtl.bundle<>
    %1 = firrtl.wire : !firrtl.bundle<>
    // CHECK: %2 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %2 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %3 = firrtl.wire : !firrtl.vector<uint<8>, 0>
    %3 = firrtl.wire : !firrtl.vector<uint<8>, 0>
    // CHECK: %4 = firrtl.wire : !firrtl.vector<uint<8>, 2>
    %4 = firrtl.wire : !firrtl.vector<uint<8>, 2>
    // CHECK: %5 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>
    %5 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK %6 = firrtl.wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    %6 = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %7 = firrtl.wire : !firrtl.bundle<a: bundle<b: vector<uint<8>, 2>>>
    %7 = firrtl.wire : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>>
  }

  // CHECK-LABEL: @TestNode
  firrtl.module @TestNode() {
    %w = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %n = firrtl.node %w : !firrtl.bundle<a: vector<uint<8>, 2>>
    %n = firrtl.node %w : !firrtl.vector<bundle<a: uint<8>>, 2>
  }
  
  // CHECK-LABEL: @TestReg
  firrtl.module @TestReg(in %clock: !firrtl.clock) {
    // CHECK: %r = firrtl.reg %clock : !firrtl.bundle<a: vector<uint<8>, 2>>
    %r = firrtl.reg %clock : !firrtl.vector<bundle<a: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestRegReset
  firrtl.module @TestRegReset(in %clock: !firrtl.clock) {
    // TODO
    // %reset1 = firrtl.constant_aggregate 
    // // CHECK: %r = firrtl.reg %clock : !firrtl.bundle<a: vector<uint<8>, 2>>
    // %reg1 = firrtl.reg %clock : !firrtl.vector<bundle<a: uint<8>>, 2
    
    // %reset2
  }

  // CHECK-LABEL: @TestInstance
  firrtl.module @TestInstance() {
  }

  //===--------------------------------------------------------------------===//
  // Connect Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestBasicConnects
  firrtl.module @TestBasicConnects() {
    %c1 = firrtl.constant 1 : !firrtl.uint<1>
    %w0 = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %w0, %c1 : !firrtl.uint<1>

    %w1 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    %w2 = firrtl.wire : !firrtl.bundle<a: bundle<>>
    firrtl.strictconnect %w1, %w2 : !firrtl.bundle<a: bundle<>>

    %w3 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    %w4 = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    firrtl.strictconnect %w3, %w4 : !firrtl.bundle<a flip: uint<1>>

    %w5 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %w6 = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    firrtl.strictconnect %w5, %w6 : !firrtl.bundle<a flip: uint<1>, b: uint<1>>

    %w7 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    %w8 = firrtl.wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    firrtl.strictconnect %w7, %w8 : !firrtl.bundle<a: bundle<b flip: uint<8>>>
  
    %w9  = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    %w10 = firrtl.wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    firrtl.strictconnect %w9, %w10 : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>

    %w9_a = firrtl.subfield %w9[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    %w10_a = firrtl.subfield %w10[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    firrtl.strictconnect %w9_a, %w10_a: !firrtl.bundle<b flip: uint<8>>
  
    %w9_a_b = firrtl.subfield %w9_a[b] : !firrtl.bundle<b flip: uint<8>>
    %w10_a_b = firrtl.subfield %w10_a[b] : !firrtl.bundle<b flip: uint<8>>
    firrtl.strictconnect %w9_a_b, %w10_a_b : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Path Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestSubindex
  firrtl.module @TestSubindex(in %port: !firrtl.vector<bundle<a flip: uint<8>>, 4>) {
    %value  = firrtl.constant 7 : !firrtl.uint<8>
    %bundle = firrtl.subindex %port[3] : !firrtl.vector<bundle<a flip: uint<8>>, 4>
    %field  = firrtl.subfield %bundle[a] : !firrtl.bundle<a flip: uint<8>>
    firrtl.connect %field, %value : !firrtl.uint<8>, !firrtl.uint<8>

    %v1 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %v2 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b1 = firrtl.subindex %v1[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b2 = firrtl.subindex %v2[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    firrtl.strictconnect %b1, %b2 : !firrtl.bundle<a: vector<uint<8>, 8>>
  }

  // CHECK-LABEL: @TestSubfield
  firrtl.module  @TestSubfield() {

  }

  firrtl.module @TestSubaccess() {

  }

  // CHECK-LABEL: @TestPathCaching()
  firrtl.module @TestPathCaching() {
    // %w = firrtl.wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    // %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    // %1 = firrtl.subfield %0[b] : !firrtl.bundle<b: uint<8>>
    // %n1 = firrtl.node %0 : !firrtl.bundle<b: uint<8>>
    // %n2 = firrtl.node %1 : !firrtl.uint<8>
    %w = firrtl.wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    %a = firrtl.subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %n1 = firrtl.node %a : !firrtl.bundle<b: uint<8>>
    %b = firrtl.subfield %a[b] : !firrtl.bundle<b: uint<8>>
    %n2 = firrtl.node %b : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Operand Explosion Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestLhsExploded
  firrtl.module @TestLhsExploded() {
    %lhs_storage  = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %lhs = firrtl.subindex %lhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %rhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestLhsExplodedWhenLhsHasFlips
  firrtl.module @TestLhsExplodedWhenLhsHasFlips() {
    %lhs_storage = firrtl.wire : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %lhs = firrtl.subindex %lhs_storage[0] : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %rhs = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestRhsExploded
  firrtl.module @TestRhsExploded() {
    %lhs = firrtl.wire : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %rhs_storage = firrtl.wire : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %rhs = firrtl.subindex %rhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestRhsExplodedWhenLhsHasFlips
  firrtl.module @TestRhsExplodedWhenLhsHasFlips() {
    %lhs = firrtl.wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %rhs_storage = firrtl.wire : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %rhs = firrtl.subindex %rhs_storage[0] : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    firrtl.strictconnect %lhs, %rhs : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestBothSidesExploded
  firrtl.module @TestBothSidesExploded() {
    %v1 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %v2 = firrtl.wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b2 = firrtl.subindex %v1[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b3 = firrtl.subindex %v2[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    firrtl.strictconnect %b2, %b3 : !firrtl.bundle<a: vector<uint<8>, 8>>
  }

  // firrtl.module @TestExplodedNode 

  //===--------------------------------------------------------------------===//
  // Structure Tests
  //===--------------------------------------------------------------------===//

  /// CHECK-LABEL: @TestWhen()
  firrtl.module @TestWhen() {
    %w = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<8>>
    %p = firrtl.constant 1 : !firrtl.uint<1>
    // %n1 = firrtl.node %a : !firrtl.uint<8>
    firrtl.when %p {
      %n2 = firrtl.node %a : !firrtl.uint<8>
    }
    %n3 = firrtl.node %a : !firrtl.uint<8>

    firrtl.when %p {
      %w2 = firrtl.wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    }
  }

  //===--------------------------------------------------------------------===//
  // Misc Tests
  //===--------------------------------------------------------------------===//


    // CHECK-LABEL: @TestDoubleSlicing()
  firrtl.module @TestDoubleSlicing() {
    // CHECK: %w = firrtl.wire : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    %w = firrtl.wire : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>

    // CHECK: %0 = firrtl.subfield %w[v] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %1 = firrtl.subindex %0[0] : !firrtl.vector<vector<vector<uint<7>, 3>, 2>, 4>
    // CHECK: %2 = firrtl.subindex %1[0] : !firrtl.vector<vector<uint<7>, 3>, 2>
    // CHECK: %3 = firrtl.subindex %2[2] : !firrtl.vector<uint<7>, 3>
    // CHECK: %4 = firrtl.subfield %w[a] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %5 = firrtl.subindex %4[0] : !firrtl.vector<vector<uint<8>, 2>, 4>
    // CHECK: %6 = firrtl.subindex %5[0] : !firrtl.vector<uint<8>, 2>

    // CHECK: %7 = firrtl.bundlecreate %5, %1 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<vector<uint<7>, 3>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    // CHECK: %n_0 = firrtl.node %7 : !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    %w_0 = firrtl.subindex %w[0] : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>
    %n_0 = firrtl.node %w_0 : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>

    // CHECK: %8 = firrtl.bundlecreate %6, %2 : (!firrtl.uint<8>, !firrtl.vector<uint<7>, 3>) -> !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    // CHECK: %n_0_1 = firrtl.node %8 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %w_0_1 = firrtl.subindex %w_0[0] : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>
    %n_0_1 = firrtl.node %w_0_1 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>

    // CHECK: %n_0_1_b = firrtl.node %2 : !firrtl.vector<uint<7>, 3>
    %w_0_1_b = firrtl.subfield %w_0_1[v] : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %n_0_1_b = firrtl.node %w_0_1_b : !firrtl.vector<uint<7>, 3>
  
    // CHECK: %n_0_1_b_2 = firrtl.node %3 : !firrtl.uint<7>
    %w_0_1_b_2 = firrtl.subindex %w_0_1_b[2] : !firrtl.vector<uint<7>, 3>
    %n_0_1_b_2 = firrtl.node %w_0_1_b_2 : !firrtl.uint<7>
  }

  // firrtl.module @TestVectorAggregate() {
  //   %0 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>>
  //   %1 = firrtl.subindex %0[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>>
  //   %2 = firrtl.subindex %0[1] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>>
  //   %3 = firrtl.vectorcreate %1, %2 : (bundle<a: uint<8>, b: uint<5>>, bundle<a: uint<8>, b: uint<5>>) !firrt.vector<bundle<a: uint<8>, b: uint<5>>, 2>
  // }

  // firrtl.module @TestVectorAggregate(in %a : !firrt.uint<8>, in %b : !firrtl.uint<8>) {
  //   %0 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
  //   %1 = firrtl.vectorcreate %0, %0 : (!firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<8>>, 2>
  //   %2 = firrtl.subindex %1[1] : !firrtl.vector<bundle<a: uint<8>, b: uint<8>>, 2>
  //   %n = firrtl.node %2 : !firrtl.bundle<a: uint<8>, b: uint<8>>
  // }

  firrtl.module @TestVBAggregate() {
    // CHECK: %0 = firrtl.aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %1 = firrtl.aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %0 = firrtl.aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %1 = firrtl.aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>


    %2 = firrtl.vectorcreate %0, %1 : (!firrtl.bundle<a: uint<8>, b: uint<5>>, !firrtl.bundle<a: uint<8>, b: uint<5>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
  }

  firrtl.module @TestVBBAggregate() {
    // %0 = firrtl.aggregateconstant [[1]] : !firrtl.bundle<a: bundle<b: uint<8>>>
    // %1 = firrtl.aggregateconstant [[2]] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %0 = firrtl.aggregateconstant [[1]] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %1 = firrtl.aggregateconstant [[2]] : !firrtl.bundle<a: bundle<b: uint<8>>>

    // %0_a = subfield %0[a]
    // %0_a_b = subfield %0_a[b]
    // %bs = vectorcreate %0_a_b, %1_a_b
    // %a = bundle_create %bs
    // bundle_create %a : 
    %2 = firrtl.vectorcreate %0, %1 :
      (!firrtl.bundle<a: bundle<b: uint<8>>>, !firrtl.bundle<a: bundle<b: uint<8>>>) ->
        !firrtl.vector<bundle<a: bundle<b: uint<8>>>, 2>
  }

  firrtl.module @TestVVBAggregate() {
    // CHECK{LITERAL}: %0 = firrtl.aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK{LITERAL}: %1 = firrtl.aggregateconstant [[5, 7], [6, 8]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %0 = firrtl.aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %1 = firrtl.aggregateconstant [[5, 6], [7, 8]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>

    // 0_a =  
    // 0_b = bundle
    // 1_a = 
    // 1_b = subfield %1[b]
    // %vec_of_a = vectorcreate ...... : vector<vector<uint>>
    // %vec_of_b = vectorcreate ....... : vector<vector<uint>>
    // bundlecreate %vec_of_a, vec_of_b
    %2 = firrtl.vectorcreate %0, %1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.vector<vector<bundle<a: uint<8>, b: uint<5>>, 2>, 2>
  }

  // connect rhs is a rematerialized bundle with flip, Do we preserve the flip?
  firrtl.module @VBF(
    in  %i : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>,
    out %o : !firrtl.bundle<a: uint<8>, b flip: uint<8>>) {
    // CHECK: %0 = firrtl.subfield %i[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %1 = firrtl.subfield %i[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %2 = firrtl.subindex %0[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %3 = firrtl.subindex %1[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %4 = firrtl.bundlecreate %2, %3 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: firrtl.strictconnect %o, %4 : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    %0 = firrtl.subindex %i[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    firrtl.strictconnect %o, %0 : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // connect lhs is an exploded bundle with flip, Do we connect in the right direction?
  firrtl.module @VBF2(
    in  %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>,
    out %o : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>) {
    // CHECK: %0 = firrtl.subfield %i[a] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %1 = firrtl.subfield %i[b] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %2 = firrtl.subfield %o[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %3 = firrtl.subfield %o[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %4 = firrtl.subindex %2[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %5 = firrtl.subindex %3[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: firrtl.strictconnect %4, %0 : !firrtl.uint<8>
    // CHECK: firrtl.strictconnect %1, %5 : !firrtl.uint<8>
    %0 = firrtl.subindex %o[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    firrtl.strictconnect %0, %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // CHECK-LABEL: TestBundleCreate_VB
  firrtl.module @TestBundleCreate_VB(out %out : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
    // CHECK: %0 = firrtl.constant 1 : !firrtl.uint<8>
    %0 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %0 = firrtl.constant 1 : !firrtl.uint<8>
    %1 = firrtl.constant 2 : !firrtl.uint<16>
    // CHECK: %1 = firrtl.vectorcreate
    // CHECK: %2 = firrtl.vectorcreate
    // CHECK: %3 = firrtl.bundlecreate
    %bundle1 = firrtl.bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    firrtl.connect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }

  //===--------------------------------------------------------------------===//
  // Ref Types
  //===--------------------------------------------------------------------===//

  // firrtl.module @RefSender(out %ref_a_port: !firrtl.ref<uint<1>>) {
  //   %a = firrtl.wire : !firrtl.uint<1>
  //   %ref_a = firrtl.ref.send %a : !firrtl.uint<1>
  //   firrtl.strictconnect %ref_a_port, %ref_a : !firrtl.ref<uint<1>>
  // }

  // firrtl.module @RefResolver() {
  //   %bar_ref_a_port = firrtl.instance bar @Bar(out ref_a_port: !firrtl.ref<uint<1>>)
  //   %b = firrtl.wire: !firrtl.uint<1>
  //   %0 = firrtl.ref.resolve %bar_ref_a_port : !firrtl.ref<uint<1>>
  //   firrtl.strictconnect %b, %0 : !firrtl.uint<1>
  // }
}
