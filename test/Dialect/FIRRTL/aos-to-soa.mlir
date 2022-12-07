// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-aos-to-soa)))' %s | FileCheck %s

// test cases
// - base type
// - struct
// - vector of base type
// - vector of struct
// - vector of vector of base type
// - vector of vector of struct
// - struct of vector of struct
// - 

// - wire
// - port
// - register?

// have to use the value, too
// 
firrtl.circuit "Test" {
  firrtl.module @Test() {}

  // vector<ground> -> vector<ground> (no change)
  // CHECK-LABEL:  @VG
  // CHECK-SAME:   (in %port: !firrtl.vector<uint<8>, 4>)
  firrtl.module @VG(in %port: !firrtl.vector<uint<8>, 4>) {
    // no change
  }

  // bundle<ground> -> bundle<ground> (no change)
  // CHECK-LABEL:  @BG
  // CHECK-SAME:   (in %port: !firrtl.bundle<a: uint<8>>)
  firrtl.module @BG(in %port: !firrtl.bundle<a: uint<8>>) {
  // no change
  }

  // vector<bundle> -> bundle<vector>
  // CHECK-LABEL:  @VB
  // CHECK-SAME:   (in %port: !firrtl.bundle<field: vector<uint<1>, 4>>)
  firrtl.module @VB(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {
    // port[i].field => port.field[i]
    // %bundle = firrtl.subindex %port[0] : !firrtl.vector<bundle<a: uint<8>>, 4> -> bundle<a: uint<8>>
    // %field  = firrlt.subfield %bundle(0) : !firrtl.bundle<a: uint<8>> -> uint<8>
    // %self_port = firrtl.instance self @VB(in port: !firrtl.vector<bundle<a: uint<8>>, 4>)
    // firrtl.connect %self_port, %port : !firrtl.vector<bundle<a: uint<8>>, 4>, !firrtl.vector<bundle<a: uint<8>>, 4>
  }

  firrtl.module @VBX(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {
    // port[i].field => port.field[i]
    // %bundle = firrtl.subindex %port[0] : !firrtl.vector<bundle<a: uint<8>>, 4> -> bundle<a: uint<8>>
    // %field  = firrlt.subfield %bundle(0) : !firrtl.bundle<a: uint<8>> -> uint<8>
    %self_port = firrtl.instance self @VB(in port: !firrtl.vector<bundle<a: uint<8>>, 4>)
    firrtl.connect %self_port, %port : !firrtl.vector<bundle<a: uint<8>>, 4>, !firrtl.vector<bundle<a: uint<8>>, 4>

    // %value   = firrtl.constant 7 : !firrtl.uint<8>
    // %bundle  = firrtl.subindex %port[3] : !firrtl.vector<bundle<a: uint<8>>, 4>
    // %field   = firrtl.subfield %bundle(0) : (!firrtl.bundle<a: uint<8>>) -> !firrtl.uint<8>
    // firrtl.connect %field, %value : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // vector<bundle<bundle>> -> bundle<bundle<vector>>
  // CHECK-LABEL:   @VBB
  // CHECK-SAME:    (in %port: !firrtl.bundle<nested: bundle<field: vector<uint<1>, 4>>>)
  firrtl.module @VBB(in %port: !firrtl.vector<bundle<nested: bundle<field: uint<1>>>, 4>) {
    // port[i].nested.field => port.nested.field[i]
  }

  // vector<vector<bundle>> -> bundle<vector<vector>
  // CHECK-LABEL:   @VVB
  // CHECK-SAME:    (in %port: !firrtl.bundle<field: vector<vector<uint<1>, 6>, 4>>)
  firrtl.module @VVB(in %port: !firrtl.vector<vector<bundle<field: uint<1>>, 6>, 4>) {
    // port[i][j].field => port.field[i][j]
  }

  // vector<bundle<vector<bundle>> -> bundle<bundle<vector<vector>>>
  // CHECK-LABEL:    @VBVB
  // CHECK-SAME:     (in %port: !firrtl.bundle<field_a: bundle<field_b: vector<vector<uint<1>, 4>, 8>>>)
  firrtl.module @VBVB(in %port: !firrtl.vector<bundle<field_a: vector<bundle<field_b: uint<1>>, 4>>, 8>) {

  }
  // bundle<vector<bundle>> -> bundle<bundle<vector>>


  // firrtl.module @TestSubindex() {

  //   %value   = firrtl.bundleconstant
  //   %element = firrtl.subindex %port[3] : !firrtl.bundle<a: uint<8>>
  //   firrtl.connect %element, 
  // }

  firrtl.module @TestIndexing(in %port: !firrtl.vector<bundle<a flip: uint<8>>, 4>) {
    %value  = firrtl.constant 7 : !firrtl.uint<8>
    %bundle = firrtl.subindex %port[3] : !firrtl.vector<bundle<a flip: uint<8>>, 4>
    %field  = firrtl.subfield %bundle(0) : (!firrtl.bundle<a flip: uint<8>>) -> !firrtl.uint<8>
    firrtl.connect %field, %value : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
