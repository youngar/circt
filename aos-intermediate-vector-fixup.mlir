firrtl.circuit "Example" {
  // Connect two ports at the root. The types of the ports change but the
  // connect remains valid.
  firrtl.module @Example(
    out %out: !firrtl.vector<uint<8>, 2>
  ) {
    %a = firrtl.constant 1 : !firrtl.uint<8>
    %b = firrtl.constant 2 : !firrtl.uint<16>

    %bundle1 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>

    %vector1  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    
    %bundle1_cpy = firrtl.subindex %vector1[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle2_cpy = firrtl.subindex %vector1[1] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>

    %bundle1_cpy_a = firrtl.subfield %bundle1_cpy[a]: !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2_cpy_a = firrtl.subfield %bundle2_cpy[a]: !firrtl.bundle<a: uint<8>, b: uint<16>>

    %vector2 = firrtl.vectorcreate %bundle1_cpy_a, %bundle2_cpy_a : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>

    firrtl.connect %out, %vector2 : !firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>
  }
}
