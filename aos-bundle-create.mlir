firrtl.circuit "Example" {
  firrtl.module @Example(out %out : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
    %a = firrtl.constant 1 : !firrtl.uint<8>
    %b = firrtl.constant 2 : !firrtl.uint<16>
    %bundle1 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    firrtl.connect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }
}
