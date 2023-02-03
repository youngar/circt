firrtl.circuit "Example" {
  firrtl.module @Example(out %out : !firrtl.bundle<a: uint<8>, b: uint<16>>) {
    // CHECK: %0 = firrtl.constant 1 : !firrtl.uint<8>
    %a = firrtl.constant 1 : !firrtl.uint<8>

    // CHECK: %0 = firrtl.constant 1 : !firrtl.uint<8>
    %b = firrtl.constant 2 : !firrtl.uint<16>

    // CHECK: %1 = firrtl.vectorcreate
    // CHECK: %2 = firrtl.vectorcreate
    // CHECK: %3 = firrtl.bundlecreate
    %bundle1 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle3 = firrtl.subindex %vector[0] :  !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    firrtl.connect %out, %bundle3 : !firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>
  }
}
