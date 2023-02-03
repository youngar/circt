firrtl.circuit "Example" {
  firrtl.module @Example(
    out %out: !firrtl.bundle<c: uint<8>, d: uint<16>>
  ) {
    %a = firrtl.constant 1 : !firrtl.uint<8>
    %b = firrtl.constant 2 : !firrtl.uint<16>

    // %bundle1 and %bundle2 are stored in a vector-of-bundles. They will both be destroyed.
    %bundle1 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle1_cpy = firrtl.subindex %vector[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle2_cpy = firrtl.subindex %vector[1] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle1_cpy_a = firrtl.subfield %bundle1_cpy[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2_cpy_b = firrtl.subfield %bundle2_cpy[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>

    // %bundle3 refers to %a and %b indirectly through %vector->%bundle1 and %vector->%bundle2, all of which will be destroyed
    %bundle3 = firrtl.bundlecreate %bundle1_cpy_a, %bundle2_cpy_b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<c: uint<8>, d: uint<16>>
    firrtl.connect %out, %bundle3 : !firrtl.bundle<c: uint<8>, d: uint<16>>, !firrtl.bundle<c: uint<8>, d: uint<16>>

    // The end result is, %bundle3 will refer directly to %a and %b
    // CHECK: %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %c2_ui16 = firrtl.constant 2 : !firrtl.uint<16>
    // CHECK: %0 = firrtl.bundlecreate %c1_ui8, %c2_ui16 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<c: uint<8>, d: uint<16>>
    // CHECK: firrtl.connect %out, %0 : !firrtl.bundle<c: uint<8>, d: uint<16>>, !firrtl.bundle<c: uint<8>, d: uint<16>>
  }
}


firrtl.circuit "Example" {
  firrtl.module @Example(
    in %in: !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>,
    out %out: !firrtl.bundle<c: uint<8>, d: uint<16>>
  ) {

    %bundle1_cpy = firrtl.subindex %in[0] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    %bundle2_cpy = firrtl.subindex %in[1] : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>

    %bundle1_cpy_a = firrtl.subfield %bundle1_cpy[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2_cpy_b = firrtl.subfield %bundle2_cpy[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>

    // %bundle3 refers to %a and %b indirectly through %vector->%bundle1 and %vector->%bundle2, all of which will be destroyed
    %bundle3 = firrtl.bundlecreate %bundle1_cpy_a, %bundle2_cpy_b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<c: uint<8>, d: uint<16>>
    firrtl.connect %out, %bundle3 : !firrtl.bundle<c: uint<8>, d: uint<16>>, !firrtl.bundle<c: uint<8>, d: uint<16>>
  }
}

firrtl.circuit "Trying to sink a vector into a bundle port" {
  firrtl.module @"Trying to sink a vector into a bundle port"(
    in  %in:  !firrtl.bundle<a: uint<8>, b: uint<16>>,
    out %out: !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  ) {

    %vector = firrtl.vectorcreate %in, %in :
      (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) ->
        !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>

    // in_a    = firrtl.subfield %in[a]
    // in_b    = firrtl.subfield %in[b]
    // %as     = firrtl.vectorcreate %in_a, %in_a
    // %bs     = firrtl.vectorcreate %in_b, %in_b
    // %bundle = firrtl.bundlecreate %as, %bs
    // 
    firrtl.strictconnect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }
}

firrtl.circuit "Example" {
  firrtl.module @Example(
    in %in: !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>,
    out %out: !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 1>
  ) {
    %vector = firrtl.vectorcreate %in : (!firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>)
      -> !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 1>
    firrtl.strictconnect %out, %vector : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 1>
  }
}

firrtl.circuit "Example" {
  firrtl.module @Example(
    in %in: !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>,
    out %out: !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<16>>, 2>>
  ) {
    %bundle = firrtl.bundlecreate %in : (!firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>)
      -> !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<16>>, 2>>
    firrtl.strictconnect %out, %bundle : !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<16>>, 2>>
  }
}

