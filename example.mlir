firrtl.circuit "Example1" {
  // Connect two ports at the root. The types of the ports change but the
  // connect remains valid.
  firrtl.module @Example1(
    in  %ip: !firrtl.vector<bundle<a: uint<8>>, 4>,
    out %op: !firrtl.vector<bundle<a: uint<8>>, 4>
  ) {
    firrtl.connect %op, %ip : !firrtl.vector<bundle<a: uint<8>>, 4>, !firrtl.vector<bundle<a: uint<8>>, 4>
  }

//   firrtl.module @Example2(
//     in  %ip: !firrtl.vector<bundle<a: uint<8>>, 4>,
//     out %op: !firrtl.vector<bundle<a: uint<8>>, 4>
//   ) {
//     %ib = firrtl.subindex %ip[0] : !firrtl.vector<bundle<a: uint<8>>, 4>
//     %ix = firrtl.subfield %ib(0) : !firrtl.bundle<a: uint<8>>)
//     %ob = firrtl.subindex %op[0] : !firrtl.vector<bundle<a: uint<8>>, 4>
//     %ox = firrtl.subfield %ob(0) : !firrtl.bundle<a: uint<8>>)

//     firrtl.connect %ox, %ix : !firrtl.uint<8>, !firrtl.uint<8>
//   }

  firrtl.module @VBVB(
    in  %ip: !firrtl.vector<bundle<a: vector<bundle<b: uint<8>>, 2>>, 4>,
    out %op: !firrtl.vector<bundle<a: vector<bundle<b: uint<8>>, 2>>, 4>
  ) {
    %ib = firrtl.subindex %ip[0] { test = "fart" } : !firrtl.vector<bundle<a: vector<bundle<b: uint<8>>, 2>>, 4>
    %ob = firrtl.subindex %op[0] : !firrtl.vector<bundle<a: vector<bundle<b: uint<8>>, 2>>, 4>

    %ibv = firrtl.subfield %ib[a] { test = "fart" } : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>>
    %obv = firrtl.subfield %ob[a] : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>>
    firrtl.connect %obv, %ibv: !firrtl.vector<bundle<b: uint<8>>, 2>, !firrtl.vector<bundle<b: uint<8>>, 2>

    // %ix = firrtl.subfield %ib(0) : !firrtl.bundle<a: uint<8>>)
    // %ob = firrtl.subindex %op[0] : !firrtl.vector<bundle<a: uint<8>>, 4>
    // %ox = firrtl.subfield %ob(0) : !firrtl.bundle<a: uint<8>>)
    // firrtl.connect %ox, %ix : !firrtl.uint<8>, !firrtl.uint<8>
  }

  firrtl.module @Example2(
    in  %ip: !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>,
    out %op: !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
  ) {
    %ib = firrtl.subindex %ip[2] : !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
    %ob = firrtl.subindex %op[3] : !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
    firrtl.connect %ob, %ib : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
  }

// //   firrtl.module @Example2(
// //     in  %ip: !firrtl.bundle<x: uint<8>, v: !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>>
// //     out %op: !firrtl.bundle<x: uint<8>, v: !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>>
// //   ) {
// //     %ib = firrtl.subindex %ip[3] : !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
// //     %ob = firrtl.subindex %op[3] : !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
// //     firrtl.connect %ob, %ib : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
// //   }

//   // The input port undergoes AOS->SOA, the output port is left alone.
//   // The subindex op slices our input SOA, so %b is exploded.
//   // op has to be manually exploded to connect to the exploded %b.
//   firrtl.module @Example3(
//     in  %ip: !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>,
//     out %op: !firrtl.bundle<a: uint<8>, b: uint<9>>
//   ) {
//     %b = firrtl.subindex %ip[3] : !firrtl.vector<bundle<a: uint<8>, b: uint<9>>, 4>
//     firrtl.connect %op, %b: !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
//   }

//   firrtl.module @Example4(
//     in  %ip: !firrtl.bundle<>,
//     out %op: !firrtl.bundle<>
//   ) {
//     firrtl.connect %op, %ip: !firrtl.bundle<>, !firrtl.bundle<>
//   }

//   firrtl.module @Example5(
//     in  %ip: !firrtl.vector<bundle<>, 2>,
//     out %op: !firrtl.vector<bundle<>, 2>
//   ) {
//     // test 1: connection of empty bundles
//     firrtl.connect %op, %ip: !firrtl.vector<bundle<>, 2>, !firrtl.vector<bundle<>, 2>

//     // test 2: connect of empty slice of vector: connect should be blown away.
//     %ib = firrtl.subindex %ip[0] : !firrtl.vector<bundle<>, 2>
//     %ob = firrtl.subindex %op[1] : !firrtl.vector<bundle<>, 2>
//     firrtl.connect %ob, %ib: !firrtl.bundle<>, !firrtl.bundle<>

//     // test 3: connec
//   }

//   firrtl.module @Example6(
//     in  %ip: !firrtl.vector<bundle<>, 2>,
//     out %op: !firrtl.bundle<>
//   ) {
//     // test 2: connect of empty slice of vector: connect should be blown away.
//     %ib = firrtl.subindex %ip[0] : !firrtl.vector<bundle<>, 2>
//     firrtl.connect %op, %ib: !firrtl.bundle<>, !firrtl.bundle<>
//   }

// //   firrtl.module @Example6(
// //     in  %ip: !firrtl.vector<bundle<>, 2>,
// //     out %op: !firrtl.bundle<>
// //   ) {
// //     // test 2: connect of empty slice of vector: connect should be blown away.
// //     %ib = firrtl.subindex %ip[0] : !firrtl.vector<bundle<>, 2>
// //     firrtl.connect %op, %ib: !firrtl.bundle<>, !firrtl.bundle<>
// //   }

//   firrtl.module @Example7(
//     in  %ip: !firrtl.vector<vector<uint<8>, 3>, 2>,
//     out %op: !firrtl.vector<vector<uint<8>, 3>, 2>
//   ) {
//     // test 2: connect of empty slice of vector: connect should be blown away.
//     firrtl.connect %op, %ip: !firrtl.vector<vector<uint<8>, 3>, 2>, !firrtl.vector<vector<uint<8>, 3>, 2>
//   }

//   firrtl.module @Example8(
//     in  %ip: !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, b: vector<uint<8>, 4>>,
//     out %op: !firrtl.vector<uint<8>, 4>
//   ) {
//     %iv = firrtl.subfield %ip(1) : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, b: vector<uint<8>, 4>>)
//     firrtl.connect %op, %iv: !firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<8>, 4>
//   }

//   firrtl.module @Example9(
//     in  %ip: !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, b: vector<uint<8>, 4>>,
//     out %op: !firrtl.uint<8>
//   ) {
//     %iv = firrtl.subfield %ip(1) : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, b: vector<uint<8>, 4>>)
//     %iu = firrtl.subindex %iv[2] : !firrtl.vector<uint<8>, 4>
//     firrtl.connect %op, %iu: !firrtl.uint<8>, !firrtl.uint<8>
//   }
 
//   firrtl.module @Example9(
//     in  %ip: !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, c: bundle<d: uint<8>, e: uint<16>>>,
//     out %op: !firrtl.bundle<d: uint<8>, e: uint<16>>
//   ) {
//     %iv = firrtl.subfield %ip(1) : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, c: bundle<d: uint<8>, e: uint<16>>>)
//     firrtl.connect %op, %iv: !firrtl.bundle<d: uint<8>, e: uint<16>>, !firrtl.bundle<d: uint<8>, e: uint<16>>
//   }

//   firrtl.module @Example10(
//     in  %ip: !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, c: vector<vector<uint<8>, 2>, 3>>,
//     out %op: !firrtl.vector<uint<8>, 2>
//   ) {
//     %iv1 = firrtl.subfield %ip(1) : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>, c: vector<vector<uint<8>, 2>, 3>>)
//     %iv2 = firrtl.subindex %iv1[0] : !firrtl.vector<vector<uint<8>, 2>, 3>
//     firrtl.connect %op, %iv2: !firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>
//   }

  // CHECK-LABEL: firrtl.module @Example
//   firrtl.module @Example2(
//     // CHECK-SAME: in  %ip: !firrtl.vector<vector<bundle<a: uint<8>, b: uint<9>>, 4>, 6>,
//     in  %ip: !firrtl.vector<vector<bundle<a: uint<8>, b: uint<9>>, 4>, 6>,
//     // CHECK-SAME: out %op: bundle<a: uint<8>, b: uint<9>>
//     out %op: !firrtl.bundle<a: uint<8>, b: uint<9>>
//   ) {


//     //  scalarize the input bundle:
//     // CHECK: %0 = firrtl.subfield %ip(0)
//     // CHECK: %1 = firrtl.subfield %ip(1)
//     %v0 = firrtl.subindex %ip[3] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<9>>, 4>, 6>
//     //  perform the index operations:
//     // CHECK: %2 = firrtl.subindex %0[3]
//     // CHECK: %3 = firrtl.subindex %1[3]
//     // CHECK: %4 = firrtl.subindex %2[5]
//     // CHECK: %5 = firrtl.subindex %3[5]

//     %v1 = firrtl.subindex %v0[5] : !firrtl<vector<bundle<a: uint<8>, b: uint<9>>, 4>
//     //  connect:
//     //   scalarize output ports:
//     // CHECK: %6 = firrtl.subfield %op(0)
//     // CHECK: %7 = firrtl.subfield %op(1)
//     // CHECK: firrtl.connect %6, %4
//     // CHECK: firrtl.connect %7, %5
//     firrtl.connect %op, %v1 : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
//   }

  firrtl.module @Constant1() {
    // CHECK:      %0 = firrtl.aggregateconstant
    // CHECK-SAME:   [[1 : ui8, 3 : ui8], [2 : ui16, 4 : ui16]]
    // CHECK-SAME:   : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
    %0 = firrtl.aggregateconstant
      [[1 : ui8, 2 : ui16], [3 : ui8, 4 : ui16]]
      : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>

    // CHE
  }

  firrtl.module @Constant2(
    out %out0 : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>,
    out %out1 : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {

    // CHECK       %1 = firrtl.aggregateconstant
    // CHECK-SAME:   [[]]
    %c1 = firrtl.aggregateconstant
      [[[1 : ui8, 2 : ui16], [3 : ui8, 4 : ui16]],
       [[5 : ui8, 6 : ui16], [7 : ui8, 8 : ui16]]]
      : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 2>

    %v0 = firrtl.subindex %c1[0] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 2>
    %v1 = firrtl.subindex %c1[1] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<16>>, 2>, 2>
    firrtl.connect %out0, %v0 : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    firrtl.connect %out1, %v1 : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>

      // %0 = firrtl.subfield %out1[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
      // %1 = firrtl.subfield %out1[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
      // %2 = firrtl.subfield %out0[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
      // %3 = firrtl.subfield %out0[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
      // %4 = firrtl.aggregateconstant [[[1 : ui8, 3 : ui8], [5 : ui8, 7 : ui8]], [[2 : ui16, 4 : ui16], [6 : ui16, 8 : ui16]]] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<16>, 2>, 2>>
      // %5 = firrtl.subfield %4[a] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<16>, 2>, 2>>
      // %6 = firrtl.subfield %4[b] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<16>, 2>, 2>>
      // %7 = firrtl.subindex %5[1] : !firrtl.vector<vector<uint<8>, 2>, 2>
      // %8 = firrtl.subindex %6[1] : !firrtl.vector<vector<uint<16>, 2>, 2>
      // %9 = firrtl.subfield %4[a] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<16>, 2>, 2>>
      // %10 = firrtl.subfield %4[b] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<16>, 2>, 2>>
      // %11 = firrtl.subindex %9[0] : !firrtl.vector<vector<uint<8>, 2>, 2>
      // %12 = firrtl.subindex %10[0] : !firrtl.vector<vector<uint<16>, 2>, 2>
      // firrtl.connect %2, %11 : !firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>
      // firrtl.connect %3, %12 : !firrtl.vector<uint<16>, 2>, !firrtl.vector<uint<16>, 2>
      // firrtl.connect %0, %7 : !firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>
      // firrtl.connect %1, %8 : !firrtl.vector<uint<16>, 2>, !firrtl.vector<uint<16>, 2>
  }

  firrtl.module @Constant3(out %out0 : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
  }

  firrtl.module @Create1(out %out : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
    %a = firrtl.constant 1 : !firrtl.uint<8>
    %b = firrtl.constant 2 : !firrtl.uint<16>
    %bundle1 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = firrtl.bundlecreate %a, %b : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    %vector  = firrtl.vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    firrtl.connect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }
}
