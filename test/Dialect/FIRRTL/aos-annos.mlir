firrtl.circuit "Annotations" {
  //===--------------------------------------------------------------------===//
  // Annotations
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: firrtl.module @Annotations(
  firrtl.module @Annotations(
    // CHECK-SAME: in %in: !firrtl.bundle<f: vector<uint<8>, 4>, g: vector<uint<8>, 4>>
    in %in: !firrtl.vector<bundle<f: uint<8>, g: uint<8>>, 4> [
      // CHECK-SAME: {circt.fieldID = 2 : i64, class = "f"}
      // CHECK-SAME: {circt.fieldID = 7 : i64, class = "f"}
      {class = "f", circt.fieldID = 1 : i64}
    ]
    ) {

    // CHECK: %w = firrtl.wire {
    %w = firrtl.wire {
      annotations =  [
        // CHECK-SAME: {circt.fieldID = 0 : i64, class = "0"}
        {circt.fieldID = 0 : i64, class = "0"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "1"}
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "1"}
        {circt.fieldID = 1 : i64, class = "1"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "2"}
        {circt.fieldID = 2 : i64, class = "2"},
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "3"}
        {circt.fieldID = 3 : i64, class = "3"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "4"}
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "4"}
        {circt.fieldID = 4 : i64, class = "4"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "5"}
        {circt.fieldID = 5 : i64, class = "5"},
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "6"}
        {circt.fieldID = 6 : i64, class = "6"}
    ]} : 
    // CHECK-SAME: !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<8>, 4>>
    !firrtl.vector<bundle<a: uint<8>, b: uint<8>>, 4>
    
    // Targeting the bundle of the data field should explode and retarget to the
    // first element of the field vector.
    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [[{circt.fieldID = 6 : i64, class = "mem0"}]]
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{circt.fieldID = 5 : i64, class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<bundle<a: uint<8>>, 5>>
    
  }
  }