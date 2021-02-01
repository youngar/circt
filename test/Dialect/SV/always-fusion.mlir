// RUN: circt-opt %s -sv-always-fusion | FileCheck %s

// CHECK-LABEL: func @if_basic(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.always posedge %arg0  {
// CHECK-NEXT:     sv.ifdef "!SYNTHESIS"  {
// CHECK-NEXT:       %0 = sv.textual_value "PRINTF_COND_" : i1
// CHECK-NEXT:       %1 = rtl.and %0, %arg1 : i1
// CHECK-NEXT:       sv.if %1  {
// CHECK-NEXT:         sv.fwrite "Hi\0A"
// CHECK-NEXT:         sv.fwrite "%x"(%1) : i1
// CHECK-NEXT:       } else  {
// CHECK-NEXT:         sv.fwrite "There\0A"
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

func @if_basic(%arg0: i1, %arg1: i1) {
  sv.always posedge  %arg0 {
    sv.ifdef "!SYNTHESIS" {
      %tmp = sv.textual_value "PRINTF_COND_" : i1
      %tmp2 = rtl.and %tmp, %arg1 : i1
      sv.if %tmp2 {
        sv.fwrite "Hi\n" 
      }
      sv.if %tmp2 {
        // Test fwrite with operands.
        sv.fwrite "%x"(%tmp2) : i1
      } else {
        sv.fwrite "There\n"
      }
    }
  }
  return
}

// CHECK-LABEL: func @if_nested(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.if %arg0  {
// CHECK-NEXT:     sv.if %arg1  {
// CHECK-NEXT:       sv.fwrite "A1"
// CHECK-NEXT:       sv.fwrite "A2"
// CHECK-NEXT:     }
// CHECK-NEXT:   } else  {
// CHECK-NEXT:     sv.if %arg1  {
// CHECK-NEXT:       sv.fwrite "B1"
// CHECK-NEXT:       sv.fwrite "B2"
// CHECK-NEXT:     } else  {
// CHECK-NEXT:       sv.fwrite "C1"
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

func @if_nested(%arg0: i1, %arg1: i1) {
  sv.if %arg0 {
    sv.if %arg1 {
        sv.fwrite "A1"
    }
  } else {
    sv.if %arg1 {
        sv.fwrite "B1"
    } else {
        sv.fwrite "C1"
    }
  }
  sv.if %arg0 {
    sv.if %arg1 {
        sv.fwrite "A2"
    }
  } else {
    sv.if %arg1 {
        sv.fwrite "B2"
    }
  }
  return
}

//CHECK-LABEL: func @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.fwrite "Middle\0A"
//CHECK-NEXT:   return
//CHECK-NEXT: }

func @alwaysff_basic(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B1"
  }
  sv.fwrite "Middle\n"
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B2"
  }
  return
}

// CHECK-LABEL: func @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

func @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  return
}

