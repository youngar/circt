; SPDX-License-Identifier: Apache-2.0
; REQUIRES: yosys
; RUN: firtool -lowering-options=disallowLocalVariables,disallowPackedArrays,noAlwaysComb -verilog %s -o %t
; RUN: equiv-rtl.sh %t %S/AddNot.sv AddNot
circuit AddNot:
  module AddNot:
    input a: UInt<8>
    input b: UInt<8>
    output o: UInt<9>
    o <= add(a, not(b))
