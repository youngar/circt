; RUN: firtool --strip-debug-info --split-input-file %s | FileCheck %s
; RUN: split-file %s %t

; These are hand-written end-to-end tests used to ensure that we have the same
; optimization behaviour as the SFC. These tests are different than the ones in
; constantProp.mlir.  These tests are split up so that they can be run by hand on SFC.
; Look for the files in 
; `tools/circt/test/Dialect/FIRRTL/SFCTests/Output/constantProp.fir.tmp` and run
; them in SFC with `firrtl -i t0.fir -o t0 && cat t0.v`

;--- tx-0.fir

circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (UInt<1>(0), r)
    o <= r

;--- tx-1.fir

; Register only connected to a constant is removed.

circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (UInt<1>(0), r)
    r <= UInt<8>(3)
    o <= r
    
;--- t0-0.fir

; Unitialized register should have the value of any muxed in value.
; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (UInt<1>(0), r)
    r <= mux(reset, UInt<8>(3), r)
    o <= r
    ; CHECK: assign o = 8'h3;

; // -----
;--- t0-1.fir

; A reset should behave the same as a mux constant. e.g. 
; `reset => (reset, UInt<8>(0))` is equivalent to 
; `r0 <= mux(reset, UInt<8>(0), r0)`.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset, UInt<8>(3))
    o <= r
    ; CHECK: assign o = 8'h3;

; // -----
;--- t0-2.fir

; An async reset should behave the same as a mux constant.
; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : AsyncReset
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset, UInt<8>(3))
      
    o <= r
    ; CHECK: assign o = 8'h3;

; // -----
;--- t0-3.fir

; A register with no driver and a non-constant reset should not replace the reg
; with the reset.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input value : UInt<8>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset, value)
    o <= r

  ; CHECK: reg [7:0] r;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (reset)
  ; CHECK:     r <= value;
  ; CHECK:   else
  ; CHECK:     r <= value
  ; CHECK: end // always @(posedge)
  ; CHECK: assign o = r;
  
; // -----
;--- t1-0.fir

; Default value of 0 should not delete the register.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input p: UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (UInt<1>(0), r)
    r <= mux(p, UInt<8>(3), UInt<8>(0))
    o <= r

  ; CHECK: reg [7:0] r;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (p)
  ; CHECK:     r <= 8'h3;
  ; CHECK:   else
  ; CHECK:     r <= 8'h0;
  ; CHECK: end // always @(posedge)
  ; CHECK: assign o = r;

; // -----
;--- t1-1.fir

; Default value of 0 should not delete the register.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input p: UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (p, UInt<8>(3))
    r <= UInt<8>(0)
    o <= r

  ; CHECK: reg [7:0] r;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (p)
  ; CHECK:     r <= 8'h3;
  ; CHECK:   else
  ; CHECK:     r <= 8'h0;
  ; CHECK: end // always @(posedge)
  ; CHECK: assign o = r;

; // -----
;--- t1-2.fir

; Default value of 0 should not delete the register.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input p: UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (p, UInt<8>(3))
    r is invalid
    o <= r

  ; CHECK: reg [7:0] r;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (p)
  ; CHECK:     r <= 8'h3;
  ; CHECK:   else
  ; CHECK:     r <= 8'h0;
  ; CHECK: end // always @(posedge)
  ; CHECK: assign o = r;

; // -----
;--- t2.fir

; Test constant propagation through multiple registers.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input p: UInt<1>
    output o : UInt<8>
    reg r0 : UInt<8>, clock with :
      reset => (p, UInt<8>(3))
    reg r1 : UInt<8>, clock with :
      reset => (p, r0)
    o <= r1
    ; CHECK: assign o = 8'h3;
  
; // -----
;--- t3.fir

; Should see through complicated mux-trees of resets and stil decide to
; optimize to the reset value.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset0 : UInt<1>
    input reset1 : UInt<1>
    input reset2 : UInt<1>
    input reset3 : UInt<1>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset0, UInt<8>(3))
    node n0 = mux(reset1, r, UInt<8>(3))
    node n1 = mux(reset2, UInt<8>(3), r)
    r <= mux(reset3, n0, n1)
    o <= r
    ; CHECK: assign o = 8'h3;

; // -----
;--- ta.fir

; Should see through wires.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    wire w : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset, mux(reset, UInt<8>(3), w))
    w <= r
    o <= r
    ; CHECK: assign o = 8'h3;

; // -----
;--- t4-0.fir

; Register only connect to another register is removed.
; SFC does not optimize away either register

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    input value : UInt<8>
    output o : UInt<8>

    reg r0 : UInt<8>, clock with :
      reset => (reset, UInt(0))
    r0 <= value

    reg r1 : UInt<8>, clock with :
      reset => (UInt(0), r1)
    r1 <= r0

    o <= r1

; // -----
;--- t4-0-1.fir

; Early propagation of constants should not affect

circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<1>

    reg r0 : UInt<1>, clock with :
      reset => (UInt(0), r0)

    reg r1 : UInt<1>, clock with :
      reset => (UInt(0), r0)
      
    r1 <= mux(reset, UInt(1), r0)
    r0 <= mux(r1, r0, UInt(0))

    o <= r1
; // -----
;--- t4-1.fir

; The first register should be optimized to 0, since it has no drives. The
; second register should be a mux(p, r0, 3). If we optimize the first register
; to `Undef` then we will optimize the second register to `r1 = 3`.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>

    reg r0 : UInt<8>, clock with :
      reset => (UInt<1>(0), r0)

    reg r1 : UInt<8>, clock with :
      reset => (reset, r0)
    r1 <= UInt<8>(3)

    o <= r1

  ; CHECK: reg [7:0] r1;
  ; CHECK: assign o = r1;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (reset) begin
  ; CHECK:     r1 <= 8'h0;
  ; CHECK:   end else begin
  ; CHECK:     r1 <= 8'h3;
  ; CHECK:   end
  ; CHECK: end

; // -----
;--- t4.fir

; Each register should be initialized to a "unique" constant.  We can't optimize
; either register away here. We can't make the assumption that `r0 == r1` and
; remove the registers here.

; CHECK-LABEL: module Test
circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o0 : UInt<8>
    output o1 : UInt<8>

    reg r0 : UInt<8>, clock with :
      reset => (UInt<1>(0), r0)

    reg r1 : UInt<8>, clock with :
      reset => (UInt<1>(0), r1)

    r0 <= mux(reset, UInt<8>(3), r1)
    r1 <= mux(reset, UInt<8>(3), r0)
    o0 <= r0
    o1 <= r1

  ; CHECK: reg [7:0] r0;
  ; CHECK: reg [7:0] r1;
  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (reset) begin
  ; CHECK:     r0 <= 8'h3;
  ; CHECK:   end else begin
  ; CHECK:     r0 <= r1;
  ; CHECK:   end
  ; CHECK:   if (reset) begin
  ; CHECK:     r1 <= 8'h3;
  ; CHECK:   end else begin
  ; CHECK:     r1 <= r0;
  ; CHECK:   end
  ; CHECK: end
  ; CHECK: assign o0 = r0;
  ; CHECK: assign o1 = r1;

; // -----
;--- t5.fir

; A register's initial state should *not* propogate across module boundaries.

circuit Test:
  module Test:
    input clock : Clock
    input reset : UInt<1>
    output o : UInt<8>
    inst child of Child
    child.clock <= clock
    child.reset <= reset
    ; Connect the child register to itself.
    child.value <= child.o
    o <= child.o

  ; CHECK-LABEL: module Child
  module Child:
    input clock : Clock
    input reset : UInt<1>
    input value: UInt<8>
    output o : UInt<8>
    reg r : UInt<8>, clock with :
      reset => (reset, UInt<8>(3))
    ; Connect the register to itself across a module boundary.
    r <= value
    o <= r

  ; CHECK: always @(posedge clock) begin
  ; CHECK:   if (reset) begin
  ; CHECK:     r <= 8'h3;
  ; CHECK:   end else begin
  ; CHECK:     r <= value;
  ; CHECK:   end
  ; CHECK: end
  ; CHECK: assign o = r;