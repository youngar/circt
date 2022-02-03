module AddNot(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/AddNot.fir:6:10
  input  [7:0] a, b,
  output [8:0] o);

  assign o = {1'h0, a} + {1'h0, ~b};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/AddNot.fir:6:10, :10:{10,17}
endmodule

