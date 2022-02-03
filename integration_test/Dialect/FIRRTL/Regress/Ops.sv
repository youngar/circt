module Ops(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:6:10
  input  [3:0]  sel,
  input  [7:0]  is, iu,
  output [13:0] os,
  output [12:0] ou,
  output        obool);

  wire _T = sel == 4'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:18:10
  wire _T_0 = sel == 4'h1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:22:12
  wire _T_1 = sel == 4'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:26:14
  wire [8:0] _T_2 = _T ? {is, 1'h0} : _T_0 ? 9'h0 : {8'h0, _T_1};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:15:11, :19:13, :31:16
  wire _T_3 = sel == 4'h4;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:34:18
  wire _T_4 = sel == 4'h5;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:37:20
  assign os = {{5{_T_2[8]}}, _T_2};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:6:10
  assign ou = {4'h0, _T ? {iu, 1'h0} : _T_0 ? 9'h0 : _T_1 ? 9'h1 : sel == 4'h3 ? 9'h0 : _T_3 | _T_4 ?
                {1'h0, is} + {1'h0, iu} : _T_3 ? 9'h0 : {7'h0, _T_4 ? 2'h2 : _T_3 ? 2'h0 : _T_4 ? 2'h2 :
                {_T_3, 1'h0}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:6:10, :15:11, :18:10, :20:13, :26:22, :27:47, :30:16, :31:16, :35:21, :47:29, :58:32
  assign obool = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/Ops.fir:6:10, :15:11
endmodule

