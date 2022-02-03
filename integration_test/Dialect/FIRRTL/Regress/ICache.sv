// Standard header to adapt well known macros to our needs.
`ifdef RANDOMIZE_REG_INIT
  `define RANDOMIZE
`endif

// RANDOM may be set to an expression that produces a 32-bit random unsigned value.
`ifndef RANDOM
  `define RANDOM {$random}
`endif

// Users can define INIT_RANDOM as general code that gets injected into the
// initializer block for modules with registers.
`ifndef INIT_RANDOM
  `define INIT_RANDOM
`endif

// If using random initialization, you can also define RANDOMIZE_DELAY to
// customize the delay used, otherwise 0.002 is used.
`ifndef RANDOMIZE_DELAY
  `define RANDOMIZE_DELAY 0.002
`endif

// Define INIT_RANDOM_PROLOG_ for use in our modules below.
`ifdef RANDOMIZE
  `ifdef VERILATOR
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
  `else
    `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
  `endif
`else
  `define INIT_RANDOM_PROLOG_
`endif

module tag_array_0_ext(	// ICache.scala:97:25
  input  [5:0]  RW0_addr,
  input         RW0_en, RW0_clk, RW0_wmode,
  input  [19:0] RW0_wdata,
  output [19:0] RW0_rdata);

  reg [19:0] Memory[0:63];

  always @(posedge RW0_clk) begin
    if (RW0_en & RW0_wmode)
      Memory[RW0_addr] <= RW0_wdata;
  end // always @(posedge)
  assign RW0_rdata = RW0_en & ~RW0_wmode ? Memory[RW0_addr] : 20'bx;	// ICache.scala:97:25
endmodule

module _ext(	// ICache.scala:132:28
  input  [8:0]  RW0_addr,
  input         RW0_en, RW0_clk, RW0_wmode,
  input  [63:0] RW0_wdata,
  output [63:0] RW0_rdata);

  reg [63:0] Memory[0:511];

  always @(posedge RW0_clk) begin
    if (RW0_en & RW0_wmode)
      Memory[RW0_addr] <= RW0_wdata;
  end // always @(posedge)
  assign RW0_rdata = RW0_en & ~RW0_wmode ? Memory[RW0_addr] : 64'bx;	// ICache.scala:132:28
endmodule

module ICache(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10
  input         clock, reset, io_req_valid,
  input  [38:0] io_req_bits_addr,
  input  [31:0] io_s1_paddr,
  input         io_s1_kill, io_s2_kill, io_resp_ready, io_invalidate, io_mem_0_a_ready,
  input         io_mem_0_b_valid,
  input  [2:0]  io_mem_0_b_bits_opcode,
  input  [1:0]  io_mem_0_b_bits_param,
  input  [3:0]  io_mem_0_b_bits_size,
  input         io_mem_0_b_bits_source,
  input  [31:0] io_mem_0_b_bits_address,
  input  [7:0]  io_mem_0_b_bits_mask,
  input  [63:0] io_mem_0_b_bits_data,
  input         io_mem_0_c_ready, io_mem_0_d_valid,
  input  [2:0]  io_mem_0_d_bits_opcode,
  input  [1:0]  io_mem_0_d_bits_param,
  input  [3:0]  io_mem_0_d_bits_size,
  input         io_mem_0_d_bits_source,
  input  [3:0]  io_mem_0_d_bits_sink,
  input  [2:0]  io_mem_0_d_bits_addr_lo,
  input  [63:0] io_mem_0_d_bits_data,
  input         io_mem_0_d_bits_error, io_mem_0_e_ready,
  output        io_resp_valid,
  output [15:0] io_resp_bits_data,
  output [63:0] io_resp_bits_datablock,
  output        io_mem_0_a_valid,
  output [2:0]  io_mem_0_a_bits_opcode, io_mem_0_a_bits_param,
  output [3:0]  io_mem_0_a_bits_size,
  output        io_mem_0_a_bits_source,
  output [31:0] io_mem_0_a_bits_address,
  output [7:0]  io_mem_0_a_bits_mask,
  output [63:0] io_mem_0_a_bits_data,
  output        io_mem_0_b_ready, io_mem_0_c_valid,
  output [2:0]  io_mem_0_c_bits_opcode, io_mem_0_c_bits_param,
  output [3:0]  io_mem_0_c_bits_size,
  output        io_mem_0_c_bits_source,
  output [31:0] io_mem_0_c_bits_address,
  output [63:0] io_mem_0_c_bits_data,
  output        io_mem_0_c_bits_error, io_mem_0_d_ready, io_mem_0_e_valid,
  output [3:0]  io_mem_0_e_bits_sink);

  wire         _T;	// ICache.scala:139:50
  wire         _T_0;	// ICache.scala:133:30
  wire         _T_1;	// ICache.scala:139:50
  wire         _T_2;	// ICache.scala:133:30
  wire         _T_3;	// ICache.scala:139:50
  wire         _T_4;	// ICache.scala:133:30
  wire         _T_5;	// ICache.scala:139:50
  wire [8:0]   _T_6;	// ICache.scala:138:28
  wire [8:0]   _T_7;	// ICache.scala:136:63
  wire         _T_8;	// ICache.scala:133:30
  wire         _T_9;	// ICache.scala:101:84
  wire         _T_10;	// ICache.scala:101:84
  wire         _T_11;	// ICache.scala:101:84
  wire         _T_12;	// ICache.scala:98:83
  wire [5:0]   _T_13;	// ICache.scala:98:42
  wire         _T_14;	// ICache.scala:97:25
  wire         _T_15;	// ICache.scala:97:25
  wire         _T_16;	// ICache.scala:97:25
  wire         _T_17;	// ICache.scala:97:25
  wire [63:0]  mem_RW0_rdata;	// ICache.scala:132:28
  wire [63:0]  mem_RW0_rdata_18;	// ICache.scala:132:28
  wire [63:0]  mem_RW0_rdata_19;	// ICache.scala:132:28
  wire [63:0]  mem_RW0_rdata_20;	// ICache.scala:132:28
  wire [19:0]  tag_array_3_RW0_rdata;	// ICache.scala:97:25
  wire [19:0]  tag_array_2_RW0_rdata;	// ICache.scala:97:25
  wire [19:0]  tag_array_1_RW0_rdata;	// ICache.scala:97:25
  wire [19:0]  tag_array_0_RW0_rdata;	// ICache.scala:97:25
  reg  [1:0]   state;	// ICache.scala:67:18
  reg          invalidated;	// ICache.scala:68:24
  reg  [31:0]  refill_addr;	// ICache.scala:71:24
  reg          s1_valid;	// ICache.scala:74:21
  reg  [4:0]   _T_21_24;	// Reg.scala:26:44
  reg  [15:0]  _T_22_26;	// LFSR.scala:22:19
  reg  [255:0] vb_array;	// ICache.scala:104:21
  reg          s1_dout_valid;	// Reg.scala:14:44
  reg          _T_23_29;	// Reg.scala:34:16
  reg          _T_24;	// Reg.scala:34:16
  reg          _T_25_36;	// Reg.scala:34:16
  reg          _T_26;	// Reg.scala:34:16
  reg  [63:0]  _T_27_42;	// Reg.scala:34:16
  reg  [63:0]  _T_28_43;	// Reg.scala:34:16
  reg  [63:0]  _T_29;	// Reg.scala:34:16
  reg  [63:0]  _T_30_44;	// Reg.scala:34:16
  reg          _T_31_45;	// Reg.scala:26:44
  reg          _T_32_46;	// Reg.scala:34:16
  reg          _T_33_47;	// Reg.scala:34:16
  reg          _T_34_48;	// Reg.scala:34:16
  reg          _T_35_49;	// Reg.scala:34:16
  reg  [63:0]  _T_36;	// Reg.scala:34:16
  reg  [63:0]  _T_37_50;	// Reg.scala:34:16
  reg  [63:0]  _T_38_51;	// Reg.scala:34:16
  reg  [63:0]  _T_39_52;	// Reg.scala:34:16

  wire _T_18 = state == 2'h0;	// ICache.scala:67:18, :75:52
  wire _T_19 = s1_valid & ~io_s1_kill & _T_18;	// ICache.scala:75:{28,31,43}
  wire [5:0] _s1_idx = io_s1_paddr[11:6];	// ICache.scala:76:27
  wire [19:0] _s1_tag = io_s1_paddr[31:12];	// ICache.scala:77:27
  wire _T_20 = _T_19 & ~io_resp_ready;	// ICache.scala:69:15, :81:67
  wire _T_21 = io_req_valid & _T_18 & ~_T_20;	// ICache.scala:81:{52,55}
  wire [19:0] _refill_tag = refill_addr[31:12];	// ICache.scala:87:17, :89:31
  wire [5:0] _refill_idx = refill_addr[11:6];	// ICache.scala:87:17, :90:31
  wire [22:0] _T_22 = 23'hFF << io_mem_0_d_bits_size;	// package.scala:19:71
  wire [4:0] _T_23 = io_mem_0_d_bits_opcode[0] ? ~(_T_22[7:3]) : 5'h0;	// Edges.scala:90:36, :199:14, package.scala:19:40
  wire [4:0] _T_25 = _T_21_24 - 5'h1;	// Edges.scala:208:28
  wire _refill_done = (_T_21_24 == 5'h1 | _T_23 == 5'h0) & io_mem_0_d_valid;	// Edges.scala:199:14, :208:28, :210:{25,37,47}, :211:22
  wire [4:0] _refill_cnt = _T_23 & ~_T_25;	// Edges.scala:212:{25,27}
  wire [1:0] _repl_way = _T_22_26[1:0];	// ICache.scala:95:56, LFSR.scala:23:40
  assign _T_17 = _T_12 | _refill_done;	// ICache.scala:97:25, :98:83
  assign _T_16 = _T_12 | _refill_done;	// ICache.scala:97:25, :98:83
  assign _T_15 = _T_12 | _refill_done;	// ICache.scala:97:25, :98:83
  assign _T_14 = _T_12 | _refill_done;	// ICache.scala:97:25, :98:83
  assign _T_13 = io_req_bits_addr[11:6];	// ICache.scala:98:42
  assign _T_12 = ~_refill_done & _T_21;	// ICache.scala:98:{70,83}
  assign _T_11 = _repl_way == 2'h0;	// ICache.scala:67:18, :101:84
  assign _T_10 = _repl_way == 2'h1;	// ICache.scala:101:84, :154:27
  assign _T_9 = _repl_way == 2'h2;	// ICache.scala:101:84
  wire [255:0] _T_27 = vb_array >> _s1_idx;	// ICache.scala:106:32, :122:43
  wire _T_28 = tag_array_0_RW0_rdata == _s1_tag;	// ICache.scala:97:25, :125:46
  wire _T_30 = ~io_invalidate & _T_27[0] & (s1_dout_valid ? _T_28 : _T_23_29);	// ICache.scala:122:{17,43}, :126:28, Reg.scala:35:23
  wire [255:0] _T_31 = vb_array >> {250'h1, _s1_idx};	// ICache.scala:106:32, :122:43
  wire _T_32 = tag_array_1_RW0_rdata == _s1_tag;	// ICache.scala:97:25, :125:46
  wire _T_33 = ~io_invalidate & _T_31[0] & (s1_dout_valid ? _T_32 : _T_24);	// ICache.scala:122:{17,43}, :126:28, Reg.scala:35:23
  wire [255:0] _T_34 = vb_array >> {250'h2, _s1_idx};	// ICache.scala:106:32, :122:43
  wire _T_35 = tag_array_2_RW0_rdata == _s1_tag;	// ICache.scala:97:25, :125:46
  wire _T_37 = ~io_invalidate & _T_34[0] & (s1_dout_valid ? _T_35 : _T_25_36);	// ICache.scala:122:{17,43}, :126:28, Reg.scala:35:23
  wire [255:0] _T_38 = vb_array >> {250'h3, _s1_idx};	// ICache.scala:106:32, :122:43
  wire _T_39 = tag_array_3_RW0_rdata == _s1_tag;	// ICache.scala:97:25, :125:46
  wire _T_40 = ~io_invalidate & _T_38[0] & (s1_dout_valid ? _T_39 : _T_26);	// ICache.scala:122:{17,43}, :126:28, Reg.scala:35:23
  wire _T_41 = _T_30 | _T_33 | _T_37 | _T_40;	// ICache.scala:129:44
  assign _T_8 = io_mem_0_d_valid & _T_11;	// ICache.scala:101:84, :133:30
  assign _T_7 = {refill_addr[11:8], refill_addr[7:6] | _refill_cnt[4:3], _refill_cnt[2:0]};	// ICache.scala:87:17, :136:63
  assign _T_6 = io_req_bits_addr[11:3];	// ICache.scala:138:28
  assign _T_5 = ~_T_8 & _T_21;	// ICache.scala:133:30, :139:{45,50}
  assign _T_4 = io_mem_0_d_valid & _T_10;	// ICache.scala:101:84, :133:30
  assign _T_3 = ~_T_4 & _T_21;	// ICache.scala:133:30, :139:{45,50}
  assign _T_2 = io_mem_0_d_valid & _T_9;	// ICache.scala:101:84, :133:30
  assign _T_1 = ~_T_2 & _T_21;	// ICache.scala:133:30, :139:{45,50}
  assign _T_0 = io_mem_0_d_valid & &_repl_way;	// ICache.scala:101:84, :133:30
  assign _T = ~_T_0 & _T_21;	// ICache.scala:133:30, :139:{45,50}
  `ifndef SYNTHESIS	// ICache.scala:67:18
    `ifdef RANDOMIZE_REG_INIT	// ICache.scala:67:18
      reg [31:0] _RANDOM;	// ICache.scala:67:18
      reg [31:0] _RANDOM_40;	// ICache.scala:71:24
      reg [31:0] _RANDOM_41;	// ICache.scala:104:21
      reg [31:0] _RANDOM_42;	// ICache.scala:104:21
      reg [31:0] _RANDOM_43;	// ICache.scala:104:21
      reg [31:0] _RANDOM_44;	// ICache.scala:104:21
      reg [31:0] _RANDOM_45;	// ICache.scala:104:21
      reg [31:0] _RANDOM_46;	// ICache.scala:104:21
      reg [31:0] _RANDOM_47;	// ICache.scala:104:21
      reg [31:0] _RANDOM_48;	// ICache.scala:104:21
      reg [31:0] _RANDOM_49;	// Reg.scala:34:16
      reg [31:0] _RANDOM_50;	// Reg.scala:34:16
      reg [31:0] _RANDOM_51;	// Reg.scala:34:16
      reg [31:0] _RANDOM_52;	// Reg.scala:34:16
      reg [31:0] _RANDOM_53;	// Reg.scala:34:16
      reg [31:0] _RANDOM_54;	// Reg.scala:34:16
      reg [31:0] _RANDOM_55;	// Reg.scala:34:16
      reg [31:0] _RANDOM_56;	// Reg.scala:34:16
      reg [31:0] _RANDOM_57;	// Reg.scala:34:16
      reg [31:0] _RANDOM_58;	// Reg.scala:34:16
      reg [31:0] _RANDOM_59;	// Reg.scala:34:16
      reg [31:0] _RANDOM_60;	// Reg.scala:34:16
      reg [31:0] _RANDOM_61;	// Reg.scala:34:16
      reg [31:0] _RANDOM_62;	// Reg.scala:34:16
      reg [31:0] _RANDOM_63;	// Reg.scala:34:16
      reg [31:0] _RANDOM_64;	// Reg.scala:34:16
      reg [31:0] _RANDOM_65;	// Reg.scala:34:16

    `endif
    initial begin	// ICache.scala:67:18
      `INIT_RANDOM_PROLOG_	// ICache.scala:67:18
      `ifdef RANDOMIZE_REG_INIT	// ICache.scala:67:18
        _RANDOM = `RANDOM;	// ICache.scala:67:18
        state = _RANDOM[1:0];	// ICache.scala:67:18
        invalidated = _RANDOM[2];	// ICache.scala:68:24
        _RANDOM_40 = `RANDOM;	// ICache.scala:71:24
        refill_addr = {_RANDOM_40[2:0], _RANDOM[31:3]};	// ICache.scala:71:24
        s1_valid = _RANDOM_40[3];	// ICache.scala:74:21
        _T_21 = _RANDOM_40[8:4];	// Reg.scala:26:44
        _T_22 = _RANDOM_40[24:9];	// LFSR.scala:22:19
        _RANDOM_41 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_42 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_43 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_44 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_45 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_46 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_47 = `RANDOM;	// ICache.scala:104:21
        _RANDOM_48 = `RANDOM;	// ICache.scala:104:21
        vb_array = {_RANDOM_48[24:0], _RANDOM_47, _RANDOM_46, _RANDOM_45, _RANDOM_44, _RANDOM_43, _RANDOM_42, _RANDOM_41, _RANDOM_40[31:25]};	// ICache.scala:104:21
        s1_dout_valid = _RANDOM_48[25];	// Reg.scala:14:44
        _T_23 = _RANDOM_48[26];	// Reg.scala:34:16
        _T_24 = _RANDOM_48[27];	// Reg.scala:34:16
        _T_25 = _RANDOM_48[28];	// Reg.scala:34:16
        _T_26 = _RANDOM_48[29];	// Reg.scala:34:16
        _RANDOM_49 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_50 = `RANDOM;	// Reg.scala:34:16
        _T_27 = {_RANDOM_50[29:0], _RANDOM_49, _RANDOM_48[31:30]};	// Reg.scala:34:16
        _RANDOM_51 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_52 = `RANDOM;	// Reg.scala:34:16
        _T_28 = {_RANDOM_52[29:0], _RANDOM_51, _RANDOM_50[31:30]};	// Reg.scala:34:16
        _RANDOM_53 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_54 = `RANDOM;	// Reg.scala:34:16
        _T_29 = {_RANDOM_54[29:0], _RANDOM_53, _RANDOM_52[31:30]};	// Reg.scala:34:16
        _RANDOM_55 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_56 = `RANDOM;	// Reg.scala:34:16
        _T_30 = {_RANDOM_56[29:0], _RANDOM_55, _RANDOM_54[31:30]};	// Reg.scala:34:16
        _T_31 = _RANDOM_56[30];	// Reg.scala:26:44
        _T_32 = _RANDOM_56[31];	// Reg.scala:34:16
        _RANDOM_57 = `RANDOM;	// Reg.scala:34:16
        _T_33 = _RANDOM_57[0];	// Reg.scala:34:16
        _T_34 = _RANDOM_57[1];	// Reg.scala:34:16
        _T_35 = _RANDOM_57[2];	// Reg.scala:34:16
        _RANDOM_58 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_59 = `RANDOM;	// Reg.scala:34:16
        _T_36 = {_RANDOM_59[2:0], _RANDOM_58, _RANDOM_57[31:3]};	// Reg.scala:34:16
        _RANDOM_60 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_61 = `RANDOM;	// Reg.scala:34:16
        _T_37 = {_RANDOM_61[2:0], _RANDOM_60, _RANDOM_59[31:3]};	// Reg.scala:34:16
        _RANDOM_62 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_63 = `RANDOM;	// Reg.scala:34:16
        _T_38 = {_RANDOM_63[2:0], _RANDOM_62, _RANDOM_61[31:3]};	// Reg.scala:34:16
        _RANDOM_64 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_65 = `RANDOM;	// Reg.scala:34:16
        _T_39 = {_RANDOM_65[2:0], _RANDOM_64, _RANDOM_63[31:3]};	// Reg.scala:34:16
      `endif
    end // initial
  `endif
      wire _T_53 = state == 2'h1;	// ICache.scala:75:52, :154:27
      wire _s1_miss = _T_19 & ~_T_41;	// ICache.scala:79:{27,30}
      wire [1:0] _T_54 = _T_18 & _s1_miss ? 2'h1 : state;	// ICache.scala:75:52, :154:27, :165:30
  always @(posedge clock) begin	// ICache.scala:67:18
    if (_s1_miss & _T_18)	// ICache.scala:86:17, :87:17
      refill_addr <= io_s1_paddr;	// ICache.scala:87:17
    s1_dout_valid <= _T_21;	// Reg.scala:14:44
    if (s1_dout_valid) begin	// Reg.scala:35:23
      _T_23_29 <= _T_28;	// Reg.scala:35:23
      _T_24 <= _T_32;	// Reg.scala:35:23
      _T_25_36 <= _T_35;	// Reg.scala:35:23
      _T_26 <= _T_39;	// Reg.scala:35:23
      _T_27_42 <= mem_RW0_rdata_20;	// ICache.scala:132:28, Reg.scala:35:23
      _T_28_43 <= mem_RW0_rdata_19;	// ICache.scala:132:28, Reg.scala:35:23
      _T_29 <= mem_RW0_rdata_18;	// ICache.scala:132:28, Reg.scala:35:23
      _T_30_44 <= mem_RW0_rdata;	// ICache.scala:132:28, Reg.scala:35:23
    end
    if (io_resp_ready) begin	// Reg.scala:35:23
      _T_32_46 <= _T_30;	// Reg.scala:35:23
      _T_33_47 <= _T_33;	// Reg.scala:35:23
      _T_34_48 <= _T_37;	// Reg.scala:35:23
      _T_35_49 <= _T_40;	// Reg.scala:35:23
      _T_36 <= s1_dout_valid ? mem_RW0_rdata_20 : _T_27_42;	// ICache.scala:132:28, Reg.scala:35:23
      _T_37_50 <= s1_dout_valid ? mem_RW0_rdata_19 : _T_28_43;	// ICache.scala:132:28, Reg.scala:35:23
      _T_38_51 <= s1_dout_valid ? mem_RW0_rdata_18 : _T_29;	// ICache.scala:132:28, Reg.scala:35:23
      _T_39_52 <= s1_dout_valid ? mem_RW0_rdata : _T_30_44;	// ICache.scala:132:28, Reg.scala:35:23
    end
    invalidated <= ~_T_18 & (io_invalidate | invalidated);	// ICache.scala:105:24, :110:17, :166:19
    if (reset) begin	// ICache.scala:67:18
      state <= 2'h0;	// ICache.scala:67:18
      s1_valid <= 1'h0;	// ICache.scala:69:15, :74:21
      _T_21_24 <= 5'h0;	// Edges.scala:199:14, Reg.scala:26:44
      _T_22_26 <= 16'h1;	// LFSR.scala:22:19
      vb_array <= 256'h0;	// ICache.scala:104:21
      _T_31_45 <= 1'h0;	// ICache.scala:69:15, Reg.scala:26:44
    end
    else begin	// ICache.scala:67:18
      s1_valid <= _T_21 | _T_20;	// ICache.scala:84:{12,24}
      _T_21_24 <= io_mem_0_d_valid ? (_T_21_24 == 5'h0 ? _T_23 : _T_25) : _T_21_24;	// Edges.scala:199:14, :208:28, :209:25, :214:{15,21}
      _T_22_26 <= _s1_miss ? {_T_22_26[0] ^ _T_22_26[2] ^ _T_22_26[3] ^ _T_22_26[5], _T_22_26[15:1]} :
                                                _T_22_26;	// Cat.scala:30:58, LFSR.scala:23:{29,40,48,56,59,64,73}
      vb_array <= io_invalidate ? 256'h0 : {256{_refill_done & ~invalidated}} & 256'h1 << {248'h0, _repl_way,
                                                _refill_idx} | vb_array;	// ICache.scala:104:21, :105:{21,24}, :106:{14,32}, :109:14, :114:51
      _T_31_45 <= io_resp_ready ? _T_19 & _T_41 : _T_31_45;	// ICache.scala:78:26, Reg.scala:43:23
      state <= &state & _refill_done ? 2'h0 : state == 2'h2 & io_mem_0_d_valid ? 2'h3 : _T_53 ?
                                                (io_s2_kill ? 2'h0 : io_mem_0_a_ready ? 2'h2 : _T_54) : _T_54;	// Conditional.scala:29:28, ICache.scala:67:18, :75:52, :101:84, :169:37, :170:33, :173:37, :176:34
    end
  end // always @(posedge)
  tag_array_0_ext tag_array_0 (	// ICache.scala:97:25
    .RW0_addr  (_refill_done ? _refill_idx : _T_13),	// ICache.scala:97:25, :98:42
    .RW0_en    (_T_17 & _T_11),	// ICache.scala:97:25, :101:84
    .RW0_clk   (clock),
    .RW0_wmode (_refill_done),
    .RW0_wdata (_refill_tag),
    .RW0_rdata (tag_array_0_RW0_rdata)
  );
  tag_array_0_ext tag_array_1 (	// ICache.scala:97:25
    .RW0_addr  (_refill_done ? _refill_idx : _T_13),	// ICache.scala:97:25, :98:42
    .RW0_en    (_T_16 & _T_10),	// ICache.scala:97:25, :101:84
    .RW0_clk   (clock),
    .RW0_wmode (_refill_done),
    .RW0_wdata (_refill_tag),
    .RW0_rdata (tag_array_1_RW0_rdata)
  );
  tag_array_0_ext tag_array_2 (	// ICache.scala:97:25
    .RW0_addr  (_refill_done ? _refill_idx : _T_13),	// ICache.scala:97:25, :98:42
    .RW0_en    (_T_15 & _T_9),	// ICache.scala:97:25, :101:84
    .RW0_clk   (clock),
    .RW0_wmode (_refill_done),
    .RW0_wdata (_refill_tag),
    .RW0_rdata (tag_array_2_RW0_rdata)
  );
  tag_array_0_ext tag_array_3 (	// ICache.scala:97:25
    .RW0_addr  (_refill_done ? _refill_idx : _T_13),	// ICache.scala:97:25, :98:42
    .RW0_en    (_T_14 & &_repl_way),	// ICache.scala:97:25, :101:84
    .RW0_clk   (clock),
    .RW0_wmode (_refill_done),
    .RW0_wdata (_refill_tag),
    .RW0_rdata (tag_array_3_RW0_rdata)
  );
  _ext mem (	// ICache.scala:132:28
    .RW0_addr  (_T_8 ? _T_7 : _T_6),	// ICache.scala:132:28, :133:30, :136:63, :138:28
    .RW0_en    (_T_5 | _T_8),	// ICache.scala:132:28, :133:30, :139:50
    .RW0_clk   (clock),
    .RW0_wmode (_T_8),	// ICache.scala:133:30
    .RW0_wdata (io_mem_0_d_bits_data),
    .RW0_rdata (mem_RW0_rdata_20)
  );
  _ext mem_66 (	// ICache.scala:132:28
    .RW0_addr  (_T_4 ? _T_7 : _T_6),	// ICache.scala:132:28, :133:30, :136:63, :138:28
    .RW0_en    (_T_3 | _T_4),	// ICache.scala:132:28, :133:30, :139:50
    .RW0_clk   (clock),
    .RW0_wmode (_T_4),	// ICache.scala:133:30
    .RW0_wdata (io_mem_0_d_bits_data),
    .RW0_rdata (mem_RW0_rdata_19)
  );
  _ext mem_67 (	// ICache.scala:132:28
    .RW0_addr  (_T_2 ? _T_7 : _T_6),	// ICache.scala:132:28, :133:30, :136:63, :138:28
    .RW0_en    (_T_1 | _T_2),	// ICache.scala:132:28, :133:30, :139:50
    .RW0_clk   (clock),
    .RW0_wmode (_T_2),	// ICache.scala:133:30
    .RW0_wdata (io_mem_0_d_bits_data),
    .RW0_rdata (mem_RW0_rdata_18)
  );
  _ext mem_68 (	// ICache.scala:132:28
    .RW0_addr  (_T_0 ? _T_7 : _T_6),	// ICache.scala:132:28, :133:30, :136:63, :138:28
    .RW0_en    (_T | _T_0),	// ICache.scala:132:28, :133:30, :139:50
    .RW0_clk   (clock),
    .RW0_wmode (_T_0),	// ICache.scala:133:30
    .RW0_wdata (io_mem_0_d_bits_data),
    .RW0_rdata (mem_RW0_rdata)
  );
  assign io_resp_valid = _T_31_45;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Reg.scala:43:23
  assign io_resp_bits_data = 16'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, :11:8
  assign io_resp_bits_datablock = (_T_32_46 ? _T_36 : 64'h0) | (_T_33_47 ? _T_37_50 : 64'h0) | (_T_34_48 ? _T_38_51 : 64'h0)
                | (_T_35_49 ? _T_39_52 : 64'h0);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Mux.scala:19:72, Reg.scala:35:23
  assign io_mem_0_a_valid = _T_53 & ~io_s2_kill;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:154:{41,44}
  assign io_mem_0_a_bits_opcode = 3'h4;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Edges.scala:343:15
  assign io_mem_0_a_bits_param = 3'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Edges.scala:342:17
  assign io_mem_0_a_bits_size = 4'h6;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Edges.scala:342:17
  assign io_mem_0_a_bits_source = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_a_bits_address = {refill_addr[31:6], 6'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:87:17, :157:{46,63}
  assign io_mem_0_a_bits_mask = 8'hFF;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, package.scala:19:64
  assign io_mem_0_a_bits_data = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Mux.scala:19:72
  assign io_mem_0_b_ready = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_c_valid = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_c_bits_opcode = 3'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Edges.scala:342:17
  assign io_mem_0_c_bits_param = 3'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Edges.scala:342:17
  assign io_mem_0_c_bits_size = 4'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, :11:8
  assign io_mem_0_c_bits_source = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_c_bits_address = 32'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, :11:8
  assign io_mem_0_c_bits_data = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, Mux.scala:19:72
  assign io_mem_0_c_bits_error = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_d_ready = 1'h1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10
  assign io_mem_0_e_valid = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, ICache.scala:69:15
  assign io_mem_0_e_bits_sink = 4'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/ICache.fir:6:10, :11:8
endmodule

