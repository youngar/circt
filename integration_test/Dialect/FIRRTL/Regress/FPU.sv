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

module regfile_ext(	// FPU.scala:547:20
  input  [4:0]  R0_addr,
  input         R0_en, R0_clk,
  input  [4:0]  R1_addr,
  input         R1_en, R1_clk,
  input  [4:0]  R2_addr,
  input         R2_en, R2_clk,
  input  [4:0]  W0_addr,
  input         W0_en, W0_clk,
  input  [64:0] W0_data,
  input  [4:0]  W1_addr,
  input         W1_en, W1_clk,
  input  [64:0] W1_data,
  output [64:0] R0_data, R1_data, R2_data);

  reg [64:0] Memory[0:31];

  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
    if (W1_en)
      Memory[W1_addr] <= W1_data;
  end // always @(posedge)
  assign R0_data = R0_en ? Memory[R0_addr] : 65'bx;	// FPU.scala:547:20
  assign R1_data = R1_en ? Memory[R1_addr] : 65'bx;	// FPU.scala:547:20
  assign R2_data = R2_en ? Memory[R2_addr] : 65'bx;	// FPU.scala:547:20
endmodule

module FPU(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
  input         clock, reset,
  input  [31:0] io_inst,
  input  [63:0] io_fromint_data,
  input  [2:0]  io_fcsr_rm,
  input         io_dmem_resp_val,
  input  [2:0]  io_dmem_resp_type,
  input  [4:0]  io_dmem_resp_tag,
  input  [63:0] io_dmem_resp_data,
  input         io_valid, io_killx, io_killm, io_cp_req_valid,
  input  [4:0]  io_cp_req_bits_cmd,
  input         io_cp_req_bits_ldst, io_cp_req_bits_wen, io_cp_req_bits_ren1,
  input         io_cp_req_bits_ren2, io_cp_req_bits_ren3, io_cp_req_bits_swap12,
  input         io_cp_req_bits_swap23, io_cp_req_bits_single, io_cp_req_bits_fromint,
  input         io_cp_req_bits_toint, io_cp_req_bits_fastpipe, io_cp_req_bits_fma,
  input         io_cp_req_bits_div, io_cp_req_bits_sqrt, io_cp_req_bits_wflags,
  input  [2:0]  io_cp_req_bits_rm,
  input  [1:0]  io_cp_req_bits_typ,
  input  [64:0] io_cp_req_bits_in1, io_cp_req_bits_in2, io_cp_req_bits_in3,
  input         io_cp_resp_ready,
  output        io_fcsr_flags_valid,
  output [4:0]  io_fcsr_flags_bits,
  output [63:0] io_store_data, io_toint_data,
  output        io_fcsr_rdy, io_nack_mem, io_illegal_rm,
  output [4:0]  io_dec_cmd,
  output        io_dec_ldst, io_dec_wen, io_dec_ren1, io_dec_ren2, io_dec_ren3,
  output        io_dec_swap12, io_dec_swap23, io_dec_single, io_dec_fromint,
  output        io_dec_toint, io_dec_fastpipe, io_dec_fma, io_dec_div, io_dec_sqrt,
  output        io_dec_wflags, io_sboard_set, io_sboard_clr,
  output [4:0]  io_sboard_clra,
  output        io_cp_req_ready, io_cp_resp_valid,
  output [64:0] io_cp_resp_bits_data,
  output [4:0]  io_cp_resp_bits_exc);

  wire [4:0]  _T;	// FPU.scala:750:43
  wire [64:0] _T_0;	// FPU.scala:749:25
  wire        _T_1;	// FPU.scala:723:27
  wire [32:0] RecFNToRecFN_io_out;	// FPU.scala:746:34
  wire [4:0]  RecFNToRecFN_io_exceptionFlags;	// FPU.scala:746:34
  wire        DivSqrtRecF64_io_inReady_div;	// FPU.scala:722:25
  wire        DivSqrtRecF64_io_inReady_sqrt;	// FPU.scala:722:25
  wire        DivSqrtRecF64_io_outValid_div;	// FPU.scala:722:25
  wire        DivSqrtRecF64_io_outValid_sqrt;	// FPU.scala:722:25
  wire [64:0] DivSqrtRecF64_io_out;	// FPU.scala:722:25
  wire [4:0]  DivSqrtRecF64_io_exceptionFlags;	// FPU.scala:722:25
  wire        FPUFMAPipe_io_out_valid;	// FPU.scala:624:28
  wire [64:0] FPUFMAPipe_io_out_bits_data;	// FPU.scala:624:28
  wire [4:0]  FPUFMAPipe_io_out_bits_exc;	// FPU.scala:624:28
  wire        fpmu_io_out_valid;	// FPU.scala:603:20
  wire [64:0] fpmu_io_out_bits_data;	// FPU.scala:603:20
  wire [4:0]  fpmu_io_out_bits_exc;	// FPU.scala:603:20
  wire        ifpu_io_out_valid;	// FPU.scala:598:20
  wire [64:0] ifpu_io_out_bits_data;	// FPU.scala:598:20
  wire [4:0]  ifpu_io_out_bits_exc;	// FPU.scala:598:20
  wire [4:0]  fpiu_io_as_double_cmd;	// FPU.scala:588:20
  wire        fpiu_io_as_double_ldst;	// FPU.scala:588:20
  wire        fpiu_io_as_double_wen;	// FPU.scala:588:20
  wire        fpiu_io_as_double_ren1;	// FPU.scala:588:20
  wire        fpiu_io_as_double_ren2;	// FPU.scala:588:20
  wire        fpiu_io_as_double_ren3;	// FPU.scala:588:20
  wire        fpiu_io_as_double_swap12;	// FPU.scala:588:20
  wire        fpiu_io_as_double_swap23;	// FPU.scala:588:20
  wire        fpiu_io_as_double_single;	// FPU.scala:588:20
  wire        fpiu_io_as_double_fromint;	// FPU.scala:588:20
  wire        fpiu_io_as_double_toint;	// FPU.scala:588:20
  wire        fpiu_io_as_double_fastpipe;	// FPU.scala:588:20
  wire        fpiu_io_as_double_fma;	// FPU.scala:588:20
  wire        fpiu_io_as_double_div;	// FPU.scala:588:20
  wire        fpiu_io_as_double_sqrt;	// FPU.scala:588:20
  wire        fpiu_io_as_double_wflags;	// FPU.scala:588:20
  wire [2:0]  fpiu_io_as_double_rm;	// FPU.scala:588:20
  wire [1:0]  fpiu_io_as_double_typ;	// FPU.scala:588:20
  wire [64:0] fpiu_io_as_double_in1;	// FPU.scala:588:20
  wire [64:0] fpiu_io_as_double_in2;	// FPU.scala:588:20
  wire [64:0] fpiu_io_as_double_in3;	// FPU.scala:588:20
  wire        fpiu_io_out_valid;	// FPU.scala:588:20
  wire        fpiu_io_out_bits_lt;	// FPU.scala:588:20
  wire [63:0] fpiu_io_out_bits_toint;	// FPU.scala:588:20
  wire [4:0]  fpiu_io_out_bits_exc;	// FPU.scala:588:20
  wire        sfma_io_out_valid;	// FPU.scala:584:20
  wire [64:0] sfma_io_out_bits_data;	// FPU.scala:584:20
  wire [4:0]  sfma_io_out_bits_exc;	// FPU.scala:584:20
  wire [64:0] regfile_R0_data;	// FPU.scala:547:20
  wire [64:0] regfile_R1_data;	// FPU.scala:547:20
  wire [64:0] regfile_R2_data;	// FPU.scala:547:20
  wire [4:0]  fp_decoder_io_sigs_cmd;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_ldst;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_wen;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_ren1;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_ren2;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_ren3;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_swap12;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_swap23;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_single;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_fromint;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_toint;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_fastpipe;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_fma;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_div;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_sqrt;	// FPU.scala:520:26
  wire        fp_decoder_io_sigs_wflags;	// FPU.scala:520:26
  reg         ex_reg_valid;	// FPU.scala:509:25
  reg  [31:0] ex_reg_inst;	// Reg.scala:34:16
  reg         mem_reg_valid;	// FPU.scala:513:26
  reg  [31:0] mem_reg_inst;	// Reg.scala:34:16
  reg         mem_cp_valid;	// FPU.scala:515:25
  reg         wb_reg_valid;	// FPU.scala:517:25
  reg         wb_cp_valid;	// FPU.scala:518:24
  reg  [4:0]  _T_2;	// Reg.scala:34:16
  reg         _T_3;	// Reg.scala:34:16
  reg         _T_4;	// Reg.scala:34:16
  reg         _T_5;	// Reg.scala:34:16
  reg         _T_6;	// Reg.scala:34:16
  reg         _T_7;	// Reg.scala:34:16
  reg         _T_8;	// Reg.scala:34:16
  reg         _T_9;	// Reg.scala:34:16
  reg         _T_10;	// Reg.scala:34:16
  reg         _T_11;	// Reg.scala:34:16
  reg         _T_12;	// Reg.scala:34:16
  reg         _T_13;	// Reg.scala:34:16
  reg         _T_14;	// Reg.scala:34:16
  reg         _T_15;	// Reg.scala:34:16
  reg         _T_16;	// Reg.scala:34:16
  reg         _T_17;	// Reg.scala:34:16
  reg  [4:0]  mem_ctrl_cmd;	// Reg.scala:34:16
  reg         mem_ctrl_ldst;	// Reg.scala:34:16
  reg         mem_ctrl_wen;	// Reg.scala:34:16
  reg         mem_ctrl_ren1;	// Reg.scala:34:16
  reg         mem_ctrl_ren2;	// Reg.scala:34:16
  reg         mem_ctrl_ren3;	// Reg.scala:34:16
  reg         mem_ctrl_swap12;	// Reg.scala:34:16
  reg         mem_ctrl_swap23;	// Reg.scala:34:16
  reg         mem_ctrl_single;	// Reg.scala:34:16
  reg         mem_ctrl_fromint;	// Reg.scala:34:16
  reg         mem_ctrl_toint;	// Reg.scala:34:16
  reg         mem_ctrl_fastpipe;	// Reg.scala:34:16
  reg         mem_ctrl_fma;	// Reg.scala:34:16
  reg         mem_ctrl_div;	// Reg.scala:34:16
  reg         mem_ctrl_sqrt;	// Reg.scala:34:16
  reg         mem_ctrl_wflags;	// Reg.scala:34:16
  reg  [4:0]  wb_ctrl_cmd;	// Reg.scala:34:16
  reg         wb_ctrl_ldst;	// Reg.scala:34:16
  reg         wb_ctrl_wen;	// Reg.scala:34:16
  reg         wb_ctrl_ren1;	// Reg.scala:34:16
  reg         wb_ctrl_ren2;	// Reg.scala:34:16
  reg         wb_ctrl_ren3;	// Reg.scala:34:16
  reg         wb_ctrl_swap12;	// Reg.scala:34:16
  reg         wb_ctrl_swap23;	// Reg.scala:34:16
  reg         wb_ctrl_single;	// Reg.scala:34:16
  reg         wb_ctrl_fromint;	// Reg.scala:34:16
  reg         wb_ctrl_toint;	// Reg.scala:34:16
  reg         wb_ctrl_fastpipe;	// Reg.scala:34:16
  reg         wb_ctrl_fma;	// Reg.scala:34:16
  reg         wb_ctrl_div;	// Reg.scala:34:16
  reg         wb_ctrl_sqrt;	// Reg.scala:34:16
  reg         wb_ctrl_wflags;	// Reg.scala:34:16
  reg         load_wb;	// FPU.scala:534:20
  reg         load_wb_single;	// Reg.scala:34:16
  reg  [63:0] load_wb_data;	// Reg.scala:34:16
  reg  [4:0]  load_wb_tag;	// Reg.scala:34:16
  reg  [4:0]  ex_ra1;	// FPU.scala:554:53
  reg  [4:0]  ex_ra2;	// FPU.scala:554:53
  reg  [4:0]  ex_ra3;	// FPU.scala:554:53
  reg         divSqrt_wen;	// FPU.scala:608:24
  reg  [4:0]  divSqrt_waddr;	// FPU.scala:610:26
  reg         divSqrt_single;	// FPU.scala:611:27
  reg         divSqrt_in_flight;	// FPU.scala:614:30
  reg         divSqrt_killed;	// FPU.scala:615:27
  reg  [2:0]  wen;	// FPU.scala:645:16
  reg  [4:0]  wbInfo_0_rd;	// FPU.scala:646:19
  reg         wbInfo_0_single;	// FPU.scala:646:19
  reg         wbInfo_0_cp;	// FPU.scala:646:19
  reg  [1:0]  wbInfo_0_pipeid;	// FPU.scala:646:19
  reg  [4:0]  wbInfo_1_rd;	// FPU.scala:646:19
  reg         wbInfo_1_single;	// FPU.scala:646:19
  reg         wbInfo_1_cp;	// FPU.scala:646:19
  reg  [1:0]  wbInfo_1_pipeid;	// FPU.scala:646:19
  reg  [4:0]  wbInfo_2_rd;	// FPU.scala:646:19
  reg         wbInfo_2_single;	// FPU.scala:646:19
  reg         wbInfo_2_cp;	// FPU.scala:646:19
  reg  [1:0]  wbInfo_2_pipeid;	// FPU.scala:646:19
  reg         write_port_busy;	// Reg.scala:34:16
  reg  [4:0]  wb_toint_exc;	// Reg.scala:34:16
  reg         _T_18_53;	// FPU.scala:708:55
  reg  [1:0]  _T_19_54;	// FPU.scala:718:25
  reg  [4:0]  _T_20_55;	// FPU.scala:719:35
  reg  [64:0] _T_21_56;	// FPU.scala:720:35

  wire _req_valid = ex_reg_valid | io_cp_req_valid;	// FPU.scala:510:32
  wire _ex_cp_valid = ~ex_reg_valid & io_cp_req_valid;	// Decoupled.scala:30:37, FPU.scala:510:32, :693:22
  wire [4:0] _ex_ctrl_cmd = _ex_cp_valid ? io_cp_req_bits_cmd : _T_2;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_single = _ex_cp_valid ? io_cp_req_bits_single : _T_10;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_fromint = _ex_cp_valid ? io_cp_req_bits_fromint : _T_11;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_toint = _ex_cp_valid ? io_cp_req_bits_toint : _T_12;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_fastpipe = _ex_cp_valid ? io_cp_req_bits_fastpipe : _T_13;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_fma = _ex_cp_valid ? io_cp_req_bits_fma : _T_14;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_div = _ex_cp_valid ? io_cp_req_bits_div : _T_15;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_sqrt = _ex_cp_valid ? io_cp_req_bits_sqrt : _T_16;	// FPU.scala:529:20, Reg.scala:35:23
  wire _ex_ctrl_wflags = _ex_cp_valid ? io_cp_req_bits_wflags : _T_17;	// FPU.scala:529:20, Reg.scala:35:23
  wire [7:0] _load_wb_data_30to23 = load_wb_data[30:23];	// Reg.scala:35:23, recFNFromFN.scala:48:23
  wire [22:0] _load_wb_data_22to0 = load_wb_data[22:0];	// Reg.scala:35:23, recFNFromFN.scala:49:25
  wire _T_18 = _load_wb_data_30to23 == 8'h0;	// recFNFromFN.scala:51:34
  wire [15:0] _load_wb_data_22to7 = load_wb_data[22:7];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [7:0] _load_wb_data_22to15 = load_wb_data[22:15];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_22to19 = load_wb_data[22:19];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_14to11 = load_wb_data[14:11];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [6:0] _load_wb_data_6to0 = load_wb_data[6:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_6to3 = load_wb_data[6:3];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _T_19 = |_load_wb_data_22to7 ? {|_load_wb_data_22to15, |_load_wb_data_22to15 ?
                {|_load_wb_data_22to19, |_load_wb_data_22to19 ? (load_wb_data[22] ? 2'h3 : load_wb_data[21]
                ? 2'h2 : {1'h0, load_wb_data[20]}) : load_wb_data[18] ? 2'h3 : load_wb_data[17] ? 2'h2 :
                {1'h0, load_wb_data[16]}} : {|_load_wb_data_14to11, |_load_wb_data_14to11 ?
                (load_wb_data[14] ? 2'h3 : load_wb_data[13] ? 2'h2 : {1'h0, load_wb_data[12]}) :
                load_wb_data[10] ? 2'h3 : load_wb_data[9] ? 2'h2 : {1'h0, load_wb_data[8]}}} :
                {|_load_wb_data_6to0, |_load_wb_data_6to0 ? {|_load_wb_data_6to3, |_load_wb_data_6to3 ?
                (load_wb_data[6] ? 2'h3 : load_wb_data[5] ? 2'h2 : {1'h0, load_wb_data[4]}) :
                load_wb_data[2] ? 2'h3 : load_wb_data[1] ? 2'h2 : {1'h0, load_wb_data[0]}} : 3'h0};	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, FPU.scala:509:25, Reg.scala:35:23
  wire [53:0] _T_20 = {31'h0, _load_wb_data_22to0} << ~{|_load_wb_data_22to7, _T_19};	// Cat.scala:30:58, CircuitMath.scala:37:22, recFNFromFN.scala:56:13, :58:25
  wire [8:0] _T_21 = (_T_18 ? {4'hF, |_load_wb_data_22to7, _T_19} : {1'h0, _load_wb_data_30to23}) + {7'h20,
                _T_18 ? 2'h2 : 2'h1};	// CircuitMath.scala:32:10, :37:22, FPU.scala:509:25, recFNFromFN.scala:61:16, :62:27, :64:{15,47}
  wire [10:0] _load_wb_data_62to52 = load_wb_data[62:52];	// Reg.scala:35:23, recFNFromFN.scala:48:23
  wire [51:0] _load_wb_data_51to0 = load_wb_data[51:0];	// Reg.scala:35:23, recFNFromFN.scala:49:25
  wire _T_22 = _load_wb_data_62to52 == 11'h0;	// recFNFromFN.scala:51:34
  wire [31:0] _load_wb_data_51to20 = load_wb_data[51:20];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [15:0] _load_wb_data_51to36 = load_wb_data[51:36];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [7:0] _load_wb_data_51to44 = load_wb_data[51:44];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_51to48 = load_wb_data[51:48];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_43to40 = load_wb_data[43:40];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [7:0] _load_wb_data_35to28 = load_wb_data[35:28];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_35to32 = load_wb_data[35:32];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_27to24 = load_wb_data[27:24];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [15:0] _load_wb_data_19to4 = load_wb_data[19:4];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [7:0] _load_wb_data_19to12 = load_wb_data[19:12];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_19to16 = load_wb_data[19:16];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_11to8 = load_wb_data[11:8];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_3to0 = load_wb_data[3:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [3:0] _load_wb_data_3to0_23 = load_wb_data[3:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
  wire [4:0] _T_24 = |_load_wb_data_51to20 ? {|_load_wb_data_51to36, |_load_wb_data_51to36 ?
                {|_load_wb_data_51to44, |_load_wb_data_51to44 ? {|_load_wb_data_51to48,
                |_load_wb_data_51to48 ? (load_wb_data[51] ? 2'h3 : load_wb_data[50] ? 2'h2 : {1'h0,
                load_wb_data[49]}) : load_wb_data[47] ? 2'h3 : load_wb_data[46] ? 2'h2 : {1'h0,
                load_wb_data[45]}} : {|_load_wb_data_43to40, |_load_wb_data_43to40 ? (load_wb_data[43] ?
                2'h3 : load_wb_data[42] ? 2'h2 : {1'h0, load_wb_data[41]}) : load_wb_data[39] ? 2'h3 :
                load_wb_data[38] ? 2'h2 : {1'h0, load_wb_data[37]}}} : {|_load_wb_data_35to28,
                |_load_wb_data_35to28 ? {|_load_wb_data_35to32, |_load_wb_data_35to32 ? (load_wb_data[35] ?
                2'h3 : load_wb_data[34] ? 2'h2 : {1'h0, load_wb_data[33]}) : load_wb_data[31] ? 2'h3 :
                load_wb_data[30] ? 2'h2 : {1'h0, load_wb_data[29]}} : {|_load_wb_data_27to24,
                |_load_wb_data_27to24 ? (load_wb_data[27] ? 2'h3 : load_wb_data[26] ? 2'h2 : {1'h0,
                load_wb_data[25]}) : load_wb_data[23] ? 2'h3 : load_wb_data[22] ? 2'h2 : {1'h0,
                load_wb_data[21]}}}} : {|_load_wb_data_19to4, |_load_wb_data_19to4 ?
                {|_load_wb_data_19to12, |_load_wb_data_19to12 ? {|_load_wb_data_19to16,
                |_load_wb_data_19to16 ? (load_wb_data[19] ? 2'h3 : load_wb_data[18] ? 2'h2 : {1'h0,
                load_wb_data[17]}) : load_wb_data[15] ? 2'h3 : load_wb_data[14] ? 2'h2 : {1'h0,
                load_wb_data[13]}} : {|_load_wb_data_11to8, |_load_wb_data_11to8 ? (load_wb_data[11] ? 2'h3
                : load_wb_data[10] ? 2'h2 : {1'h0, load_wb_data[9]}) : load_wb_data[7] ? 2'h3 :
                load_wb_data[6] ? 2'h2 : {1'h0, load_wb_data[5]}}} : {|_load_wb_data_3to0,
                |_load_wb_data_3to0 ? {|_load_wb_data_3to0_23, |_load_wb_data_3to0_23 ? (load_wb_data[3] ?
                2'h3 : load_wb_data[2] ? 2'h2 : {1'h0, load_wb_data[1]}) : 2'h0} : 3'h0}};	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, FPU.scala:509:25, :631:23, Reg.scala:35:23
  wire [114:0] _T_25 = {63'h0, _load_wb_data_51to0} << ~{|_load_wb_data_51to20, _T_24};	// Cat.scala:30:58, CircuitMath.scala:37:22, recFNFromFN.scala:56:13, :58:25
  wire [11:0] _T_26 = (_T_22 ? {6'h3F, |_load_wb_data_51to20, _T_24} : {1'h0, _load_wb_data_62to52}) + {10'h100,
                _T_22 ? 2'h2 : 2'h1};	// CircuitMath.scala:32:10, :37:22, FPU.scala:509:25, recFNFromFN.scala:56:13, :61:16, :62:27, :64:{15,47}
  wire [2:0] _ex_reg_inst_14to12 = ex_reg_inst[14:12];	// FPU.scala:567:30, Reg.scala:35:23
  wire [4:0] _T_27 = _ex_cp_valid ? io_cp_req_bits_cmd : _T_2;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_28 = _ex_cp_valid ? io_cp_req_bits_ldst : _T_3;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_29 = _ex_cp_valid ? io_cp_req_bits_wen : _T_4;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_30 = _ex_cp_valid ? io_cp_req_bits_ren1 : _T_5;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_31 = _ex_cp_valid ? io_cp_req_bits_ren2 : _T_6;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_32 = _ex_cp_valid ? io_cp_req_bits_ren3 : _T_7;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_33 = _ex_cp_valid ? io_cp_req_bits_swap12 : _T_8;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_34 = _ex_cp_valid ? io_cp_req_bits_swap23 : _T_9;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_35 = _ex_cp_valid ? io_cp_req_bits_single : _T_10;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_36 = _ex_cp_valid ? io_cp_req_bits_fromint : _T_11;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_37 = _ex_cp_valid ? io_cp_req_bits_toint : _T_12;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_38 = _ex_cp_valid ? io_cp_req_bits_fastpipe : _T_13;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_39 = _ex_cp_valid ? io_cp_req_bits_fma : _T_14;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_40 = _ex_cp_valid ? io_cp_req_bits_div : _T_15;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_41 = _ex_cp_valid ? io_cp_req_bits_sqrt : _T_16;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire _T_42 = _ex_cp_valid ? io_cp_req_bits_wflags : _T_17;	// FPU.scala:529:20, :577:9, Reg.scala:35:23
  wire [2:0] _T_43 = _ex_cp_valid ? io_cp_req_bits_rm : &_ex_reg_inst_14to12 ? io_fcsr_rm : _ex_reg_inst_14to12;	// FPU.scala:567:{18,38}, :577:9
  wire [1:0] _T_44 = _ex_cp_valid ? io_cp_req_bits_typ : ex_reg_inst[21:20];	// FPU.scala:575:25, :577:9, Reg.scala:35:23
  wire [64:0] _T_45 = _ex_cp_valid ? io_cp_req_bits_in1 : regfile_R0_data;	// FPU.scala:547:20, :577:9
  wire [64:0] _T_46 = _ex_cp_valid ? (io_cp_req_bits_swap23 ? io_cp_req_bits_in3 : io_cp_req_bits_in2) :
                regfile_R1_data;	// FPU.scala:547:20, :579:15
  wire [64:0] _T_47 = _ex_cp_valid ? (io_cp_req_bits_swap23 ? io_cp_req_bits_in2 : io_cp_req_bits_in3) :
                regfile_R2_data;	// FPU.scala:547:20, :580:15
  wire _T_48 = _req_valid & _ex_ctrl_fma;	// FPU.scala:585:33
  wire _T_49 = fpiu_io_out_valid & mem_cp_valid & mem_ctrl_toint;	// FPU.scala:516:44, :588:20, :593:42, Reg.scala:35:23
  wire [4:0] _waddr = divSqrt_wen ? divSqrt_waddr : wbInfo_0_rd;	// FPU.scala:651:33, :668:18
  wire _wbInfo_0_pipeid_0 = wbInfo_0_pipeid[0];	// FPU.scala:651:33, Package.scala:18:26
  wire _wbInfo_0_pipeid_1 = wbInfo_0_pipeid[1];	// FPU.scala:651:33, Package.scala:19:17
  wire [64:0] _wdata0 = divSqrt_wen ? _T_0 : _wbInfo_0_pipeid_1 ? (_wbInfo_0_pipeid_0 ? FPUFMAPipe_io_out_bits_data
                : sfma_io_out_bits_data) : _wbInfo_0_pipeid_0 ? ifpu_io_out_bits_data :
                fpmu_io_out_bits_data;	// FPU.scala:584:20, :598:20, :603:20, :624:28, :668:18, :669:19, :749:25, Package.scala:19:12
  wire [64:0] _wdata = (divSqrt_wen ? divSqrt_single : wbInfo_0_single) ? {32'h70020000, _wdata0[32:0]} : _wdata0;	// FPU.scala:543:33, :651:33, :668:18, :670:20, :673:{19,36,44}
  wire _wen_0 = wen[0];	// FPU.scala:648:101, :676:30
  wire _T_50 = wbInfo_0_cp & _wen_0;	// FPU.scala:651:33, :689:22
  wire _wb_toint_valid = wb_reg_valid & wb_ctrl_toint;	// FPU.scala:695:37, Reg.scala:35:23
  wire _T_51 = mem_reg_valid & (mem_ctrl_div | mem_ctrl_sqrt);	// FPU.scala:517:45, :703:{34,51}, Reg.scala:35:23
  wire _T_52 = _T_51 & (~_T_1 | |wen) | write_port_busy | divSqrt_in_flight;	// FPU.scala:648:101, :703:{69,73,90,97}, :704:131, :705:48, :723:27, Reg.scala:35:23
  `ifndef SYNTHESIS	// FPU.scala:509:25
    `ifdef RANDOMIZE_REG_INIT	// FPU.scala:509:25
      reg [31:0] _RANDOM;	// FPU.scala:509:25
      reg [31:0] _RANDOM_22;	// Reg.scala:34:16
      reg [31:0] _RANDOM_23;	// Reg.scala:34:16
      reg [31:0] _RANDOM_24;	// Reg.scala:34:16
      reg [31:0] _RANDOM_25;	// Reg.scala:34:16
      reg [31:0] _RANDOM_26;	// Reg.scala:34:16
      reg [31:0] _RANDOM_27;	// Reg.scala:34:16
      reg [31:0] _RANDOM_28;	// FPU.scala:645:16
      reg [31:0] _RANDOM_29;	// Reg.scala:34:16
      reg [31:0] _RANDOM_30;	// FPU.scala:720:35
      reg [31:0] _RANDOM_31;	// FPU.scala:720:35

    `endif
    initial begin	// FPU.scala:509:25
      `INIT_RANDOM_PROLOG_	// FPU.scala:509:25
      `ifdef RANDOMIZE_REG_INIT	// FPU.scala:509:25
        _RANDOM = `RANDOM;	// FPU.scala:509:25
        ex_reg_valid = _RANDOM[0];	// FPU.scala:509:25
        _RANDOM_22 = `RANDOM;	// Reg.scala:34:16
        ex_reg_inst = {_RANDOM_22[0], _RANDOM[31:1]};	// Reg.scala:34:16
        mem_reg_valid = _RANDOM_22[1];	// FPU.scala:513:26
        _RANDOM_23 = `RANDOM;	// Reg.scala:34:16
        mem_reg_inst = {_RANDOM_23[1:0], _RANDOM_22[31:2]};	// Reg.scala:34:16
        mem_cp_valid = _RANDOM_23[2];	// FPU.scala:515:25
        wb_reg_valid = _RANDOM_23[3];	// FPU.scala:517:25
        wb_cp_valid = _RANDOM_23[4];	// FPU.scala:518:24
        _T_2 = _RANDOM_23[9:5];	// Reg.scala:34:16
        _T_3 = _RANDOM_23[10];	// Reg.scala:34:16
        _T_4 = _RANDOM_23[11];	// Reg.scala:34:16
        _T_5 = _RANDOM_23[12];	// Reg.scala:34:16
        _T_6 = _RANDOM_23[13];	// Reg.scala:34:16
        _T_7 = _RANDOM_23[14];	// Reg.scala:34:16
        _T_8 = _RANDOM_23[15];	// Reg.scala:34:16
        _T_9 = _RANDOM_23[16];	// Reg.scala:34:16
        _T_10 = _RANDOM_23[17];	// Reg.scala:34:16
        _T_11 = _RANDOM_23[18];	// Reg.scala:34:16
        _T_12 = _RANDOM_23[19];	// Reg.scala:34:16
        _T_13 = _RANDOM_23[20];	// Reg.scala:34:16
        _T_14 = _RANDOM_23[21];	// Reg.scala:34:16
        _T_15 = _RANDOM_23[22];	// Reg.scala:34:16
        _T_16 = _RANDOM_23[23];	// Reg.scala:34:16
        _T_17 = _RANDOM_23[24];	// Reg.scala:34:16
        mem_ctrl_cmd = _RANDOM_23[29:25];	// Reg.scala:34:16
        mem_ctrl_ldst = _RANDOM_23[30];	// Reg.scala:34:16
        mem_ctrl_wen = _RANDOM_23[31];	// Reg.scala:34:16
        _RANDOM_24 = `RANDOM;	// Reg.scala:34:16
        mem_ctrl_ren1 = _RANDOM_24[0];	// Reg.scala:34:16
        mem_ctrl_ren2 = _RANDOM_24[1];	// Reg.scala:34:16
        mem_ctrl_ren3 = _RANDOM_24[2];	// Reg.scala:34:16
        mem_ctrl_swap12 = _RANDOM_24[3];	// Reg.scala:34:16
        mem_ctrl_swap23 = _RANDOM_24[4];	// Reg.scala:34:16
        mem_ctrl_single = _RANDOM_24[5];	// Reg.scala:34:16
        mem_ctrl_fromint = _RANDOM_24[6];	// Reg.scala:34:16
        mem_ctrl_toint = _RANDOM_24[7];	// Reg.scala:34:16
        mem_ctrl_fastpipe = _RANDOM_24[8];	// Reg.scala:34:16
        mem_ctrl_fma = _RANDOM_24[9];	// Reg.scala:34:16
        mem_ctrl_div = _RANDOM_24[10];	// Reg.scala:34:16
        mem_ctrl_sqrt = _RANDOM_24[11];	// Reg.scala:34:16
        mem_ctrl_wflags = _RANDOM_24[12];	// Reg.scala:34:16
        wb_ctrl_cmd = _RANDOM_24[17:13];	// Reg.scala:34:16
        wb_ctrl_ldst = _RANDOM_24[18];	// Reg.scala:34:16
        wb_ctrl_wen = _RANDOM_24[19];	// Reg.scala:34:16
        wb_ctrl_ren1 = _RANDOM_24[20];	// Reg.scala:34:16
        wb_ctrl_ren2 = _RANDOM_24[21];	// Reg.scala:34:16
        wb_ctrl_ren3 = _RANDOM_24[22];	// Reg.scala:34:16
        wb_ctrl_swap12 = _RANDOM_24[23];	// Reg.scala:34:16
        wb_ctrl_swap23 = _RANDOM_24[24];	// Reg.scala:34:16
        wb_ctrl_single = _RANDOM_24[25];	// Reg.scala:34:16
        wb_ctrl_fromint = _RANDOM_24[26];	// Reg.scala:34:16
        wb_ctrl_toint = _RANDOM_24[27];	// Reg.scala:34:16
        wb_ctrl_fastpipe = _RANDOM_24[28];	// Reg.scala:34:16
        wb_ctrl_fma = _RANDOM_24[29];	// Reg.scala:34:16
        wb_ctrl_div = _RANDOM_24[30];	// Reg.scala:34:16
        wb_ctrl_sqrt = _RANDOM_24[31];	// Reg.scala:34:16
        _RANDOM_25 = `RANDOM;	// Reg.scala:34:16
        wb_ctrl_wflags = _RANDOM_25[0];	// Reg.scala:34:16
        load_wb = _RANDOM_25[1];	// FPU.scala:534:20
        load_wb_single = _RANDOM_25[2];	// Reg.scala:34:16
        _RANDOM_26 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_27 = `RANDOM;	// Reg.scala:34:16
        load_wb_data = {_RANDOM_27[2:0], _RANDOM_26, _RANDOM_25[31:3]};	// Reg.scala:34:16
        load_wb_tag = _RANDOM_27[7:3];	// Reg.scala:34:16
        ex_ra1 = _RANDOM_27[12:8];	// FPU.scala:554:53
        ex_ra2 = _RANDOM_27[17:13];	// FPU.scala:554:53
        ex_ra3 = _RANDOM_27[22:18];	// FPU.scala:554:53
        divSqrt_wen = _RANDOM_27[23];	// FPU.scala:608:24
        divSqrt_waddr = _RANDOM_27[28:24];	// FPU.scala:610:26
        divSqrt_single = _RANDOM_27[29];	// FPU.scala:611:27
        divSqrt_in_flight = _RANDOM_27[30];	// FPU.scala:614:30
        divSqrt_killed = _RANDOM_27[31];	// FPU.scala:615:27
        _RANDOM_28 = `RANDOM;	// FPU.scala:645:16
        wen = _RANDOM_28[2:0];	// FPU.scala:645:16
        wbInfo_0_rd = _RANDOM_28[7:3];	// FPU.scala:646:19
        wbInfo_0_single = _RANDOM_28[8];	// FPU.scala:646:19
        wbInfo_0_cp = _RANDOM_28[9];	// FPU.scala:646:19
        wbInfo_0_pipeid = _RANDOM_28[11:10];	// FPU.scala:646:19
        wbInfo_1_rd = _RANDOM_28[16:12];	// FPU.scala:646:19
        wbInfo_1_single = _RANDOM_28[17];	// FPU.scala:646:19
        wbInfo_1_cp = _RANDOM_28[18];	// FPU.scala:646:19
        wbInfo_1_pipeid = _RANDOM_28[20:19];	// FPU.scala:646:19
        wbInfo_2_rd = _RANDOM_28[25:21];	// FPU.scala:646:19
        wbInfo_2_single = _RANDOM_28[26];	// FPU.scala:646:19
        wbInfo_2_cp = _RANDOM_28[27];	// FPU.scala:646:19
        wbInfo_2_pipeid = _RANDOM_28[29:28];	// FPU.scala:646:19
        write_port_busy = _RANDOM_28[30];	// Reg.scala:34:16
        _RANDOM_29 = `RANDOM;	// Reg.scala:34:16
        wb_toint_exc = {_RANDOM_29[3:0], _RANDOM_28[31]};	// Reg.scala:34:16
        _T_18 = _RANDOM_29[4];	// FPU.scala:708:55
        _T_19 = _RANDOM_29[6:5];	// FPU.scala:718:25
        _T_20 = _RANDOM_29[11:7];	// FPU.scala:719:35
        _RANDOM_30 = `RANDOM;	// FPU.scala:720:35
        _RANDOM_31 = `RANDOM;	// FPU.scala:720:35
        _T_21 = {_RANDOM_31[12:0], _RANDOM_30, _RANDOM_29[31:12]};	// FPU.scala:720:35
      `endif
    end // initial
  `endif
  assign _T_1 = mem_ctrl_sqrt ? DivSqrtRecF64_io_inReady_sqrt : DivSqrtRecF64_io_inReady_div;	// FPU.scala:722:25, :723:27, Reg.scala:35:23
      wire _T_57 = _T_51 & ~divSqrt_in_flight;	// FPU.scala:704:131, :725:{76,79}
      wire [1:0] _fpiu_io_as_double_rm_1to0 = fpiu_io_as_double_rm[1:0];	// FPU.scala:588:20, :729:29
      wire [4:0] _io_inst_19to15 = io_inst[19:15];	// FPU.scala:557:49
      wire [4:0] _io_inst_24to20 = io_inst[24:20];	// FPU.scala:561:48
      wire _killm = (io_killm | _T_52) & ~mem_cp_valid;	// FPU.scala:516:{25,41,44}
      wire _T_58 = mem_ctrl_fma & mem_ctrl_single;	// FPU.scala:622:56, Reg.scala:35:23
      wire _T_59 = mem_ctrl_fma & ~mem_ctrl_single;	// FPU.scala:627:{62,65}, Reg.scala:35:23
      wire _T_60 = mem_ctrl_fastpipe | mem_ctrl_fromint;	// FPU.scala:631:78, Reg.scala:35:23
      wire _mem_wen = mem_reg_valid & (mem_ctrl_fma | mem_ctrl_fastpipe | mem_ctrl_fromint);	// FPU.scala:517:45, :647:{31,69}, Reg.scala:35:23
      wire _wen_1 = wen[1];	// FPU.scala:648:101, :651:14
      wire _wen_2 = wen[2];	// FPU.scala:648:101, :651:14
      wire _T_61 = ~write_port_busy & _T_60;	// FPU.scala:659:{13,30}, Reg.scala:35:23
      wire [1:0] _T_62 = {_T_58, mem_ctrl_fromint} | {2{_T_59}};	// FPU.scala:633:{63,108}, Reg.scala:35:23
      wire [4:0] _mem_reg_inst_11to7 = mem_reg_inst[11:7];	// FPU.scala:663:37, Reg.scala:35:23
      wire _T_63 = ~write_port_busy & _T_58;	// FPU.scala:659:{13,30}, Reg.scala:35:23
      wire _T_64 = ~write_port_busy & _T_59;	// FPU.scala:659:{13,30}, Reg.scala:35:23
      wire _T_65 = DivSqrtRecF64_io_outValid_div | DivSqrtRecF64_io_outValid_sqrt;	// FPU.scala:722:25, :724:52
      wire _T_66 = _T_57 & _T_1;	// FPU.scala:723:27, :731:30
  always @(posedge clock) begin	// Reg.scala:35:23
    if (reset) begin	// FPU.scala:509:25
      ex_reg_valid <= 1'h0;	// FPU.scala:509:25
      mem_reg_valid <= 1'h0;	// FPU.scala:509:25, :513:26
      mem_cp_valid <= 1'h0;	// FPU.scala:509:25, :515:25
      wb_reg_valid <= 1'h0;	// FPU.scala:509:25, :517:25
      wb_cp_valid <= 1'h0;	// FPU.scala:509:25, :518:24
      divSqrt_in_flight <= 1'h0;	// FPU.scala:509:25, :614:30
      wen <= 3'h0;	// Bitwise.scala:71:12, FPU.scala:645:16
    end
    else begin	// FPU.scala:509:25
      ex_reg_valid <= io_valid;	// FPU.scala:509:25
      mem_reg_valid <= ex_reg_valid & ~io_killx | _ex_cp_valid;	// FPU.scala:510:32, :513:{26,45,48,58}
      mem_cp_valid <= _ex_cp_valid;	// FPU.scala:515:25
      wb_reg_valid <= mem_reg_valid & (~_killm | mem_cp_valid);	// FPU.scala:516:44, :517:{25,45,49,56}
      wb_cp_valid <= mem_cp_valid;	// FPU.scala:516:44, :518:24
      wen <= ~_mem_wen | _killm ? {1'h0, wen[2:1]} : {_T_59, wen[2] | _T_58, wen[1] | _T_60};	// FPU.scala:509:25, :648:101, :653:14, :656:{11,23}
      divSqrt_in_flight <= ~_T_65 & (_T_66 | divSqrt_in_flight);	// FPU.scala:704:131, :732:25, :742:25
    end
    if (io_valid)	// Reg.scala:35:23
      ex_reg_inst <= io_inst;	// Reg.scala:35:23
    if (ex_reg_valid)	// FPU.scala:510:32, Reg.scala:35:23
      mem_reg_inst <= ex_reg_inst;	// Reg.scala:35:23
    if (io_valid) begin	// Reg.scala:35:23
      _T_2 <= fp_decoder_io_sigs_cmd;	// FPU.scala:520:26, Reg.scala:35:23
      _T_3 <= fp_decoder_io_sigs_ldst;	// FPU.scala:520:26, Reg.scala:35:23
      _T_4 <= fp_decoder_io_sigs_wen;	// FPU.scala:520:26, Reg.scala:35:23
      _T_5 <= fp_decoder_io_sigs_ren1;	// FPU.scala:520:26, Reg.scala:35:23
      _T_6 <= fp_decoder_io_sigs_ren2;	// FPU.scala:520:26, Reg.scala:35:23
      _T_7 <= fp_decoder_io_sigs_ren3;	// FPU.scala:520:26, Reg.scala:35:23
      _T_8 <= fp_decoder_io_sigs_swap12;	// FPU.scala:520:26, Reg.scala:35:23
      _T_9 <= fp_decoder_io_sigs_swap23;	// FPU.scala:520:26, Reg.scala:35:23
      _T_10 <= fp_decoder_io_sigs_single;	// FPU.scala:520:26, Reg.scala:35:23
      _T_11 <= fp_decoder_io_sigs_fromint;	// FPU.scala:520:26, Reg.scala:35:23
      _T_12 <= fp_decoder_io_sigs_toint;	// FPU.scala:520:26, Reg.scala:35:23
      _T_13 <= fp_decoder_io_sigs_fastpipe;	// FPU.scala:520:26, Reg.scala:35:23
      _T_14 <= fp_decoder_io_sigs_fma;	// FPU.scala:520:26, Reg.scala:35:23
      _T_15 <= fp_decoder_io_sigs_div;	// FPU.scala:520:26, Reg.scala:35:23
      _T_16 <= fp_decoder_io_sigs_sqrt;	// FPU.scala:520:26, Reg.scala:35:23
      _T_17 <= fp_decoder_io_sigs_wflags;	// FPU.scala:520:26, Reg.scala:35:23
    end
    if (_req_valid) begin	// Reg.scala:35:23
      mem_ctrl_cmd <= _ex_ctrl_cmd;	// Reg.scala:35:23
      mem_ctrl_ldst <= _ex_cp_valid ? io_cp_req_bits_ldst : _T_3;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_wen <= _ex_cp_valid ? io_cp_req_bits_wen : _T_4;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_ren1 <= _ex_cp_valid ? io_cp_req_bits_ren1 : _T_5;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_ren2 <= _ex_cp_valid ? io_cp_req_bits_ren2 : _T_6;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_ren3 <= _ex_cp_valid ? io_cp_req_bits_ren3 : _T_7;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_swap12 <= _ex_cp_valid ? io_cp_req_bits_swap12 : _T_8;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_swap23 <= _ex_cp_valid ? io_cp_req_bits_swap23 : _T_9;	// FPU.scala:529:20, Reg.scala:35:23
      mem_ctrl_single <= _ex_ctrl_single;	// Reg.scala:35:23
      mem_ctrl_fromint <= _ex_ctrl_fromint;	// Reg.scala:35:23
      mem_ctrl_toint <= _ex_ctrl_toint;	// Reg.scala:35:23
      mem_ctrl_fastpipe <= _ex_ctrl_fastpipe;	// Reg.scala:35:23
      mem_ctrl_fma <= _ex_ctrl_fma;	// Reg.scala:35:23
      mem_ctrl_div <= _ex_ctrl_div;	// Reg.scala:35:23
      mem_ctrl_sqrt <= _ex_ctrl_sqrt;	// Reg.scala:35:23
      mem_ctrl_wflags <= _ex_ctrl_wflags;	// Reg.scala:35:23
    end
    if (mem_reg_valid) begin	// FPU.scala:517:45, Reg.scala:35:23
      wb_ctrl_cmd <= mem_ctrl_cmd;	// Reg.scala:35:23
      wb_ctrl_ldst <= mem_ctrl_ldst;	// Reg.scala:35:23
      wb_ctrl_wen <= mem_ctrl_wen;	// Reg.scala:35:23
      wb_ctrl_ren1 <= mem_ctrl_ren1;	// Reg.scala:35:23
      wb_ctrl_ren2 <= mem_ctrl_ren2;	// Reg.scala:35:23
      wb_ctrl_ren3 <= mem_ctrl_ren3;	// Reg.scala:35:23
      wb_ctrl_swap12 <= mem_ctrl_swap12;	// Reg.scala:35:23
      wb_ctrl_swap23 <= mem_ctrl_swap23;	// Reg.scala:35:23
      wb_ctrl_single <= mem_ctrl_single;	// Reg.scala:35:23
      wb_ctrl_fromint <= mem_ctrl_fromint;	// Reg.scala:35:23
      wb_ctrl_toint <= mem_ctrl_toint;	// Reg.scala:35:23
      wb_ctrl_fastpipe <= mem_ctrl_fastpipe;	// Reg.scala:35:23
      wb_ctrl_fma <= mem_ctrl_fma;	// Reg.scala:35:23
      wb_ctrl_div <= mem_ctrl_div;	// Reg.scala:35:23
      wb_ctrl_sqrt <= mem_ctrl_sqrt;	// Reg.scala:35:23
      wb_ctrl_wflags <= mem_ctrl_wflags;	// Reg.scala:35:23
    end
    load_wb <= io_dmem_resp_val;	// FPU.scala:534:20
    if (io_dmem_resp_val) begin	// Reg.scala:35:23
      load_wb_single <= ~(io_dmem_resp_type[0]);	// FPU.scala:535:{34,52}, Reg.scala:35:23
      load_wb_data <= io_dmem_resp_data;	// Reg.scala:35:23
      load_wb_tag <= io_dmem_resp_tag;	// Reg.scala:35:23
    end
    if (io_valid) begin	// FPU.scala:565:34
      ex_ra1 <= fp_decoder_io_sigs_ren2 & fp_decoder_io_sigs_swap12 ? _io_inst_24to20 :
                                                ~fp_decoder_io_sigs_ren1 | fp_decoder_io_sigs_swap12 ? ex_ra1 : _io_inst_19to15;	// FPU.scala:520:26, :557:39, :561:38
      ex_ra2 <= ~fp_decoder_io_sigs_ren2 | fp_decoder_io_sigs_swap12 | fp_decoder_io_sigs_swap23 ?
                                                (fp_decoder_io_sigs_ren1 & fp_decoder_io_sigs_swap12 ? _io_inst_19to15 : ex_ra2) :
                                                _io_inst_24to20;	// FPU.scala:520:26, :558:38, :563:58
      ex_ra3 <= fp_decoder_io_sigs_ren3 ? io_inst[31:27] : fp_decoder_io_sigs_ren2 &
                                                fp_decoder_io_sigs_swap23 ? _io_inst_24to20 : ex_ra3;	// FPU.scala:520:26, :562:38, :565:{34,44}
    end
    if (_req_valid)	// Reg.scala:35:23
      write_port_busy <= _mem_wen & |({_T_59, _T_58} & {_ex_ctrl_fma & _ex_ctrl_single, _ex_ctrl_fastpipe |
                                                _ex_ctrl_fromint}) | wen[2] & (_ex_ctrl_fastpipe | _ex_ctrl_fromint);	// FPU.scala:622:56, :631:78, :648:{43,62,89,93,101}, Reg.scala:35:23
    wbInfo_0_cp <= _mem_wen & _T_61 ? mem_cp_valid : _wen_1 ? wbInfo_1_cp : wbInfo_0_cp;	// FPU.scala:516:44, :651:33, :660:22
    wbInfo_0_single <= _mem_wen & _T_61 ? mem_ctrl_single : _wen_1 ? wbInfo_1_single : wbInfo_0_single;	// FPU.scala:651:33, :661:26, Reg.scala:35:23
    wbInfo_0_pipeid <= _mem_wen & _T_61 ? _T_62 : _wen_1 ? wbInfo_1_pipeid : wbInfo_0_pipeid;	// FPU.scala:651:33, :662:26
    wbInfo_0_rd <= _mem_wen & _T_61 ? _mem_reg_inst_11to7 : _wen_1 ? wbInfo_1_rd : wbInfo_0_rd;	// FPU.scala:651:33, :663:22
    wbInfo_1_cp <= _mem_wen & _T_63 ? mem_cp_valid : _wen_2 ? wbInfo_2_cp : wbInfo_1_cp;	// FPU.scala:516:44, :651:33, :660:22
    wbInfo_1_single <= _mem_wen & _T_63 ? mem_ctrl_single : _wen_2 ? wbInfo_2_single : wbInfo_1_single;	// FPU.scala:651:33, :661:26, Reg.scala:35:23
    wbInfo_1_pipeid <= _mem_wen & _T_63 ? _T_62 : _wen_2 ? wbInfo_2_pipeid : wbInfo_1_pipeid;	// FPU.scala:651:33, :662:26
    wbInfo_1_rd <= _mem_wen & _T_63 ? _mem_reg_inst_11to7 : _wen_2 ? wbInfo_2_rd : wbInfo_1_rd;	// FPU.scala:651:33, :663:22
    if (_mem_wen & _T_64)	// FPU.scala:660:22
      wbInfo_2_cp <= mem_cp_valid;	// FPU.scala:516:44, :660:22
    if (_mem_wen & _T_64)	// FPU.scala:661:26
      wbInfo_2_single <= mem_ctrl_single;	// FPU.scala:661:26, Reg.scala:35:23
    if (_mem_wen & _T_64)	// FPU.scala:662:26
      wbInfo_2_pipeid <= _T_62;	// FPU.scala:662:26
    if (_mem_wen & _T_64)	// FPU.scala:663:22
      wbInfo_2_rd <= _mem_reg_inst_11to7;	// FPU.scala:663:22
    if (mem_ctrl_toint)	// Reg.scala:35:23
      wb_toint_exc <= fpiu_io_out_bits_exc;	// FPU.scala:588:20, Reg.scala:35:23
    _T_18_53 <= _T_59 | mem_ctrl_div | mem_ctrl_sqrt;	// FPU.scala:708:{55,112}, Reg.scala:35:23
    if (_T_66) begin	// FPU.scala:736:18
      divSqrt_killed <= _killm;	// FPU.scala:733:22
      divSqrt_single <= mem_ctrl_single;	// FPU.scala:734:22, Reg.scala:35:23
      divSqrt_waddr <= _mem_reg_inst_11to7;	// FPU.scala:735:21
      _T_19_54 <= _fpiu_io_as_double_rm_1to0;	// FPU.scala:736:18
    end
    divSqrt_wen <= _T_65 & ~divSqrt_killed;	// FPU.scala:733:22, :740:{19,22}
    if (_T_65) begin	// FPU.scala:743:28
      _T_21_56 <= DivSqrtRecF64_io_out;	// FPU.scala:722:25, :741:28
      _T_20_55 <= DivSqrtRecF64_io_exceptionFlags;	// FPU.scala:722:25, :743:28
    end
  end // always @(posedge)
  assign _T_0 = divSqrt_single ? {32'h0, RecFNToRecFN_io_out} : _T_21_56;	// CircuitMath.scala:37:22, FPU.scala:670:20, :741:28, :746:34, :749:25
  assign _T = _T_20_55 | (divSqrt_single ? RecFNToRecFN_io_exceptionFlags : 5'h0);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:12:8, FPU.scala:670:20, :743:28, :746:34, :750:{43,48}
  FPUDecoder fp_decoder (	// FPU.scala:520:26
    .io_inst          (io_inst),
    .io_sigs_cmd      (fp_decoder_io_sigs_cmd),
    .io_sigs_ldst     (fp_decoder_io_sigs_ldst),
    .io_sigs_wen      (fp_decoder_io_sigs_wen),
    .io_sigs_ren1     (fp_decoder_io_sigs_ren1),
    .io_sigs_ren2     (fp_decoder_io_sigs_ren2),
    .io_sigs_ren3     (fp_decoder_io_sigs_ren3),
    .io_sigs_swap12   (fp_decoder_io_sigs_swap12),
    .io_sigs_swap23   (fp_decoder_io_sigs_swap23),
    .io_sigs_single   (fp_decoder_io_sigs_single),
    .io_sigs_fromint  (fp_decoder_io_sigs_fromint),
    .io_sigs_toint    (fp_decoder_io_sigs_toint),
    .io_sigs_fastpipe (fp_decoder_io_sigs_fastpipe),
    .io_sigs_fma      (fp_decoder_io_sigs_fma),
    .io_sigs_div      (fp_decoder_io_sigs_div),
    .io_sigs_sqrt     (fp_decoder_io_sigs_sqrt),
    .io_sigs_wflags   (fp_decoder_io_sigs_wflags)
  );
  regfile_ext regfile (	// FPU.scala:547:20
    .R0_addr (ex_ra1),	// FPU.scala:557:39
    .R0_en   (1'h1),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
    .R0_clk  (clock),
    .R1_addr (ex_ra2),	// FPU.scala:558:38
    .R1_en   (1'h1),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
    .R1_clk  (clock),
    .R2_addr (ex_ra3),	// FPU.scala:562:38
    .R2_en   (1'h1),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
    .R2_clk  (clock),
    .W0_addr (_waddr),
    .W0_en   (~wbInfo_0_cp & _wen_0 | divSqrt_wen),	// FPU.scala:651:33, :668:18, :676:{10,24,35}
    .W0_clk  (clock),
    .W0_data (_wdata),
    .W1_addr (load_wb_tag),	// Reg.scala:35:23
    .W1_en   (load_wb),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:380:7
    .W1_clk  (clock),
    .W1_data (load_wb_single ? {32'h70020000, load_wb_data[31], _T_21 & {~{3{_T_18 &
                ~(|_load_wb_data_22to0)}}, 6'h3F} | {2'h0, &(_T_21[8:7]) & |_load_wb_data_22to0, 6'h0},
                _T_18 ? {_T_20[21:0], 1'h0} : _load_wb_data_22to0} : {load_wb_data[63], _T_26 & {~{3{_T_22
                & ~(|_load_wb_data_51to0)}}, 9'h1FF} | {2'h0, &(_T_26[11:10]) & |_load_wb_data_51to0,
                9'h0}, _T_22 ? {_T_25[50:0], 1'h0} : _load_wb_data_51to0}),	// Bitwise.scala:71:12, Cat.scala:30:58, FPU.scala:509:25, :543:{10,33}, :631:23, Reg.scala:35:23, recFNFromFN.scala:47:22, :52:38, :53:34, :56:{13,26}, :58:37, :64:42, :67:{25,50,63}, :71:{26,28,45,64}, :73:27
    .R0_data (regfile_R0_data),
    .R1_data (regfile_R1_data),
    .R2_data (regfile_R2_data)
  );
  FPUFMAPipe sfma (	// FPU.scala:584:20
    .clock               (clock),
    .reset               (reset),
    .io_in_valid         (_T_48 & _ex_ctrl_single),	// FPU.scala:585:48
    .io_in_bits_cmd      (_T_27),
    .io_in_bits_ldst     (_T_28),
    .io_in_bits_wen      (_T_29),
    .io_in_bits_ren1     (_T_30),
    .io_in_bits_ren2     (_T_31),
    .io_in_bits_ren3     (_T_32),
    .io_in_bits_swap12   (_T_33),
    .io_in_bits_swap23   (_T_34),
    .io_in_bits_single   (_T_35),
    .io_in_bits_fromint  (_T_36),
    .io_in_bits_toint    (_T_37),
    .io_in_bits_fastpipe (_T_38),
    .io_in_bits_fma      (_T_39),
    .io_in_bits_div      (_T_40),
    .io_in_bits_sqrt     (_T_41),
    .io_in_bits_wflags   (_T_42),
    .io_in_bits_rm       (_T_43),
    .io_in_bits_typ      (_T_44),
    .io_in_bits_in1      (_T_45),
    .io_in_bits_in2      (_T_46),
    .io_in_bits_in3      (_T_47),
    .io_out_valid        (sfma_io_out_valid),
    .io_out_bits_data    (sfma_io_out_bits_data),
    .io_out_bits_exc     (sfma_io_out_bits_exc)
  );
  FPToInt fpiu (	// FPU.scala:588:20
    .clock                 (clock),
    .io_in_valid           (_req_valid & (_ex_ctrl_toint | _ex_ctrl_div | _ex_ctrl_sqrt | {_ex_ctrl_cmd[3:2],
                _ex_ctrl_cmd[0]} == 3'h3)),	// FPU.scala:589:{33,82,97}
    .io_in_bits_cmd        (_T_27),
    .io_in_bits_ldst       (_T_28),
    .io_in_bits_wen        (_T_29),
    .io_in_bits_ren1       (_T_30),
    .io_in_bits_ren2       (_T_31),
    .io_in_bits_ren3       (_T_32),
    .io_in_bits_swap12     (_T_33),
    .io_in_bits_swap23     (_T_34),
    .io_in_bits_single     (_T_35),
    .io_in_bits_fromint    (_T_36),
    .io_in_bits_toint      (_T_37),
    .io_in_bits_fastpipe   (_T_38),
    .io_in_bits_fma        (_T_39),
    .io_in_bits_div        (_T_40),
    .io_in_bits_sqrt       (_T_41),
    .io_in_bits_wflags     (_T_42),
    .io_in_bits_rm         (_T_43),
    .io_in_bits_typ        (_T_44),
    .io_in_bits_in1        (_T_45),
    .io_in_bits_in2        (_T_46),
    .io_in_bits_in3        (_T_47),
    .io_as_double_cmd      (fpiu_io_as_double_cmd),
    .io_as_double_ldst     (fpiu_io_as_double_ldst),
    .io_as_double_wen      (fpiu_io_as_double_wen),
    .io_as_double_ren1     (fpiu_io_as_double_ren1),
    .io_as_double_ren2     (fpiu_io_as_double_ren2),
    .io_as_double_ren3     (fpiu_io_as_double_ren3),
    .io_as_double_swap12   (fpiu_io_as_double_swap12),
    .io_as_double_swap23   (fpiu_io_as_double_swap23),
    .io_as_double_single   (fpiu_io_as_double_single),
    .io_as_double_fromint  (fpiu_io_as_double_fromint),
    .io_as_double_toint    (fpiu_io_as_double_toint),
    .io_as_double_fastpipe (fpiu_io_as_double_fastpipe),
    .io_as_double_fma      (fpiu_io_as_double_fma),
    .io_as_double_div      (fpiu_io_as_double_div),
    .io_as_double_sqrt     (fpiu_io_as_double_sqrt),
    .io_as_double_wflags   (fpiu_io_as_double_wflags),
    .io_as_double_rm       (fpiu_io_as_double_rm),
    .io_as_double_typ      (fpiu_io_as_double_typ),
    .io_as_double_in1      (fpiu_io_as_double_in1),
    .io_as_double_in2      (fpiu_io_as_double_in2),
    .io_as_double_in3      (fpiu_io_as_double_in3),
    .io_out_valid          (fpiu_io_out_valid),
    .io_out_bits_lt        (fpiu_io_out_bits_lt),
    .io_out_bits_store     (io_store_data),
    .io_out_bits_toint     (fpiu_io_out_bits_toint),
    .io_out_bits_exc       (fpiu_io_out_bits_exc)
  );
  IntToFP ifpu (	// FPU.scala:598:20
    .clock               (clock),
    .reset               (reset),
    .io_in_valid         (_req_valid & _ex_ctrl_fromint),	// FPU.scala:599:33
    .io_in_bits_cmd      (_T_27),
    .io_in_bits_ldst     (_T_28),
    .io_in_bits_wen      (_T_29),
    .io_in_bits_ren1     (_T_30),
    .io_in_bits_ren2     (_T_31),
    .io_in_bits_ren3     (_T_32),
    .io_in_bits_swap12   (_T_33),
    .io_in_bits_swap23   (_T_34),
    .io_in_bits_single   (_T_35),
    .io_in_bits_fromint  (_T_36),
    .io_in_bits_toint    (_T_37),
    .io_in_bits_fastpipe (_T_38),
    .io_in_bits_fma      (_T_39),
    .io_in_bits_div      (_T_40),
    .io_in_bits_sqrt     (_T_41),
    .io_in_bits_wflags   (_T_42),
    .io_in_bits_rm       (_T_43),
    .io_in_bits_typ      (_T_44),
    .io_in_bits_in1      (_ex_cp_valid ? io_cp_req_bits_in1 : {1'h0, io_fromint_data}),	// FPU.scala:509:25, :601:29
    .io_in_bits_in2      (_T_46),
    .io_in_bits_in3      (_T_47),
    .io_out_valid        (ifpu_io_out_valid),
    .io_out_bits_data    (ifpu_io_out_bits_data),
    .io_out_bits_exc     (ifpu_io_out_bits_exc)
  );
  FPToFP fpmu (	// FPU.scala:603:20
    .clock               (clock),
    .reset               (reset),
    .io_in_valid         (_req_valid & _ex_ctrl_fastpipe),	// FPU.scala:604:33
    .io_in_bits_cmd      (_T_27),
    .io_in_bits_ldst     (_T_28),
    .io_in_bits_wen      (_T_29),
    .io_in_bits_ren1     (_T_30),
    .io_in_bits_ren2     (_T_31),
    .io_in_bits_ren3     (_T_32),
    .io_in_bits_swap12   (_T_33),
    .io_in_bits_swap23   (_T_34),
    .io_in_bits_single   (_T_35),
    .io_in_bits_fromint  (_T_36),
    .io_in_bits_toint    (_T_37),
    .io_in_bits_fastpipe (_T_38),
    .io_in_bits_fma      (_T_39),
    .io_in_bits_div      (_T_40),
    .io_in_bits_sqrt     (_T_41),
    .io_in_bits_wflags   (_T_42),
    .io_in_bits_rm       (_T_43),
    .io_in_bits_typ      (_T_44),
    .io_in_bits_in1      (_T_45),
    .io_in_bits_in2      (_T_46),
    .io_in_bits_in3      (_T_47),
    .io_lt               (fpiu_io_out_bits_lt),	// FPU.scala:588:20
    .io_out_valid        (fpmu_io_out_valid),
    .io_out_bits_data    (fpmu_io_out_bits_data),
    .io_out_bits_exc     (fpmu_io_out_bits_exc)
  );
  FPUFMAPipe_1 FPUFMAPipe (	// FPU.scala:624:28
    .clock               (clock),
    .reset               (reset),
    .io_in_valid         (_T_48 & ~_ex_ctrl_single),	// FPU.scala:625:{56,59}
    .io_in_bits_cmd      (_T_27),
    .io_in_bits_ldst     (_T_28),
    .io_in_bits_wen      (_T_29),
    .io_in_bits_ren1     (_T_30),
    .io_in_bits_ren2     (_T_31),
    .io_in_bits_ren3     (_T_32),
    .io_in_bits_swap12   (_T_33),
    .io_in_bits_swap23   (_T_34),
    .io_in_bits_single   (_T_35),
    .io_in_bits_fromint  (_T_36),
    .io_in_bits_toint    (_T_37),
    .io_in_bits_fastpipe (_T_38),
    .io_in_bits_fma      (_T_39),
    .io_in_bits_div      (_T_40),
    .io_in_bits_sqrt     (_T_41),
    .io_in_bits_wflags   (_T_42),
    .io_in_bits_rm       (_T_43),
    .io_in_bits_typ      (_T_44),
    .io_in_bits_in1      (_T_45),
    .io_in_bits_in2      (_T_46),
    .io_in_bits_in3      (_T_47),
    .io_out_valid        (FPUFMAPipe_io_out_valid),
    .io_out_bits_data    (FPUFMAPipe_io_out_bits_data),
    .io_out_bits_exc     (FPUFMAPipe_io_out_bits_exc)
  );
  DivSqrtRecF64 DivSqrtRecF64 (	// FPU.scala:722:25
    .clock             (clock),
    .reset             (reset),
    .io_inValid        (_T_57),
    .io_sqrtOp         (mem_ctrl_sqrt),	// Reg.scala:35:23
    .io_a              (fpiu_io_as_double_in1),	// FPU.scala:588:20
    .io_b              (fpiu_io_as_double_in2),	// FPU.scala:588:20
    .io_roundingMode   (_fpiu_io_as_double_rm_1to0),
    .io_inReady_div    (DivSqrtRecF64_io_inReady_div),
    .io_inReady_sqrt   (DivSqrtRecF64_io_inReady_sqrt),
    .io_outValid_div   (DivSqrtRecF64_io_outValid_div),
    .io_outValid_sqrt  (DivSqrtRecF64_io_outValid_sqrt),
    .io_out            (DivSqrtRecF64_io_out),
    .io_exceptionFlags (DivSqrtRecF64_io_exceptionFlags)
  );
  RecFNToRecFN_2 RecFNToRecFN (	// FPU.scala:746:34
    .io_in             (_T_21_56),	// FPU.scala:741:28
    .io_roundingMode   (_T_19_54),	// FPU.scala:736:18
    .io_out            (RecFNToRecFN_io_out),
    .io_exceptionFlags (RecFNToRecFN_io_exceptionFlags)
  );
  assign io_fcsr_flags_valid = _wb_toint_valid | divSqrt_wen | _wen_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:668:18, :697:56
  assign io_fcsr_flags_bits = (_wb_toint_valid ? wb_toint_exc : 5'h0) | (divSqrt_wen ? _T : 5'h0) | (_wen_0 ?
                (_wbInfo_0_pipeid_1 ? (_wbInfo_0_pipeid_0 ? FPUFMAPipe_io_out_bits_exc :
                sfma_io_out_bits_exc) : _wbInfo_0_pipeid_0 ? ifpu_io_out_bits_exc : fpmu_io_out_bits_exc) :
                5'h0);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, :12:8, FPU.scala:584:20, :598:20, :603:20, :624:28, :668:18, :699:8, :700:{8,46}, :701:8, :750:43, Package.scala:19:12, Reg.scala:35:23
  assign io_toint_data = fpiu_io_out_bits_toint;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:588:20
  assign io_fcsr_rdy = ~(ex_reg_valid & _ex_ctrl_wflags | mem_reg_valid & mem_ctrl_wflags | _wb_toint_valid | |wen
                | divSqrt_in_flight);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:510:32, :517:45, :648:101, :703:97, :704:{18,33,68,131}, Reg.scala:35:23
  assign io_nack_mem = _T_52;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
  assign io_illegal_rm = io_inst[14] & (io_inst[13:12] != 2'h3 | io_fcsr_rm[2]);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, CircuitMath.scala:32:10, FPU.scala:712:{27,32,43,51,55,69}
  assign io_dec_cmd = fp_decoder_io_sigs_cmd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_ldst = fp_decoder_io_sigs_ldst;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_wen = fp_decoder_io_sigs_wen;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_ren1 = fp_decoder_io_sigs_ren1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_ren2 = fp_decoder_io_sigs_ren2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_ren3 = fp_decoder_io_sigs_ren3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_swap12 = fp_decoder_io_sigs_swap12;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_swap23 = fp_decoder_io_sigs_swap23;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_single = fp_decoder_io_sigs_single;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_fromint = fp_decoder_io_sigs_fromint;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_toint = fp_decoder_io_sigs_toint;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_fastpipe = fp_decoder_io_sigs_fastpipe;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_fma = fp_decoder_io_sigs_fma;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_div = fp_decoder_io_sigs_div;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_sqrt = fp_decoder_io_sigs_sqrt;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_dec_wflags = fp_decoder_io_sigs_wflags;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:520:26
  assign io_sboard_set = wb_reg_valid & ~wb_cp_valid & _T_18_53;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:695:37, :708:{36,49}
  assign io_sboard_clr = ~wb_cp_valid & (divSqrt_wen | _wen_0 & &wbInfo_0_pipeid);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:651:33, :668:18, :708:36, :709:{33,49,60,99}
  assign io_sboard_clra = _waddr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10
  assign io_cp_req_ready = ~ex_reg_valid;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:510:32, :693:22
  assign io_cp_resp_valid = _T_50 | _T_49;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:691:22
  assign io_cp_resp_bits_data = _T_50 ? _wdata : {1'h0, _T_49 ? fpiu_io_out_bits_toint : 64'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, FPU.scala:509:25, :588:20, :594:26, :690:26
  assign io_cp_resp_bits_exc = 5'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:7:10, :12:8
endmodule

module FPUDecoder(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10
  input  [31:0] io_inst,
  output [4:0]  io_sigs_cmd,
  output        io_sigs_ldst, io_sigs_wen, io_sigs_ren1, io_sigs_ren2, io_sigs_ren3,
  output        io_sigs_swap12, io_sigs_swap23, io_sigs_single, io_sigs_fromint,
  output        io_sigs_toint, io_sigs_fastpipe, io_sigs_fma, io_sigs_div, io_sigs_sqrt,
  output        io_sigs_wflags);

  wire _io_inst_6 = io_inst[6];	// Decode.scala:13:65
  wire _T = {io_inst[6], io_inst[4]} == 2'h2;	// Decode.scala:13:{65,121}
  wire _io_inst_5 = io_inst[5];	// Decode.scala:13:65
  assign io_sigs_cmd = {~(io_inst[4]), ~_io_inst_6 | io_inst[30], ~_io_inst_6 | io_inst[29], io_inst[3] |
                &{io_inst[28], io_inst[4]}, io_inst[2] | &{io_inst[27], io_inst[4]}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_ldst = ~_io_inst_6;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:121
  assign io_sigs_wen = {io_inst[31], io_inst[5]} == 2'h0 | io_inst[5:4] == 2'h0 | {io_inst[28], io_inst[5]} ==
                2'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_ren1 = {io_inst[31], io_inst[2]} == 2'h0 | {io_inst[28], io_inst[2]} == 2'h0 | _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_ren2 = {io_inst[30], io_inst[2]} == 2'h0 | _io_inst_5 | _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_ren3 = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10
  assign io_sigs_swap12 = ~_io_inst_6 | &{io_inst[30], io_inst[28], io_inst[4]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_swap23 = {io_inst[29:28], io_inst[4]} == 3'h1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}
  assign io_sigs_single = {io_inst[12], io_inst[6]} == 2'h0 | {io_inst[25], io_inst[6]} == 2'h1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_fromint = &{io_inst[31], io_inst[28], io_inst[4]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}
  assign io_sigs_toint = _io_inst_5 | {io_inst[31], io_inst[28], io_inst[4]} == 3'h5;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_fastpipe = {io_inst[31], io_inst[29], io_inst[4]} == 3'h3 | {io_inst[31:30], io_inst[28], io_inst[4]}
                == 4'h5;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_fma = {io_inst[30:28], io_inst[2]} == 4'h0 | {io_inst[30:29], io_inst[27], io_inst[2]} == 4'h0 |
                _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
  assign io_sigs_div = {io_inst[30], io_inst[28:27], io_inst[4]} == 4'h7;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}
  assign io_sigs_sqrt = {io_inst[31:30], io_inst[28], io_inst[4]} == 4'h7;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}
  assign io_sigs_wflags = {io_inst[29], io_inst[2]} == 2'h0 | _T | {io_inst[27], io_inst[13]} == 2'h2 |
                {io_inst[31:30], io_inst[2]} == 3'h4;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:791:10, Decode.scala:13:{65,121}, :14:30
endmodule

module FPUFMAPipe(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:920:10
  input         clock, reset, io_in_valid,
  input  [4:0]  io_in_bits_cmd,
  input         io_in_bits_ldst, io_in_bits_wen, io_in_bits_ren1, io_in_bits_ren2,
  input         io_in_bits_ren3, io_in_bits_swap12, io_in_bits_swap23, io_in_bits_single,
  input         io_in_bits_fromint, io_in_bits_toint, io_in_bits_fastpipe,
  input         io_in_bits_fma, io_in_bits_div, io_in_bits_sqrt, io_in_bits_wflags,
  input  [2:0]  io_in_bits_rm,
  input  [1:0]  io_in_bits_typ,
  input  [64:0] io_in_bits_in1, io_in_bits_in2, io_in_bits_in3,
  output        io_out_valid,
  output [64:0] io_out_bits_data,
  output [4:0]  io_out_bits_exc);

  wire [32:0] fma_io_out;	// FPU.scala:493:19
  wire [4:0]  fma_io_exceptionFlags;	// FPU.scala:493:19
  reg         valid;	// FPU.scala:482:18
  reg  [4:0]  in_cmd;	// FPU.scala:483:15
  reg         in_ldst;	// FPU.scala:483:15
  reg         in_wen;	// FPU.scala:483:15
  reg         in_ren1;	// FPU.scala:483:15
  reg         in_ren2;	// FPU.scala:483:15
  reg         in_ren3;	// FPU.scala:483:15
  reg         in_swap12;	// FPU.scala:483:15
  reg         in_swap23;	// FPU.scala:483:15
  reg         in_single;	// FPU.scala:483:15
  reg         in_fromint;	// FPU.scala:483:15
  reg         in_toint;	// FPU.scala:483:15
  reg         in_fastpipe;	// FPU.scala:483:15
  reg         in_fma;	// FPU.scala:483:15
  reg         in_div;	// FPU.scala:483:15
  reg         in_sqrt;	// FPU.scala:483:15
  reg         in_wflags;	// FPU.scala:483:15
  reg  [2:0]  in_rm;	// FPU.scala:483:15
  reg  [1:0]  in_typ;	// FPU.scala:483:15
  reg  [64:0] in_in1;	// FPU.scala:483:15
  reg  [64:0] in_in2;	// FPU.scala:483:15
  reg  [64:0] in_in3;	// FPU.scala:483:15
  reg         _T;	// Valid.scala:47:18
  reg  [64:0] _T_0;	// Reg.scala:34:16
  reg  [4:0]  _T_1;	// Reg.scala:34:16
  reg         _T_2;	// Valid.scala:47:18
  reg  [64:0] _T_3;	// Reg.scala:34:16
  reg  [4:0]  _T_4;	// Reg.scala:34:16

  `ifndef SYNTHESIS	// FPU.scala:482:18
    `ifdef RANDOMIZE_REG_INIT	// FPU.scala:482:18
      reg [31:0] _RANDOM;	// FPU.scala:482:18
      reg [31:0] _RANDOM_5;	// FPU.scala:483:15
      reg [31:0] _RANDOM_6;	// FPU.scala:483:15
      reg [31:0] _RANDOM_7;	// FPU.scala:483:15
      reg [31:0] _RANDOM_8;	// FPU.scala:483:15
      reg [31:0] _RANDOM_9;	// FPU.scala:483:15
      reg [31:0] _RANDOM_10;	// FPU.scala:483:15
      reg [31:0] _RANDOM_11;	// Reg.scala:34:16
      reg [31:0] _RANDOM_12;	// Reg.scala:34:16
      reg [31:0] _RANDOM_13;	// Reg.scala:34:16
      reg [31:0] _RANDOM_14;	// Reg.scala:34:16
      reg [31:0] _RANDOM_15;	// Reg.scala:34:16

    `endif
    initial begin	// FPU.scala:482:18
      `INIT_RANDOM_PROLOG_	// FPU.scala:482:18
      `ifdef RANDOMIZE_REG_INIT	// FPU.scala:482:18
        _RANDOM = `RANDOM;	// FPU.scala:482:18
        valid = _RANDOM[0];	// FPU.scala:482:18
        in_cmd = _RANDOM[5:1];	// FPU.scala:483:15
        in_ldst = _RANDOM[6];	// FPU.scala:483:15
        in_wen = _RANDOM[7];	// FPU.scala:483:15
        in_ren1 = _RANDOM[8];	// FPU.scala:483:15
        in_ren2 = _RANDOM[9];	// FPU.scala:483:15
        in_ren3 = _RANDOM[10];	// FPU.scala:483:15
        in_swap12 = _RANDOM[11];	// FPU.scala:483:15
        in_swap23 = _RANDOM[12];	// FPU.scala:483:15
        in_single = _RANDOM[13];	// FPU.scala:483:15
        in_fromint = _RANDOM[14];	// FPU.scala:483:15
        in_toint = _RANDOM[15];	// FPU.scala:483:15
        in_fastpipe = _RANDOM[16];	// FPU.scala:483:15
        in_fma = _RANDOM[17];	// FPU.scala:483:15
        in_div = _RANDOM[18];	// FPU.scala:483:15
        in_sqrt = _RANDOM[19];	// FPU.scala:483:15
        in_wflags = _RANDOM[20];	// FPU.scala:483:15
        in_rm = _RANDOM[23:21];	// FPU.scala:483:15
        in_typ = _RANDOM[25:24];	// FPU.scala:483:15
        _RANDOM_5 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_6 = `RANDOM;	// FPU.scala:483:15
        in_in1 = {_RANDOM_6[26:0], _RANDOM_5, _RANDOM[31:26]};	// FPU.scala:483:15
        _RANDOM_7 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_8 = `RANDOM;	// FPU.scala:483:15
        in_in2 = {_RANDOM_8[27:0], _RANDOM_7, _RANDOM_6[31:27]};	// FPU.scala:483:15
        _RANDOM_9 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_10 = `RANDOM;	// FPU.scala:483:15
        in_in3 = {_RANDOM_10[28:0], _RANDOM_9, _RANDOM_8[31:28]};	// FPU.scala:483:15
        _T = _RANDOM_10[29];	// Valid.scala:47:18
        _RANDOM_11 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_12 = `RANDOM;	// Reg.scala:34:16
        _T_0 = {_RANDOM_12[30:0], _RANDOM_11, _RANDOM_10[31:30]};	// Reg.scala:34:16
        _RANDOM_13 = `RANDOM;	// Reg.scala:34:16
        _T_1 = {_RANDOM_13[3:0], _RANDOM_12[31]};	// Reg.scala:34:16
        _T_2 = _RANDOM_13[4];	// Valid.scala:47:18
        _RANDOM_14 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_15 = `RANDOM;	// Reg.scala:34:16
        _T_3 = {_RANDOM_15[5:0], _RANDOM_14, _RANDOM_13[31:5]};	// Reg.scala:34:16
        _T_4 = _RANDOM_15[10:6];	// Reg.scala:34:16
      `endif
    end // initial
  `endif
      wire _T_5 = io_in_bits_ren3 | io_in_bits_swap23;	// FPU.scala:488:48
  always @(posedge clock) begin	// FPU.scala:482:18
    if (reset) begin	// Valid.scala:47:18
      _T <= 1'h0;	// Conditional.scala:19:11, Valid.scala:47:18
      _T_2 <= 1'h0;	// Conditional.scala:19:11, Valid.scala:47:18
    end
    else begin	// Valid.scala:47:18
      _T <= valid;	// Valid.scala:47:18
      _T_2 <= _T;	// Valid.scala:47:18
    end
    valid <= io_in_valid;	// FPU.scala:482:18
    if (io_in_valid) begin	// FPU.scala:490:45
      in_ldst <= io_in_bits_ldst;	// FPU.scala:485:8
      in_wen <= io_in_bits_wen;	// FPU.scala:485:8
      in_ren1 <= io_in_bits_ren1;	// FPU.scala:485:8
      in_ren2 <= io_in_bits_ren2;	// FPU.scala:485:8
      in_ren3 <= io_in_bits_ren3;	// FPU.scala:485:8
      in_swap12 <= io_in_bits_swap12;	// FPU.scala:485:8
      in_swap23 <= io_in_bits_swap23;	// FPU.scala:485:8
      in_single <= io_in_bits_single;	// FPU.scala:485:8
      in_fromint <= io_in_bits_fromint;	// FPU.scala:485:8
      in_toint <= io_in_bits_toint;	// FPU.scala:485:8
      in_fastpipe <= io_in_bits_fastpipe;	// FPU.scala:485:8
      in_fma <= io_in_bits_fma;	// FPU.scala:485:8
      in_div <= io_in_bits_div;	// FPU.scala:485:8
      in_sqrt <= io_in_bits_sqrt;	// FPU.scala:485:8
      in_wflags <= io_in_bits_wflags;	// FPU.scala:485:8
      in_rm <= io_in_bits_rm;	// FPU.scala:485:8
      in_typ <= io_in_bits_typ;	// FPU.scala:485:8
      in_in1 <= io_in_bits_in1;	// FPU.scala:485:8
      in_cmd <= {3'h0, io_in_bits_cmd[1] & _T_5, io_in_bits_cmd[0]};	// FPU.scala:488:{12,33,37,78}
      in_in2 <= io_in_bits_swap23 ? 65'h80000000 : io_in_bits_in2;	// FPU.scala:489:32
      in_in3 <= _T_5 ? io_in_bits_in3 : {32'h0, io_in_bits_in1[32] ^ io_in_bits_in2[32], 32'h0};	// FPU.scala:480:{29,37,53,62}, :490:45
    end
    if (valid) begin	// Reg.scala:35:23, Valid.scala:47:18
      _T_0 <= {32'h0, fma_io_out};	// FPU.scala:480:62, :493:19, :501:12, Reg.scala:35:23
      _T_1 <= fma_io_exceptionFlags;	// FPU.scala:493:19, Reg.scala:35:23
    end
    if (_T) begin	// Reg.scala:35:23, Valid.scala:47:18
      _T_3 <= _T_0;	// Reg.scala:35:23
      _T_4 <= _T_1;	// Reg.scala:35:23
    end
  end // always @(posedge)
  MulAddRecFN fma (	// FPU.scala:493:19
    .io_op             (in_cmd[1:0]),	// FPU.scala:488:12, :494:13
    .io_a              (in_in1[32:0]),	// FPU.scala:485:8, :496:12
    .io_b              (in_in2[32:0]),	// FPU.scala:489:32, :497:12
    .io_c              (in_in3[32:0]),	// FPU.scala:490:45, :498:12
    .io_roundingMode   (in_rm[1:0]),	// FPU.scala:485:8, :495:23
    .io_out            (fma_io_out),
    .io_exceptionFlags (fma_io_exceptionFlags)
  );
  assign io_out_valid = _T_2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:920:10, Valid.scala:43:17
  assign io_out_bits_data = _T_3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:920:10, Reg.scala:35:23
  assign io_out_bits_exc = _T_4;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:920:10, Reg.scala:35:23
endmodule

module FPToInt(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10
  input         clock, io_in_valid,
  input  [4:0]  io_in_bits_cmd,
  input         io_in_bits_ldst, io_in_bits_wen, io_in_bits_ren1, io_in_bits_ren2,
  input         io_in_bits_ren3, io_in_bits_swap12, io_in_bits_swap23, io_in_bits_single,
  input         io_in_bits_fromint, io_in_bits_toint, io_in_bits_fastpipe,
  input         io_in_bits_fma, io_in_bits_div, io_in_bits_sqrt, io_in_bits_wflags,
  input  [2:0]  io_in_bits_rm,
  input  [1:0]  io_in_bits_typ,
  input  [64:0] io_in_bits_in1, io_in_bits_in2, io_in_bits_in3,
  output [4:0]  io_as_double_cmd,
  output        io_as_double_ldst, io_as_double_wen, io_as_double_ren1,
  output        io_as_double_ren2, io_as_double_ren3, io_as_double_swap12,
  output        io_as_double_swap23, io_as_double_single, io_as_double_fromint,
  output        io_as_double_toint, io_as_double_fastpipe, io_as_double_fma,
  output        io_as_double_div, io_as_double_sqrt, io_as_double_wflags,
  output [2:0]  io_as_double_rm,
  output [1:0]  io_as_double_typ,
  output [64:0] io_as_double_in1, io_as_double_in2, io_as_double_in3,
  output        io_out_valid, io_out_bits_lt,
  output [63:0] io_out_bits_store, io_out_bits_toint,
  output [4:0]  io_out_bits_exc);

  wire        _T;	// FPU.scala:339:35
  wire [63:0] RecFNToIN_1_io_out;	// FPU.scala:336:24
  wire [2:0]  RecFNToIN_1_io_intExceptionFlags;	// FPU.scala:336:24
  wire [31:0] RecFNToIN_io_out;	// FPU.scala:336:24
  wire [2:0]  RecFNToIN_io_intExceptionFlags;	// FPU.scala:336:24
  wire        dcmp_io_lt;	// FPU.scala:319:20
  wire        dcmp_io_eq;	// FPU.scala:319:20
  wire        dcmp_io_gt;	// FPU.scala:319:20
  wire [4:0]  dcmp_io_exceptionFlags;	// FPU.scala:319:20
  reg  [4:0]  in_cmd;	// FPU.scala:286:15
  reg         in_ldst;	// FPU.scala:286:15
  reg         in_wen;	// FPU.scala:286:15
  reg         in_ren1;	// FPU.scala:286:15
  reg         in_ren2;	// FPU.scala:286:15
  reg         in_ren3;	// FPU.scala:286:15
  reg         in_swap12;	// FPU.scala:286:15
  reg         in_swap23;	// FPU.scala:286:15
  reg         in_single;	// FPU.scala:286:15
  reg         in_fromint;	// FPU.scala:286:15
  reg         in_toint;	// FPU.scala:286:15
  reg         in_fastpipe;	// FPU.scala:286:15
  reg         in_fma;	// FPU.scala:286:15
  reg         in_div;	// FPU.scala:286:15
  reg         in_sqrt;	// FPU.scala:286:15
  reg         in_wflags;	// FPU.scala:286:15
  reg  [2:0]  in_rm;	// FPU.scala:286:15
  reg  [1:0]  in_typ;	// FPU.scala:286:15
  reg  [64:0] in_in1;	// FPU.scala:286:15
  reg  [64:0] in_in2;	// FPU.scala:286:15
  reg  [64:0] in_in3;	// FPU.scala:286:15
  reg         valid;	// FPU.scala:287:18

  `ifndef SYNTHESIS	// FPU.scala:286:15
    `ifdef RANDOMIZE_REG_INIT	// FPU.scala:286:15
      reg [31:0] _RANDOM;	// FPU.scala:286:15
      reg [31:0] _RANDOM_0;	// FPU.scala:286:15
      reg [31:0] _RANDOM_1;	// FPU.scala:286:15
      reg [31:0] _RANDOM_2;	// FPU.scala:286:15
      reg [31:0] _RANDOM_3;	// FPU.scala:286:15
      reg [31:0] _RANDOM_4;	// FPU.scala:286:15
      reg [31:0] _RANDOM_5;	// FPU.scala:286:15

    `endif
    initial begin	// FPU.scala:286:15
      `INIT_RANDOM_PROLOG_	// FPU.scala:286:15
      `ifdef RANDOMIZE_REG_INIT	// FPU.scala:286:15
        _RANDOM = `RANDOM;	// FPU.scala:286:15
        in_cmd = _RANDOM[4:0];	// FPU.scala:286:15
        in_ldst = _RANDOM[5];	// FPU.scala:286:15
        in_wen = _RANDOM[6];	// FPU.scala:286:15
        in_ren1 = _RANDOM[7];	// FPU.scala:286:15
        in_ren2 = _RANDOM[8];	// FPU.scala:286:15
        in_ren3 = _RANDOM[9];	// FPU.scala:286:15
        in_swap12 = _RANDOM[10];	// FPU.scala:286:15
        in_swap23 = _RANDOM[11];	// FPU.scala:286:15
        in_single = _RANDOM[12];	// FPU.scala:286:15
        in_fromint = _RANDOM[13];	// FPU.scala:286:15
        in_toint = _RANDOM[14];	// FPU.scala:286:15
        in_fastpipe = _RANDOM[15];	// FPU.scala:286:15
        in_fma = _RANDOM[16];	// FPU.scala:286:15
        in_div = _RANDOM[17];	// FPU.scala:286:15
        in_sqrt = _RANDOM[18];	// FPU.scala:286:15
        in_wflags = _RANDOM[19];	// FPU.scala:286:15
        in_rm = _RANDOM[22:20];	// FPU.scala:286:15
        in_typ = _RANDOM[24:23];	// FPU.scala:286:15
        _RANDOM_0 = `RANDOM;	// FPU.scala:286:15
        _RANDOM_1 = `RANDOM;	// FPU.scala:286:15
        in_in1 = {_RANDOM_1[25:0], _RANDOM_0, _RANDOM[31:25]};	// FPU.scala:286:15
        _RANDOM_2 = `RANDOM;	// FPU.scala:286:15
        _RANDOM_3 = `RANDOM;	// FPU.scala:286:15
        in_in2 = {_RANDOM_3[26:0], _RANDOM_2, _RANDOM_1[31:26]};	// FPU.scala:286:15
        _RANDOM_4 = `RANDOM;	// FPU.scala:286:15
        _RANDOM_5 = `RANDOM;	// FPU.scala:286:15
        in_in3 = {_RANDOM_5[27:0], _RANDOM_4, _RANDOM_3[31:27]};	// FPU.scala:286:15
        valid = _RANDOM_5[28];	// FPU.scala:287:18
      `endif
    end // initial
  `endif
      wire _T_0 = io_in_bits_single & ~io_in_bits_ldst & io_in_bits_cmd[3:2] != 2'h3;	// FPU.scala:293:{47,64,82}, fNFromRecFN.scala:58:55
      wire [2:0] _io_in_bits_in1_31to29 = io_in_bits_in1[31:29];	// FPU.scala:242:26
      wire [11:0] _T_1 = {3'h0, io_in_bits_in1[31:23]} + 12'h700;	// Cat.scala:30:58, FPU.scala:239:19, :243:{31,53}
      wire [2:0] _io_in_bits_in2_31to29 = io_in_bits_in2[31:29];	// FPU.scala:242:26
      wire [11:0] _T_2 = {3'h0, io_in_bits_in2[31:23]} + 12'h700;	// Cat.scala:30:58, FPU.scala:239:19, :243:{31,53}
  always @(posedge clock) begin	// FPU.scala:287:18
    valid <= io_in_valid;	// FPU.scala:287:18
    if (io_in_valid) begin	// FPU.scala:295:14
      in_cmd <= io_in_bits_cmd;	// FPU.scala:292:8
      in_ldst <= io_in_bits_ldst;	// FPU.scala:292:8
      in_wen <= io_in_bits_wen;	// FPU.scala:292:8
      in_ren1 <= io_in_bits_ren1;	// FPU.scala:292:8
      in_ren2 <= io_in_bits_ren2;	// FPU.scala:292:8
      in_ren3 <= io_in_bits_ren3;	// FPU.scala:292:8
      in_swap12 <= io_in_bits_swap12;	// FPU.scala:292:8
      in_swap23 <= io_in_bits_swap23;	// FPU.scala:292:8
      in_single <= io_in_bits_single;	// FPU.scala:292:8
      in_fromint <= io_in_bits_fromint;	// FPU.scala:292:8
      in_toint <= io_in_bits_toint;	// FPU.scala:292:8
      in_fastpipe <= io_in_bits_fastpipe;	// FPU.scala:292:8
      in_fma <= io_in_bits_fma;	// FPU.scala:292:8
      in_div <= io_in_bits_div;	// FPU.scala:292:8
      in_sqrt <= io_in_bits_sqrt;	// FPU.scala:292:8
      in_wflags <= io_in_bits_wflags;	// FPU.scala:292:8
      in_rm <= io_in_bits_rm;	// FPU.scala:292:8
      in_typ <= io_in_bits_typ;	// FPU.scala:292:8
      in_in3 <= io_in_bits_in3;	// FPU.scala:292:8
      in_in1 <= _T_0 ? {io_in_bits_in1[32], _io_in_bits_in1_31to29 == 3'h0 | _io_in_bits_in1_31to29 > 3'h5
                                                ? {_io_in_bits_in1_31to29, _T_1[8:0]} : _T_1, io_in_bits_in1[22:0], 29'h0} : io_in_bits_in1;	// Cat.scala:30:58, FPU.scala:237:18, :238:21, :240:43, :244:{10,19,25,36,65}, :294:14
      in_in2 <= _T_0 ? {io_in_bits_in2[32], _io_in_bits_in2_31to29 == 3'h0 | _io_in_bits_in2_31to29 > 3'h5
                                                ? {_io_in_bits_in2_31to29, _T_2[8:0]} : _T_2, io_in_bits_in2[22:0], 29'h0} : io_in_bits_in2;	// Cat.scala:30:58, FPU.scala:237:18, :238:21, :240:43, :244:{10,19,25,36,65}, :295:14
    end
  end // always @(posedge)
      wire _in_in1_32 = in_in1[32];	// FPU.scala:294:14, fNFromRecFN.scala:45:22
      wire [22:0] _in_in1_22to0 = in_in1[22:0];	// FPU.scala:294:14, fNFromRecFN.scala:47:25
      wire _T_3 = in_in1[29:23] < 7'h2;	// FPU.scala:294:14, fNFromRecFN.scala:49:{39,57}
      wire [2:0] _in_in1_31to29 = in_in1[31:29];	// FPU.scala:294:14, fNFromRecFN.scala:51:19
      wire [1:0] _in_in1_31to30 = in_in1[31:30];	// FPU.scala:294:14, fNFromRecFN.scala:52:24
      wire _T_4 = _in_in1_31to30 == 2'h1;	// fNFromRecFN.scala:52:49
      wire _T_5 = _in_in1_31to29 == 3'h1 | _T_4 & _T_3;	// fNFromRecFN.scala:51:{44,57}, :52:62
      wire _T_6 = _T_4 & ~_T_3 | _in_in1_31to30 == 2'h2;	// fNFromRecFN.scala:49:57, :55:58, :56:{18,39}, :57:48
      wire _in_in1_29 = in_in1[29];	// FPU.scala:294:14, fNFromRecFN.scala:59:39
      wire [23:0] _T_7 = {1'h1, _in_in1_22to0} >> 5'h2 - in_in1[27:23];	// Cat.scala:30:58, FPU.scala:294:14, fNFromRecFN.scala:51:44, :61:{39,46}, :63:35
      wire _in_in1_64 = in_in1[64];	// FPU.scala:294:14, fNFromRecFN.scala:45:22
      wire [51:0] _in_in1_51to0 = in_in1[51:0];	// FPU.scala:294:14, fNFromRecFN.scala:47:25
      wire _T_8 = in_in1[61:52] < 10'h2;	// FPU.scala:294:14, fNFromRecFN.scala:49:{39,57}
      wire [2:0] _in_in1_63to61 = in_in1[63:61];	// FPU.scala:294:14, fNFromRecFN.scala:51:19
      wire [1:0] _in_in1_63to62 = in_in1[63:62];	// FPU.scala:294:14, fNFromRecFN.scala:52:24
      wire _T_9 = _in_in1_63to62 == 2'h1;	// fNFromRecFN.scala:52:49
      wire _T_10 = _in_in1_63to61 == 3'h1 | _T_9 & _T_8;	// fNFromRecFN.scala:51:{44,57}, :52:62
      wire _T_11 = _T_9 & ~_T_8 | _in_in1_63to62 == 2'h2;	// fNFromRecFN.scala:49:57, :55:58, :56:{18,39}, :57:48
      wire _in_in1_61 = in_in1[61];	// FPU.scala:294:14, fNFromRecFN.scala:59:39
      wire [52:0] _T_12 = {1'h1, _in_in1_51to0} >> 6'h2 - in_in1[57:52];	// Cat.scala:30:58, FPU.scala:294:14, fNFromRecFN.scala:51:44, :61:{39,46}, :63:35
      wire [63:0] _unrec_int = in_single ? {{33{_in_in1_32}}, _T_6 ? in_in1[30:23] + 8'h7F : {8{&_in_in1_31to30}}, _T_6 |
                &_in_in1_31to30 & _in_in1_29 ? _in_in1_22to0 : _T_5 ? _T_7[22:0] : 23'h0} : {_in_in1_64,
                _T_11 ? in_in1[62:52] + 11'h3FF : {11{&_in_in1_63to62}}, _T_11 | &_in_in1_63to62 &
                _in_in1_61 ? _in_in1_51to0 : _T_10 ? _T_12[51:0] : 52'h0};	// Bitwise.scala:71:12, Cat.scala:30:58, FPU.scala:292:8, :294:14, :304:10, fNFromRecFN.scala:58:55, :59:31, :63:53, :65:{18,36}, :68:16, :70:{16,26}, :72:20
      wire _T_13 = _in_in1_31to29 == 3'h0;	// Cat.scala:30:58, FPU.scala:212:23
      wire _T_14 = &_in_in1_31to30 & ~_in_in1_29;	// FPU.scala:213:{27,30}, fNFromRecFN.scala:58:55
      wire _in_in1_22 = in_in1[22];	// FPU.scala:215:31, :294:14
      wire _T_15 = _in_in1_63to61 == 3'h0;	// Cat.scala:30:58, FPU.scala:212:23
      wire _T_16 = &_in_in1_63to62 & ~_in_in1_61;	// FPU.scala:213:{27,30}, fNFromRecFN.scala:58:55
      wire _in_in1_51 = in_in1[51];	// FPU.scala:215:31, :294:14
      wire _T_17 = in_cmd[3:2] == 2'h1;	// FPU.scala:292:8, :328:16, fNFromRecFN.scala:52:49
      wire _T_18 = in_cmd[3:2] == 2'h2;	// FPU.scala:292:8, :328:16, :332:16, fNFromRecFN.scala:49:57
      wire [1:0] _in_rm_1to0 = in_rm[1:0];	// FPU.scala:292:8, :338:28
  assign _T = in_typ[0];	// FPU.scala:292:8, :339:35
      wire _in_typ_1 = in_typ[1];	// FPU.scala:292:8, Package.scala:44:13
  CompareRecFN dcmp (	// FPU.scala:319:20
    .io_a              (in_in1),	// FPU.scala:294:14
    .io_b              (in_in2),	// FPU.scala:295:14
    .io_signaling      (~(in_rm[1])),	// FPU.scala:292:8, :322:{24,30}
    .io_lt             (dcmp_io_lt),
    .io_eq             (dcmp_io_eq),
    .io_gt             (dcmp_io_gt),
    .io_exceptionFlags (dcmp_io_exceptionFlags)
  );
  RecFNToIN RecFNToIN (	// FPU.scala:336:24
    .io_in                (in_in1),	// FPU.scala:294:14
    .io_roundingMode      (_in_rm_1to0),
    .io_signedOut         (~_T),	// FPU.scala:339:{28,35}
    .io_out               (RecFNToIN_io_out),
    .io_intExceptionFlags (RecFNToIN_io_intExceptionFlags)
  );
  RecFNToIN_1 RecFNToIN_1 (	// FPU.scala:336:24
    .io_in                (in_in1),	// FPU.scala:294:14
    .io_roundingMode      (_in_rm_1to0),
    .io_signedOut         (~_T),	// FPU.scala:339:{28,35}
    .io_out               (RecFNToIN_1_io_out),
    .io_intExceptionFlags (RecFNToIN_1_io_intExceptionFlags)
  );
  assign io_as_double_cmd = in_cmd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_ldst = in_ldst;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_wen = in_wen;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_ren1 = in_ren1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_ren2 = in_ren2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_ren3 = in_ren3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_swap12 = in_swap12;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_swap23 = in_swap23;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_single = in_single;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_fromint = in_fromint;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_toint = in_toint;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_fastpipe = in_fastpipe;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_fma = in_fma;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_div = in_div;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_sqrt = in_sqrt;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_wflags = in_wflags;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_rm = in_rm;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_typ = in_typ;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_as_double_in1 = in_in1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:294:14
  assign io_as_double_in2 = in_in2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:295:14
  assign io_as_double_in3 = in_in3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:292:8
  assign io_out_valid = valid;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10
  assign io_out_bits_lt = dcmp_io_lt;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, FPU.scala:319:20
  assign io_out_bits_store = _unrec_int;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10
  assign io_out_bits_toint = _T_18 ? (_in_typ_1 ? RecFNToIN_1_io_out : {{32{RecFNToIN_io_out[31]}}, RecFNToIN_io_out}) :
                _T_17 ? {63'h0, |(~(in_rm[1:0]) & {dcmp_io_lt, dcmp_io_eq})} : in_rm[0] ? {54'h0, in_single
                ? {&_in_in1_31to29 & _in_in1_22, &_in_in1_31to29 & ~_in_in1_22, _T_14 & ~_in_in1_32, _T_6 &
                ~_in_in1_32, _T_5 & ~_in_in1_32, _T_13 & ~_in_in1_32, _T_13 & _in_in1_32, _T_5 &
                _in_in1_32, _T_6 & _in_in1_32, _T_14 & _in_in1_32} : {&_in_in1_63to61 & _in_in1_51,
                &_in_in1_63to61 & ~_in_in1_51, _T_16 & ~_in_in1_64, _T_11 & ~_in_in1_64, _T_10 &
                ~_in_in1_64, _T_15 & ~_in_in1_64, _T_15 & _in_in1_64, _T_10 & _in_in1_64, _T_11 &
                _in_in1_64, _T_16 & _in_in1_64}} : _unrec_int;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, Bitwise.scala:71:12, Cat.scala:30:58, FPU.scala:214:22, :215:{24,27}, :216:24, :218:{31,34,50}, :219:{21,38,55}, :220:{21,39,54}, :292:8, :316:10, :319:20, :324:{27,33}, :329:{23,27,34,65}, :336:24, :341:27, Package.scala:40:38
  assign io_out_bits_exc = _T_18 ? (_in_typ_1 ? {|(RecFNToIN_1_io_intExceptionFlags[2:1]), 3'h0,
                RecFNToIN_1_io_intExceptionFlags[0]} : {|(RecFNToIN_io_intExceptionFlags[2:1]), 3'h0,
                RecFNToIN_io_intExceptionFlags[0]}) : _T_17 ? dcmp_io_exceptionFlags : 5'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:983:10, Cat.scala:30:58, FPU.scala:319:20, :330:21, :336:24, :342:{25,57,64,106}
endmodule

module IntToFP(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1308:10
  input         clock, reset, io_in_valid,
  input  [4:0]  io_in_bits_cmd,
  input         io_in_bits_ldst, io_in_bits_wen, io_in_bits_ren1, io_in_bits_ren2,
  input         io_in_bits_ren3, io_in_bits_swap12, io_in_bits_swap23, io_in_bits_single,
  input         io_in_bits_fromint, io_in_bits_toint, io_in_bits_fastpipe,
  input         io_in_bits_fma, io_in_bits_div, io_in_bits_sqrt, io_in_bits_wflags,
  input  [2:0]  io_in_bits_rm,
  input  [1:0]  io_in_bits_typ,
  input  [64:0] io_in_bits_in1, io_in_bits_in2, io_in_bits_in3,
  output        io_out_valid,
  output [64:0] io_out_bits_data,
  output [4:0]  io_out_bits_exc);

  wire [64:0] INToRecFN_1_io_out;	// FPU.scala:391:25
  wire [4:0]  INToRecFN_1_io_exceptionFlags;	// FPU.scala:391:25
  wire [32:0] INToRecFN_io_out;	// FPU.scala:381:21
  wire [4:0]  INToRecFN_io_exceptionFlags;	// FPU.scala:381:21
  reg         _T;	// Valid.scala:47:18
  reg  [4:0]  _T_0;	// Reg.scala:34:16
  reg         _T_1;	// Reg.scala:34:16
  reg         _T_2;	// Reg.scala:34:16
  reg         _T_3;	// Reg.scala:34:16
  reg         _T_4;	// Reg.scala:34:16
  reg         _T_5;	// Reg.scala:34:16
  reg         _T_6;	// Reg.scala:34:16
  reg         _T_7;	// Reg.scala:34:16
  reg         _T_8;	// Reg.scala:34:16
  reg         _T_9;	// Reg.scala:34:16
  reg         _T_10;	// Reg.scala:34:16
  reg         _T_11;	// Reg.scala:34:16
  reg         _T_12;	// Reg.scala:34:16
  reg         _T_13;	// Reg.scala:34:16
  reg         _T_14;	// Reg.scala:34:16
  reg         _T_15;	// Reg.scala:34:16
  reg  [2:0]  _T_16;	// Reg.scala:34:16
  reg  [1:0]  _T_17;	// Reg.scala:34:16
  reg  [64:0] _T_18;	// Reg.scala:34:16
  reg  [64:0] _T_19;	// Reg.scala:34:16
  reg  [64:0] _T_20;	// Reg.scala:34:16
  reg         _T_21_22;	// Valid.scala:47:18
  reg  [64:0] _T_22;	// Reg.scala:34:16
  reg  [4:0]  _T_23;	// Reg.scala:34:16

  wire _T_17_0 = _T_17[0];	// FPU.scala:374:31, Reg.scala:35:23
  wire [63:0] _T_21 = _T_17[1] ? _T_18[63:0] : {{32{~_T_17_0 & _T_18[31]}}, _T_18[31:0]};	// FPU.scala:372:33, :374:{13,19}, Package.scala:44:13, Reg.scala:35:23
  wire [1:0] _T_16_1to0 = _T_16[1:0];	// FPU.scala:384:25, Reg.scala:35:23
  `ifndef SYNTHESIS	// Valid.scala:47:18
    `ifdef RANDOMIZE_REG_INIT	// Valid.scala:47:18
      reg [31:0] _RANDOM;	// Valid.scala:47:18
      reg [31:0] _RANDOM_24;	// Reg.scala:34:16
      reg [31:0] _RANDOM_25;	// Reg.scala:34:16
      reg [31:0] _RANDOM_26;	// Reg.scala:34:16
      reg [31:0] _RANDOM_27;	// Reg.scala:34:16
      reg [31:0] _RANDOM_28;	// Reg.scala:34:16
      reg [31:0] _RANDOM_29;	// Reg.scala:34:16
      reg [31:0] _RANDOM_30;	// Reg.scala:34:16
      reg [31:0] _RANDOM_31;	// Reg.scala:34:16
      reg [31:0] _RANDOM_32;	// Reg.scala:34:16

    `endif
    initial begin	// Valid.scala:47:18
      `INIT_RANDOM_PROLOG_	// Valid.scala:47:18
      `ifdef RANDOMIZE_REG_INIT	// Valid.scala:47:18
        _RANDOM = `RANDOM;	// Valid.scala:47:18
        _T = _RANDOM[0];	// Valid.scala:47:18
        _T_0 = _RANDOM[5:1];	// Reg.scala:34:16
        _T_1 = _RANDOM[6];	// Reg.scala:34:16
        _T_2 = _RANDOM[7];	// Reg.scala:34:16
        _T_3 = _RANDOM[8];	// Reg.scala:34:16
        _T_4 = _RANDOM[9];	// Reg.scala:34:16
        _T_5 = _RANDOM[10];	// Reg.scala:34:16
        _T_6 = _RANDOM[11];	// Reg.scala:34:16
        _T_7 = _RANDOM[12];	// Reg.scala:34:16
        _T_8 = _RANDOM[13];	// Reg.scala:34:16
        _T_9 = _RANDOM[14];	// Reg.scala:34:16
        _T_10 = _RANDOM[15];	// Reg.scala:34:16
        _T_11 = _RANDOM[16];	// Reg.scala:34:16
        _T_12 = _RANDOM[17];	// Reg.scala:34:16
        _T_13 = _RANDOM[18];	// Reg.scala:34:16
        _T_14 = _RANDOM[19];	// Reg.scala:34:16
        _T_15 = _RANDOM[20];	// Reg.scala:34:16
        _T_16 = _RANDOM[23:21];	// Reg.scala:34:16
        _T_17 = _RANDOM[25:24];	// Reg.scala:34:16
        _RANDOM_24 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_25 = `RANDOM;	// Reg.scala:34:16
        _T_18 = {_RANDOM_25[26:0], _RANDOM_24, _RANDOM[31:26]};	// Reg.scala:34:16
        _RANDOM_26 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_27 = `RANDOM;	// Reg.scala:34:16
        _T_19 = {_RANDOM_27[27:0], _RANDOM_26, _RANDOM_25[31:27]};	// Reg.scala:34:16
        _RANDOM_28 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_29 = `RANDOM;	// Reg.scala:34:16
        _T_20 = {_RANDOM_29[28:0], _RANDOM_28, _RANDOM_27[31:28]};	// Reg.scala:34:16
        _T_21 = _RANDOM_29[29];	// Valid.scala:47:18
        _RANDOM_30 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_31 = `RANDOM;	// Reg.scala:34:16
        _T_22 = {_RANDOM_31[30:0], _RANDOM_30, _RANDOM_29[31:30]};	// Reg.scala:34:16
        _RANDOM_32 = `RANDOM;	// Reg.scala:34:16
        _T_23 = {_RANDOM_32[3:0], _RANDOM_31[31]};	// Reg.scala:34:16
      `endif
    end // initial
  `endif
      wire [7:0] _T_18_30to23 = _T_18[30:23];	// Reg.scala:35:23, recFNFromFN.scala:48:23
      wire [22:0] _T_18_22to0 = _T_18[22:0];	// Reg.scala:35:23, recFNFromFN.scala:49:25
      wire _T_24 = _T_18_30to23 == 8'h0;	// recFNFromFN.scala:51:34
      wire [15:0] _T_18_22to7 = _T_18[22:7];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [7:0] _T_18_22to15 = _T_18[22:15];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_22to19 = _T_18[22:19];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_14to11 = _T_18[14:11];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [6:0] _T_18_6to0 = _T_18[6:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_6to3 = _T_18[6:3];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_25 = |_T_18_22to7 ? {|_T_18_22to15, |_T_18_22to15 ? {|_T_18_22to19, |_T_18_22to19 ? (_T_18[22] ?
                2'h3 : _T_18[21] ? 2'h2 : {1'h0, _T_18[20]}) : _T_18[18] ? 2'h3 : _T_18[17] ? 2'h2 : {1'h0,
                _T_18[16]}} : {|_T_18_14to11, |_T_18_14to11 ? (_T_18[14] ? 2'h3 : _T_18[13] ? 2'h2 : {1'h0,
                _T_18[12]}) : _T_18[10] ? 2'h3 : _T_18[9] ? 2'h2 : {1'h0, _T_18[8]}}} : {|_T_18_6to0,
                |_T_18_6to0 ? {|_T_18_6to3, |_T_18_6to3 ? (_T_18[6] ? 2'h3 : _T_18[5] ? 2'h2 : {1'h0,
                _T_18[4]}) : _T_18[2] ? 2'h3 : _T_18[1] ? 2'h2 : {1'h0, _T_18[0]}} : 3'h0};	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, Reg.scala:35:23, Valid.scala:47:18
      wire [8:0] _T_26 = (_T_24 ? {4'hF, |_T_18_22to7, _T_25} : {1'h0, _T_18_30to23}) + {7'h20, _T_24 ? 2'h2 : 2'h1};	// CircuitMath.scala:32:10, :37:22, Valid.scala:47:18, recFNFromFN.scala:61:16, :62:27, :64:{15,47}
      wire [10:0] _T_18_62to52 = _T_18[62:52];	// Reg.scala:35:23, recFNFromFN.scala:48:23
      wire [51:0] _T_18_51to0 = _T_18[51:0];	// Reg.scala:35:23, recFNFromFN.scala:49:25
      wire _T_27 = _T_18_62to52 == 11'h0;	// recFNFromFN.scala:51:34
      wire [31:0] _T_18_51to20 = _T_18[51:20];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [15:0] _T_18_51to36 = _T_18[51:36];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [7:0] _T_18_51to44 = _T_18[51:44];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_51to48 = _T_18[51:48];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_43to40 = _T_18[43:40];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [7:0] _T_18_35to28 = _T_18[35:28];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_35to32 = _T_18[35:32];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_27to24 = _T_18[27:24];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [15:0] _T_18_19to4 = _T_18[19:4];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [7:0] _T_18_19to12 = _T_18[19:12];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_19to16 = _T_18[19:16];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_11to8 = _T_18[11:8];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_3to0 = _T_18[3:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [3:0] _T_18_3to0_28 = _T_18[3:0];	// CircuitMath.scala:35:17, Reg.scala:35:23
      wire [4:0] _T_29 = |_T_18_51to20 ? {|_T_18_51to36, |_T_18_51to36 ? {|_T_18_51to44, |_T_18_51to44 ?
                {|_T_18_51to48, |_T_18_51to48 ? (_T_18[51] ? 2'h3 : _T_18[50] ? 2'h2 : {1'h0, _T_18[49]}) :
                _T_18[47] ? 2'h3 : _T_18[46] ? 2'h2 : {1'h0, _T_18[45]}} : {|_T_18_43to40, |_T_18_43to40 ?
                (_T_18[43] ? 2'h3 : _T_18[42] ? 2'h2 : {1'h0, _T_18[41]}) : _T_18[39] ? 2'h3 : _T_18[38] ?
                2'h2 : {1'h0, _T_18[37]}}} : {|_T_18_35to28, |_T_18_35to28 ? {|_T_18_35to32, |_T_18_35to32
                ? (_T_18[35] ? 2'h3 : _T_18[34] ? 2'h2 : {1'h0, _T_18[33]}) : _T_18[31] ? 2'h3 : _T_18[30]
                ? 2'h2 : {1'h0, _T_18[29]}} : {|_T_18_27to24, |_T_18_27to24 ? (_T_18[27] ? 2'h3 : _T_18[26]
                ? 2'h2 : {1'h0, _T_18[25]}) : _T_18[23] ? 2'h3 : _T_18[22] ? 2'h2 : {1'h0, _T_18[21]}}}} :
                {|_T_18_19to4, |_T_18_19to4 ? {|_T_18_19to12, |_T_18_19to12 ? {|_T_18_19to16, |_T_18_19to16
                ? (_T_18[19] ? 2'h3 : _T_18[18] ? 2'h2 : {1'h0, _T_18[17]}) : _T_18[15] ? 2'h3 : _T_18[14]
                ? 2'h2 : {1'h0, _T_18[13]}} : {|_T_18_11to8, |_T_18_11to8 ? (_T_18[11] ? 2'h3 : _T_18[10] ?
                2'h2 : {1'h0, _T_18[9]}) : _T_18[7] ? 2'h3 : _T_18[6] ? 2'h2 : {1'h0, _T_18[5]}}} :
                {|_T_18_3to0, |_T_18_3to0 ? {|_T_18_3to0_28, |_T_18_3to0_28 ? (_T_18[3] ? 2'h3 : _T_18[2] ?
                2'h2 : {1'h0, _T_18[1]}) : 2'h0} : 3'h0}};	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, Reg.scala:35:23, Valid.scala:47:18, recFNFromFN.scala:64:15
      wire [11:0] _T_30 = (_T_27 ? {6'h3F, |_T_18_51to20, _T_29} : {1'h0, _T_18_62to52}) + {10'h100, _T_27 ? 2'h2 :
                2'h1};	// CircuitMath.scala:32:10, :37:22, Valid.scala:47:18, recFNFromFN.scala:56:13, :61:16, :62:27, :64:{15,47}
      wire _T_0_2 = _T_0[2];	// FPU.scala:380:21, Reg.scala:35:23
      wire [53:0] _T_31 = {31'h0, _T_18_22to0} << ~{|_T_18_22to7, _T_25};	// Cat.scala:30:58, CircuitMath.scala:37:22, recFNFromFN.scala:56:13, :58:25
      wire [114:0] _T_32 = {63'h0, _T_18_51to0} << ~{|_T_18_51to20, _T_29};	// Cat.scala:30:58, CircuitMath.scala:37:22, recFNFromFN.scala:56:13, :58:25
  always @(posedge clock) begin	// Reg.scala:35:23
    if (reset) begin	// Valid.scala:47:18
      _T <= 1'h0;	// Valid.scala:47:18
      _T_21_22 <= 1'h0;	// Valid.scala:47:18
    end
    else begin	// Valid.scala:47:18
      _T <= io_in_valid;	// Valid.scala:47:18
      _T_21_22 <= _T;	// Valid.scala:43:17, :47:18
    end
    if (io_in_valid) begin	// Reg.scala:35:23
      _T_0 <= io_in_bits_cmd;	// Reg.scala:35:23
      _T_1 <= io_in_bits_ldst;	// Reg.scala:35:23
      _T_2 <= io_in_bits_wen;	// Reg.scala:35:23
      _T_3 <= io_in_bits_ren1;	// Reg.scala:35:23
      _T_4 <= io_in_bits_ren2;	// Reg.scala:35:23
      _T_5 <= io_in_bits_ren3;	// Reg.scala:35:23
      _T_6 <= io_in_bits_swap12;	// Reg.scala:35:23
      _T_7 <= io_in_bits_swap23;	// Reg.scala:35:23
      _T_8 <= io_in_bits_single;	// Reg.scala:35:23
      _T_9 <= io_in_bits_fromint;	// Reg.scala:35:23
      _T_10 <= io_in_bits_toint;	// Reg.scala:35:23
      _T_11 <= io_in_bits_fastpipe;	// Reg.scala:35:23
      _T_12 <= io_in_bits_fma;	// Reg.scala:35:23
      _T_13 <= io_in_bits_div;	// Reg.scala:35:23
      _T_14 <= io_in_bits_sqrt;	// Reg.scala:35:23
      _T_15 <= io_in_bits_wflags;	// Reg.scala:35:23
      _T_16 <= io_in_bits_rm;	// Reg.scala:35:23
      _T_17 <= io_in_bits_typ;	// Reg.scala:35:23
      _T_18 <= io_in_bits_in1;	// Reg.scala:35:23
      _T_19 <= io_in_bits_in2;	// Reg.scala:35:23
      _T_20 <= io_in_bits_in3;	// Reg.scala:35:23
    end
    if (_T) begin	// Reg.scala:35:23, Valid.scala:43:17
      _T_22 <= _T_0_2 ? (_T_8 ? {32'h0, _T_18[31], _T_26 & {~{3{_T_24 & ~(|_T_18_22to0)}}, 6'h3F} | {2'h0,
                                                &(_T_26[8:7]) & |_T_18_22to0, 6'h0}, _T_24 ? {_T_31[21:0], 1'h0} : _T_18_22to0} :
                                                {_T_18[63], _T_30 & {~{3{_T_27 & ~(|_T_18_51to0)}}, 9'h1FF} | {2'h0, &(_T_30[11:10]) &
                                                |_T_18_51to0, 9'h0}, _T_27 ? {_T_32[50:0], 1'h0} : _T_18_51to0}) : _T_8 ?
                                                {INToRecFN_1_io_out[64:33], INToRecFN_io_out} : INToRecFN_1_io_out;	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:37:22, FPU.scala:364:14, :381:21, :391:25, :396:36, :399:20, Reg.scala:35:23, Valid.scala:47:18, recFNFromFN.scala:47:22, :52:38, :53:34, :56:{13,26}, :58:37, :64:{15,42}, :67:{25,50,63}, :71:{26,28,45,64}, :73:27
      _T_23 <= _T_0_2 ? 5'h0 : _T_8 ? INToRecFN_io_exceptionFlags : INToRecFN_1_io_exceptionFlags;	// FPU.scala:381:21, :391:25, :400:19, Reg.scala:35:23
    end
  end // always @(posedge)
  INToRecFN INToRecFN (	// FPU.scala:381:21
    .io_signedIn       (~_T_17_0),	// FPU.scala:382:24
    .io_in             (_T_21),
    .io_roundingMode   (_T_16_1to0),
    .io_out            (INToRecFN_io_out),
    .io_exceptionFlags (INToRecFN_io_exceptionFlags)
  );
  INToRecFN_1 INToRecFN_1 (	// FPU.scala:391:25
    .io_signedIn       (~_T_17_0),	// FPU.scala:382:24
    .io_in             (_T_21),
    .io_roundingMode   (_T_16_1to0),
    .io_out            (INToRecFN_1_io_out),
    .io_exceptionFlags (INToRecFN_1_io_exceptionFlags)
  );
  assign io_out_valid = _T_21_22;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1308:10, Valid.scala:43:17
  assign io_out_bits_data = _T_22;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1308:10, Reg.scala:35:23
  assign io_out_bits_exc = _T_23;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1308:10, Reg.scala:35:23
endmodule

module FPToFP(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1685:10
  input         clock, reset, io_in_valid,
  input  [4:0]  io_in_bits_cmd,
  input         io_in_bits_ldst, io_in_bits_wen, io_in_bits_ren1, io_in_bits_ren2,
  input         io_in_bits_ren3, io_in_bits_swap12, io_in_bits_swap23, io_in_bits_single,
  input         io_in_bits_fromint, io_in_bits_toint, io_in_bits_fastpipe,
  input         io_in_bits_fma, io_in_bits_div, io_in_bits_sqrt, io_in_bits_wflags,
  input  [2:0]  io_in_bits_rm,
  input  [1:0]  io_in_bits_typ,
  input  [64:0] io_in_bits_in1, io_in_bits_in2, io_in_bits_in3,
  input         io_lt,
  output        io_out_valid,
  output [64:0] io_out_bits_data,
  output [4:0]  io_out_bits_exc);

  wire [64:0] RecFNToRecFN_1_io_out;	// FPU.scala:455:25
  wire [4:0]  RecFNToRecFN_1_io_exceptionFlags;	// FPU.scala:455:25
  wire [32:0] RecFNToRecFN_io_out;	// FPU.scala:451:25
  wire [4:0]  RecFNToRecFN_io_exceptionFlags;	// FPU.scala:451:25
  reg         _T;	// Valid.scala:47:18
  reg  [4:0]  _T_0;	// Reg.scala:34:16
  reg         _T_1;	// Reg.scala:34:16
  reg         _T_2;	// Reg.scala:34:16
  reg         _T_3;	// Reg.scala:34:16
  reg         _T_4;	// Reg.scala:34:16
  reg         _T_5;	// Reg.scala:34:16
  reg         _T_6;	// Reg.scala:34:16
  reg         _T_7;	// Reg.scala:34:16
  reg         _T_8;	// Reg.scala:34:16
  reg         _T_9;	// Reg.scala:34:16
  reg         _T_10;	// Reg.scala:34:16
  reg         _T_11;	// Reg.scala:34:16
  reg         _T_12;	// Reg.scala:34:16
  reg         _T_13;	// Reg.scala:34:16
  reg         _T_14;	// Reg.scala:34:16
  reg         _T_15;	// Reg.scala:34:16
  reg  [2:0]  _T_16;	// Reg.scala:34:16
  reg  [1:0]  _T_17;	// Reg.scala:34:16
  reg  [64:0] _T_18;	// Reg.scala:34:16
  reg  [64:0] _T_19;	// Reg.scala:34:16
  reg  [64:0] _T_20;	// Reg.scala:34:16
  reg         _T_21;	// Valid.scala:47:18
  reg  [64:0] _T_22;	// Reg.scala:34:16
  reg  [4:0]  _T_23;	// Reg.scala:34:16

  `ifndef SYNTHESIS	// Valid.scala:47:18
    `ifdef RANDOMIZE_REG_INIT	// Valid.scala:47:18
      reg [31:0] _RANDOM;	// Valid.scala:47:18
      reg [31:0] _RANDOM_24;	// Reg.scala:34:16
      reg [31:0] _RANDOM_25;	// Reg.scala:34:16
      reg [31:0] _RANDOM_26;	// Reg.scala:34:16
      reg [31:0] _RANDOM_27;	// Reg.scala:34:16
      reg [31:0] _RANDOM_28;	// Reg.scala:34:16
      reg [31:0] _RANDOM_29;	// Reg.scala:34:16
      reg [31:0] _RANDOM_30;	// Reg.scala:34:16
      reg [31:0] _RANDOM_31;	// Reg.scala:34:16
      reg [31:0] _RANDOM_32;	// Reg.scala:34:16

    `endif
    initial begin	// Valid.scala:47:18
      `INIT_RANDOM_PROLOG_	// Valid.scala:47:18
      `ifdef RANDOMIZE_REG_INIT	// Valid.scala:47:18
        _RANDOM = `RANDOM;	// Valid.scala:47:18
        _T = _RANDOM[0];	// Valid.scala:47:18
        _T_0 = _RANDOM[5:1];	// Reg.scala:34:16
        _T_1 = _RANDOM[6];	// Reg.scala:34:16
        _T_2 = _RANDOM[7];	// Reg.scala:34:16
        _T_3 = _RANDOM[8];	// Reg.scala:34:16
        _T_4 = _RANDOM[9];	// Reg.scala:34:16
        _T_5 = _RANDOM[10];	// Reg.scala:34:16
        _T_6 = _RANDOM[11];	// Reg.scala:34:16
        _T_7 = _RANDOM[12];	// Reg.scala:34:16
        _T_8 = _RANDOM[13];	// Reg.scala:34:16
        _T_9 = _RANDOM[14];	// Reg.scala:34:16
        _T_10 = _RANDOM[15];	// Reg.scala:34:16
        _T_11 = _RANDOM[16];	// Reg.scala:34:16
        _T_12 = _RANDOM[17];	// Reg.scala:34:16
        _T_13 = _RANDOM[18];	// Reg.scala:34:16
        _T_14 = _RANDOM[19];	// Reg.scala:34:16
        _T_15 = _RANDOM[20];	// Reg.scala:34:16
        _T_16 = _RANDOM[23:21];	// Reg.scala:34:16
        _T_17 = _RANDOM[25:24];	// Reg.scala:34:16
        _RANDOM_24 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_25 = `RANDOM;	// Reg.scala:34:16
        _T_18 = {_RANDOM_25[26:0], _RANDOM_24, _RANDOM[31:26]};	// Reg.scala:34:16
        _RANDOM_26 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_27 = `RANDOM;	// Reg.scala:34:16
        _T_19 = {_RANDOM_27[27:0], _RANDOM_26, _RANDOM_25[31:27]};	// Reg.scala:34:16
        _RANDOM_28 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_29 = `RANDOM;	// Reg.scala:34:16
        _T_20 = {_RANDOM_29[28:0], _RANDOM_28, _RANDOM_27[31:28]};	// Reg.scala:34:16
        _T_21 = _RANDOM_29[29];	// Valid.scala:47:18
        _RANDOM_30 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_31 = `RANDOM;	// Reg.scala:34:16
        _T_22 = {_RANDOM_31[30:0], _RANDOM_30, _RANDOM_29[31:30]};	// Reg.scala:34:16
        _RANDOM_32 = `RANDOM;	// Reg.scala:34:16
        _T_23 = {_RANDOM_32[3:0], _RANDOM_31[31]};	// Reg.scala:34:16
      `endif
    end // initial
  `endif
      wire _T_16_0 = _T_16[0];	// FPU.scala:417:79, Reg.scala:35:23
      wire [32:0] _T_24 = _T_16[1] ? _T_18[64:32] ^ _T_19[64:32] : {33{_T_16_0}} ^ _T_19[64:32];	// FPU.scala:417:{22,33,50,68}, Reg.scala:35:23
      wire _T_25 = {_T_0[3:2], _T_0[0]} == 3'h3;	// FPU.scala:428:23, Reg.scala:35:23
      wire _T_26 = _T_18[31:29] != 3'h7;	// FPU.scala:226:{7,58}, Reg.scala:35:23
      wire [2:0] _T_19_31to29 = _T_19[31:29];	// FPU.scala:226:7, Reg.scala:35:23
      wire _T_27 = ~_T_26 & ~(_T_18[22]) | &_T_19_31to29 & ~(_T_19[22]);	// FPU.scala:226:58, :231:{40,43,46}, :434:31, Reg.scala:35:23
      wire _T_28 = _T_16_0 != io_lt;	// FPU.scala:437:34
      wire _T_29 = _T_18[63:61] != 3'h7;	// FPU.scala:226:{7,58}, Reg.scala:35:23
      wire [2:0] _T_19_63to61 = _T_19[63:61];	// FPU.scala:226:7, Reg.scala:35:23
      wire _T_30 = ~_T_29 & ~(_T_18[51]) | &_T_19_63to61 & ~(_T_19[51]);	// FPU.scala:226:58, :231:{40,43,46}, :434:31, Reg.scala:35:23
      wire _T_0_2 = _T_0[2];	// FPU.scala:450:25, Reg.scala:35:23
  always @(posedge clock) begin	// Reg.scala:35:23
    if (reset) begin	// Valid.scala:47:18
      _T <= 1'h0;	// Valid.scala:47:18
      _T_21 <= 1'h0;	// Valid.scala:47:18
    end
    else begin	// Valid.scala:47:18
      _T <= io_in_valid;	// Valid.scala:47:18
      _T_21 <= _T;	// Valid.scala:43:17, :47:18
    end
    if (io_in_valid) begin	// Reg.scala:35:23
      _T_0 <= io_in_bits_cmd;	// Reg.scala:35:23
      _T_1 <= io_in_bits_ldst;	// Reg.scala:35:23
      _T_2 <= io_in_bits_wen;	// Reg.scala:35:23
      _T_3 <= io_in_bits_ren1;	// Reg.scala:35:23
      _T_4 <= io_in_bits_ren2;	// Reg.scala:35:23
      _T_5 <= io_in_bits_ren3;	// Reg.scala:35:23
      _T_6 <= io_in_bits_swap12;	// Reg.scala:35:23
      _T_7 <= io_in_bits_swap23;	// Reg.scala:35:23
      _T_8 <= io_in_bits_single;	// Reg.scala:35:23
      _T_9 <= io_in_bits_fromint;	// Reg.scala:35:23
      _T_10 <= io_in_bits_toint;	// Reg.scala:35:23
      _T_11 <= io_in_bits_fastpipe;	// Reg.scala:35:23
      _T_12 <= io_in_bits_fma;	// Reg.scala:35:23
      _T_13 <= io_in_bits_div;	// Reg.scala:35:23
      _T_14 <= io_in_bits_sqrt;	// Reg.scala:35:23
      _T_15 <= io_in_bits_wflags;	// Reg.scala:35:23
      _T_16 <= io_in_bits_rm;	// Reg.scala:35:23
      _T_17 <= io_in_bits_typ;	// Reg.scala:35:23
      _T_18 <= io_in_bits_in1;	// Reg.scala:35:23
      _T_19 <= io_in_bits_in2;	// Reg.scala:35:23
      _T_20 <= io_in_bits_in3;	// Reg.scala:35:23
    end
    if (_T) begin	// Reg.scala:35:23, Valid.scala:43:17
      _T_22 <= _T_0_2 ? (_T_25 ? ((_T_8 ? _T_27 | ~_T_26 & &_T_19_31to29 : _T_30 | ~_T_29 & &_T_19_63to61)
                                                ? (_T_8 ? 65'hE0080000E0400000 : 65'hE008000000000000) : (_T_8 ? &_T_19_31to29 | _T_28 &
                                                _T_26 : &_T_19_63to61 | _T_28 & _T_29) ? _T_18 : _T_19) : _T_8 ? {_T_18[64:33], _T_24[0],
                                                _T_18[31:0]} : {_T_24[32], _T_18[63:0]}) : _T_8 ? {RecFNToRecFN_1_io_out[64:33],
                                                RecFNToRecFN_io_out} : RecFNToRecFN_1_io_out;	// Cat.scala:30:58, FPU.scala:226:58, :418:{30,47}, :421:{21,54}, :422:{49,66}, :435:{32,43}, :436:100, :437:{17,44}, :444:{16,22,42}, :451:25, :455:25, :460:38, :463:20, Misc.scala:42:{9,63,90}, Reg.scala:35:23
      _T_23 <= _T_0_2 ? (_T_25 ? {_T_8 ? _T_27 : _T_30, 4'h0} : 5'h0) : _T_8 ?
                                                RecFNToRecFN_io_exceptionFlags : RecFNToRecFN_1_io_exceptionFlags;	// FPU.scala:443:{15,28}, :451:25, :455:25, :464:19, Misc.scala:42:36, Reg.scala:35:23
    end
  end // always @(posedge)
  RecFNToRecFN RecFNToRecFN (	// FPU.scala:451:25
    .io_in             (_T_18),	// Reg.scala:35:23
    .io_roundingMode   (_T_16[1:0]),	// FPU.scala:453:29, Reg.scala:35:23
    .io_out            (RecFNToRecFN_io_out),
    .io_exceptionFlags (RecFNToRecFN_io_exceptionFlags)
  );
  RecFNToRecFN_1 RecFNToRecFN_1 (	// FPU.scala:455:25
    .io_in             (_T_18[32:0]),	// FPU.scala:456:19, Reg.scala:35:23
    .io_out            (RecFNToRecFN_1_io_out),
    .io_exceptionFlags (RecFNToRecFN_1_io_exceptionFlags)
  );
  assign io_out_valid = _T_21;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1685:10, Valid.scala:43:17
  assign io_out_bits_data = _T_22;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1685:10, Reg.scala:35:23
  assign io_out_bits_exc = _T_23;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1685:10, Reg.scala:35:23
endmodule

module FPUFMAPipe_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1827:10
  input         clock, reset, io_in_valid,
  input  [4:0]  io_in_bits_cmd,
  input         io_in_bits_ldst, io_in_bits_wen, io_in_bits_ren1, io_in_bits_ren2,
  input         io_in_bits_ren3, io_in_bits_swap12, io_in_bits_swap23, io_in_bits_single,
  input         io_in_bits_fromint, io_in_bits_toint, io_in_bits_fastpipe,
  input         io_in_bits_fma, io_in_bits_div, io_in_bits_sqrt, io_in_bits_wflags,
  input  [2:0]  io_in_bits_rm,
  input  [1:0]  io_in_bits_typ,
  input  [64:0] io_in_bits_in1, io_in_bits_in2, io_in_bits_in3,
  output        io_out_valid,
  output [64:0] io_out_bits_data,
  output [4:0]  io_out_bits_exc);

  wire [64:0] fma_io_out;	// FPU.scala:493:19
  wire [4:0]  fma_io_exceptionFlags;	// FPU.scala:493:19
  reg         valid;	// FPU.scala:482:18
  reg  [4:0]  in_cmd;	// FPU.scala:483:15
  reg         in_ldst;	// FPU.scala:483:15
  reg         in_wen;	// FPU.scala:483:15
  reg         in_ren1;	// FPU.scala:483:15
  reg         in_ren2;	// FPU.scala:483:15
  reg         in_ren3;	// FPU.scala:483:15
  reg         in_swap12;	// FPU.scala:483:15
  reg         in_swap23;	// FPU.scala:483:15
  reg         in_single;	// FPU.scala:483:15
  reg         in_fromint;	// FPU.scala:483:15
  reg         in_toint;	// FPU.scala:483:15
  reg         in_fastpipe;	// FPU.scala:483:15
  reg         in_fma;	// FPU.scala:483:15
  reg         in_div;	// FPU.scala:483:15
  reg         in_sqrt;	// FPU.scala:483:15
  reg         in_wflags;	// FPU.scala:483:15
  reg  [2:0]  in_rm;	// FPU.scala:483:15
  reg  [1:0]  in_typ;	// FPU.scala:483:15
  reg  [64:0] in_in1;	// FPU.scala:483:15
  reg  [64:0] in_in2;	// FPU.scala:483:15
  reg  [64:0] in_in3;	// FPU.scala:483:15
  reg         _T;	// Valid.scala:47:18
  reg  [64:0] _T_0;	// Reg.scala:34:16
  reg  [4:0]  _T_1;	// Reg.scala:34:16
  reg         _T_2;	// Valid.scala:47:18
  reg  [64:0] _T_3;	// Reg.scala:34:16
  reg  [4:0]  _T_4;	// Reg.scala:34:16
  reg         _T_5;	// Valid.scala:47:18
  reg  [64:0] _T_6;	// Reg.scala:34:16
  reg  [4:0]  _T_7;	// Reg.scala:34:16

  `ifndef SYNTHESIS	// FPU.scala:482:18
    `ifdef RANDOMIZE_REG_INIT	// FPU.scala:482:18
      reg [31:0] _RANDOM;	// FPU.scala:482:18
      reg [31:0] _RANDOM_8;	// FPU.scala:483:15
      reg [31:0] _RANDOM_9;	// FPU.scala:483:15
      reg [31:0] _RANDOM_10;	// FPU.scala:483:15
      reg [31:0] _RANDOM_11;	// FPU.scala:483:15
      reg [31:0] _RANDOM_12;	// FPU.scala:483:15
      reg [31:0] _RANDOM_13;	// FPU.scala:483:15
      reg [31:0] _RANDOM_14;	// Reg.scala:34:16
      reg [31:0] _RANDOM_15;	// Reg.scala:34:16
      reg [31:0] _RANDOM_16;	// Reg.scala:34:16
      reg [31:0] _RANDOM_17;	// Reg.scala:34:16
      reg [31:0] _RANDOM_18;	// Reg.scala:34:16
      reg [31:0] _RANDOM_19;	// Reg.scala:34:16
      reg [31:0] _RANDOM_20;	// Reg.scala:34:16

    `endif
    initial begin	// FPU.scala:482:18
      `INIT_RANDOM_PROLOG_	// FPU.scala:482:18
      `ifdef RANDOMIZE_REG_INIT	// FPU.scala:482:18
        _RANDOM = `RANDOM;	// FPU.scala:482:18
        valid = _RANDOM[0];	// FPU.scala:482:18
        in_cmd = _RANDOM[5:1];	// FPU.scala:483:15
        in_ldst = _RANDOM[6];	// FPU.scala:483:15
        in_wen = _RANDOM[7];	// FPU.scala:483:15
        in_ren1 = _RANDOM[8];	// FPU.scala:483:15
        in_ren2 = _RANDOM[9];	// FPU.scala:483:15
        in_ren3 = _RANDOM[10];	// FPU.scala:483:15
        in_swap12 = _RANDOM[11];	// FPU.scala:483:15
        in_swap23 = _RANDOM[12];	// FPU.scala:483:15
        in_single = _RANDOM[13];	// FPU.scala:483:15
        in_fromint = _RANDOM[14];	// FPU.scala:483:15
        in_toint = _RANDOM[15];	// FPU.scala:483:15
        in_fastpipe = _RANDOM[16];	// FPU.scala:483:15
        in_fma = _RANDOM[17];	// FPU.scala:483:15
        in_div = _RANDOM[18];	// FPU.scala:483:15
        in_sqrt = _RANDOM[19];	// FPU.scala:483:15
        in_wflags = _RANDOM[20];	// FPU.scala:483:15
        in_rm = _RANDOM[23:21];	// FPU.scala:483:15
        in_typ = _RANDOM[25:24];	// FPU.scala:483:15
        _RANDOM_8 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_9 = `RANDOM;	// FPU.scala:483:15
        in_in1 = {_RANDOM_9[26:0], _RANDOM_8, _RANDOM[31:26]};	// FPU.scala:483:15
        _RANDOM_10 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_11 = `RANDOM;	// FPU.scala:483:15
        in_in2 = {_RANDOM_11[27:0], _RANDOM_10, _RANDOM_9[31:27]};	// FPU.scala:483:15
        _RANDOM_12 = `RANDOM;	// FPU.scala:483:15
        _RANDOM_13 = `RANDOM;	// FPU.scala:483:15
        in_in3 = {_RANDOM_13[28:0], _RANDOM_12, _RANDOM_11[31:28]};	// FPU.scala:483:15
        _T = _RANDOM_13[29];	// Valid.scala:47:18
        _RANDOM_14 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_15 = `RANDOM;	// Reg.scala:34:16
        _T_0 = {_RANDOM_15[30:0], _RANDOM_14, _RANDOM_13[31:30]};	// Reg.scala:34:16
        _RANDOM_16 = `RANDOM;	// Reg.scala:34:16
        _T_1 = {_RANDOM_16[3:0], _RANDOM_15[31]};	// Reg.scala:34:16
        _T_2 = _RANDOM_16[4];	// Valid.scala:47:18
        _RANDOM_17 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_18 = `RANDOM;	// Reg.scala:34:16
        _T_3 = {_RANDOM_18[5:0], _RANDOM_17, _RANDOM_16[31:5]};	// Reg.scala:34:16
        _T_4 = _RANDOM_18[10:6];	// Reg.scala:34:16
        _T_5 = _RANDOM_18[11];	// Valid.scala:47:18
        _RANDOM_19 = `RANDOM;	// Reg.scala:34:16
        _RANDOM_20 = `RANDOM;	// Reg.scala:34:16
        _T_6 = {_RANDOM_20[12:0], _RANDOM_19, _RANDOM_18[31:12]};	// Reg.scala:34:16
        _T_7 = _RANDOM_20[17:13];	// Reg.scala:34:16
      `endif
    end // initial
  `endif
      wire _T_8 = io_in_bits_ren3 | io_in_bits_swap23;	// FPU.scala:488:48
  always @(posedge clock) begin	// FPU.scala:482:18
    if (reset) begin	// Valid.scala:47:18
      _T <= 1'h0;	// Conditional.scala:19:11, Valid.scala:47:18
      _T_2 <= 1'h0;	// Conditional.scala:19:11, Valid.scala:47:18
      _T_5 <= 1'h0;	// Conditional.scala:19:11, Valid.scala:47:18
    end
    else begin	// Valid.scala:47:18
      _T <= valid;	// Valid.scala:47:18
      _T_2 <= _T;	// Valid.scala:47:18
      _T_5 <= _T_2;	// Valid.scala:47:18
    end
    valid <= io_in_valid;	// FPU.scala:482:18
    if (io_in_valid) begin	// FPU.scala:490:45
      in_ldst <= io_in_bits_ldst;	// FPU.scala:485:8
      in_wen <= io_in_bits_wen;	// FPU.scala:485:8
      in_ren1 <= io_in_bits_ren1;	// FPU.scala:485:8
      in_ren2 <= io_in_bits_ren2;	// FPU.scala:485:8
      in_ren3 <= io_in_bits_ren3;	// FPU.scala:485:8
      in_swap12 <= io_in_bits_swap12;	// FPU.scala:485:8
      in_swap23 <= io_in_bits_swap23;	// FPU.scala:485:8
      in_single <= io_in_bits_single;	// FPU.scala:485:8
      in_fromint <= io_in_bits_fromint;	// FPU.scala:485:8
      in_toint <= io_in_bits_toint;	// FPU.scala:485:8
      in_fastpipe <= io_in_bits_fastpipe;	// FPU.scala:485:8
      in_fma <= io_in_bits_fma;	// FPU.scala:485:8
      in_div <= io_in_bits_div;	// FPU.scala:485:8
      in_sqrt <= io_in_bits_sqrt;	// FPU.scala:485:8
      in_wflags <= io_in_bits_wflags;	// FPU.scala:485:8
      in_rm <= io_in_bits_rm;	// FPU.scala:485:8
      in_typ <= io_in_bits_typ;	// FPU.scala:485:8
      in_in1 <= io_in_bits_in1;	// FPU.scala:485:8
      in_cmd <= {3'h0, io_in_bits_cmd[1] & _T_8, io_in_bits_cmd[0]};	// FPU.scala:488:{12,33,37,78}
      in_in2 <= io_in_bits_swap23 ? 65'h8000000000000000 : io_in_bits_in2;	// FPU.scala:489:32
      in_in3 <= _T_8 ? io_in_bits_in3 : {io_in_bits_in1[64] ^ io_in_bits_in2[64], 64'h0};	// FPU.scala:480:{29,37,53,62}, :490:45
    end
    if (valid) begin	// Reg.scala:35:23, Valid.scala:47:18
      _T_0 <= fma_io_out;	// FPU.scala:493:19, Reg.scala:35:23
      _T_1 <= fma_io_exceptionFlags;	// FPU.scala:493:19, Reg.scala:35:23
    end
    if (_T) begin	// Reg.scala:35:23, Valid.scala:47:18
      _T_3 <= _T_0;	// Reg.scala:35:23
      _T_4 <= _T_1;	// Reg.scala:35:23
    end
    if (_T_2) begin	// Reg.scala:35:23, Valid.scala:47:18
      _T_6 <= _T_3;	// Reg.scala:35:23
      _T_7 <= _T_4;	// Reg.scala:35:23
    end
  end // always @(posedge)
  MulAddRecFN_1 fma (	// FPU.scala:493:19
    .io_op             (in_cmd[1:0]),	// FPU.scala:488:12, :494:13
    .io_a              (in_in1),	// FPU.scala:485:8
    .io_b              (in_in2),	// FPU.scala:489:32
    .io_c              (in_in3),	// FPU.scala:490:45
    .io_roundingMode   (in_rm[1:0]),	// FPU.scala:485:8, :495:23
    .io_out            (fma_io_out),
    .io_exceptionFlags (fma_io_exceptionFlags)
  );
  assign io_out_valid = _T_5;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1827:10, Valid.scala:43:17
  assign io_out_bits_data = _T_6;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1827:10, Reg.scala:35:23
  assign io_out_bits_exc = _T_7;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1827:10, Reg.scala:35:23
endmodule

module DivSqrtRecF64(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1896:10
  input         clock, reset, io_inValid, io_sqrtOp,
  input  [64:0] io_a, io_b,
  input  [1:0]  io_roundingMode,
  output        io_inReady_div, io_inReady_sqrt, io_outValid_div, io_outValid_sqrt,
  output [64:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [104:0] mul_io_result_s3;	// DivSqrtRecF64.scala:73:21
  wire [3:0]   ds_io_usingMulAdd;	// DivSqrtRecF64.scala:59:20
  wire         ds_io_latchMulAddA_0;	// DivSqrtRecF64.scala:59:20
  wire [53:0]  ds_io_mulAddA_0;	// DivSqrtRecF64.scala:59:20
  wire         ds_io_latchMulAddB_0;	// DivSqrtRecF64.scala:59:20
  wire [53:0]  ds_io_mulAddB_0;	// DivSqrtRecF64.scala:59:20
  wire [104:0] ds_io_mulAddC_2;	// DivSqrtRecF64.scala:59:20

  DivSqrtRecF64_mulAddZ31 ds (	// DivSqrtRecF64.scala:59:20
    .clock             (clock),
    .reset             (reset),
    .io_inValid        (io_inValid),
    .io_sqrtOp         (io_sqrtOp),
    .io_a              (io_a),
    .io_b              (io_b),
    .io_roundingMode   (io_roundingMode),
    .io_mulAddResult_3 (mul_io_result_s3),	// DivSqrtRecF64.scala:73:21
    .io_inReady_div    (io_inReady_div),
    .io_inReady_sqrt   (io_inReady_sqrt),
    .io_outValid_div   (io_outValid_div),
    .io_outValid_sqrt  (io_outValid_sqrt),
    .io_out            (io_out),
    .io_exceptionFlags (io_exceptionFlags),
    .io_usingMulAdd    (ds_io_usingMulAdd),
    .io_latchMulAddA_0 (ds_io_latchMulAddA_0),
    .io_mulAddA_0      (ds_io_mulAddA_0),
    .io_latchMulAddB_0 (ds_io_latchMulAddB_0),
    .io_mulAddB_0      (ds_io_mulAddB_0),
    .io_mulAddC_2      (ds_io_mulAddC_2)
  );
  Mul54 mul (	// DivSqrtRecF64.scala:73:21
    .clock         (clock),
    .io_val_s0     (ds_io_usingMulAdd[0]),	// DivSqrtRecF64.scala:59:20, :75:39
    .io_latch_a_s0 (ds_io_latchMulAddA_0),	// DivSqrtRecF64.scala:59:20
    .io_a_s0       (ds_io_mulAddA_0),	// DivSqrtRecF64.scala:59:20
    .io_latch_b_s0 (ds_io_latchMulAddB_0),	// DivSqrtRecF64.scala:59:20
    .io_b_s0       (ds_io_mulAddB_0),	// DivSqrtRecF64.scala:59:20
    .io_c_s2       (ds_io_mulAddC_2),	// DivSqrtRecF64.scala:59:20
    .io_result_s3  (mul_io_result_s3)
  );
endmodule

module RecFNToRecFN_2(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1931:10
  input  [64:0] io_in,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [2:0] _io_in_63to61 = io_in[63:61];	// rawFNFromRecFN.scala:51:29
  wire [1:0] _io_in_63to62 = io_in[63:62];	// rawFNFromRecFN.scala:52:29
  wire _io_in_61 = io_in[61];	// rawFNFromRecFN.scala:56:40
  wire _T = &_io_in_63to62 & _io_in_61;	// rawFNFromRecFN.scala:52:54, :56:32
  wire [13:0] _T_0 = {2'h0, io_in[63:52]} - 14'h700;	// Cat.scala:30:58, rawFNFromRecFN.scala:50:21, resizeRawFN.scala:49:31
  RoundRawFNToRecFN_1 RoundRawFNToRecFN (	// RecFNToRecFN.scala:102:19
    .io_invalidExc     (_T & ~(io_in[51])),	// RoundRawFNToRecFN.scala:61:{46,49,57}
    .io_in_sign        (io_in[64]),	// rawFNFromRecFN.scala:55:23
    .io_in_isNaN       (_T),
    .io_in_isInf       (&_io_in_63to62 & ~_io_in_61),	// rawFNFromRecFN.scala:52:54, :57:{32,35}
    .io_in_isZero      (~(|_io_in_63to61)),	// rawFNFromRecFN.scala:51:54
    .io_in_sExp        ({$signed(_T_0) < 14'sh0, |(_T_0[12:9]) ? 9'h1FC : _T_0[8:0]}),	// Cat.scala:30:58, resizeRawFN.scala:60:31, :61:{25,33,65}, :63:33
    .io_in_sig         ({1'h0, |_io_in_63to61, io_in[51:28], |(io_in[27:0])}),	// Cat.scala:30:58, rawFNFromRecFN.scala:51:54, resizeRawFN.scala:71:28, :72:{28,56}
    .io_roundingMode   (io_roundingMode),
    .io_out            (io_out),
    .io_exceptionFlags (io_exceptionFlags)
  );
endmodule

module MulAddRecFN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:1999:10
  input  [1:0]  io_op,
  input  [32:0] io_a, io_b, io_c,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [23:0] mulAddRecFN_preMul_io_mulAddA;	// MulAddRecFN.scala:598:15
  wire [23:0] mulAddRecFN_preMul_io_mulAddB;	// MulAddRecFN.scala:598:15
  wire [47:0] mulAddRecFN_preMul_io_mulAddC;	// MulAddRecFN.scala:598:15
  wire [2:0]  mulAddRecFN_preMul_io_toPostMul_highExpA;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA;	// MulAddRecFN.scala:598:15
  wire [2:0]  mulAddRecFN_preMul_io_toPostMul_highExpB;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_signProd;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_isZeroProd;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_opSignC;	// MulAddRecFN.scala:598:15
  wire [2:0]  mulAddRecFN_preMul_io_toPostMul_highExpC;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_isCDominant;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_CAlignDist_0;	// MulAddRecFN.scala:598:15
  wire [6:0]  mulAddRecFN_preMul_io_toPostMul_CAlignDist;	// MulAddRecFN.scala:598:15
  wire        mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC;	// MulAddRecFN.scala:598:15
  wire [25:0] mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC;	// MulAddRecFN.scala:598:15
  wire [10:0] mulAddRecFN_preMul_io_toPostMul_sExpSum;	// MulAddRecFN.scala:598:15
  wire [1:0]  mulAddRecFN_preMul_io_toPostMul_roundingMode;	// MulAddRecFN.scala:598:15

  MulAddRecFN_preMul mulAddRecFN_preMul (	// MulAddRecFN.scala:598:15
    .io_op                           (io_op),
    .io_a                            (io_a),
    .io_b                            (io_b),
    .io_c                            (io_c),
    .io_roundingMode                 (io_roundingMode),
    .io_mulAddA                      (mulAddRecFN_preMul_io_mulAddA),
    .io_mulAddB                      (mulAddRecFN_preMul_io_mulAddB),
    .io_mulAddC                      (mulAddRecFN_preMul_io_mulAddC),
    .io_toPostMul_highExpA           (mulAddRecFN_preMul_io_toPostMul_highExpA),
    .io_toPostMul_isNaN_isQuietNaNA  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA),
    .io_toPostMul_highExpB           (mulAddRecFN_preMul_io_toPostMul_highExpB),
    .io_toPostMul_isNaN_isQuietNaNB  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB),
    .io_toPostMul_signProd           (mulAddRecFN_preMul_io_toPostMul_signProd),
    .io_toPostMul_isZeroProd         (mulAddRecFN_preMul_io_toPostMul_isZeroProd),
    .io_toPostMul_opSignC            (mulAddRecFN_preMul_io_toPostMul_opSignC),
    .io_toPostMul_highExpC           (mulAddRecFN_preMul_io_toPostMul_highExpC),
    .io_toPostMul_isNaN_isQuietNaNC  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC),
    .io_toPostMul_isCDominant        (mulAddRecFN_preMul_io_toPostMul_isCDominant),
    .io_toPostMul_CAlignDist_0       (mulAddRecFN_preMul_io_toPostMul_CAlignDist_0),
    .io_toPostMul_CAlignDist         (mulAddRecFN_preMul_io_toPostMul_CAlignDist),
    .io_toPostMul_bit0AlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC),
    .io_toPostMul_highAlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC),
    .io_toPostMul_sExpSum            (mulAddRecFN_preMul_io_toPostMul_sExpSum),
    .io_toPostMul_roundingMode       (mulAddRecFN_preMul_io_toPostMul_roundingMode)
  );
  MulAddRecFN_postMul mulAddRecFN_postMul (	// MulAddRecFN.scala:600:15
    .io_fromPreMul_highExpA           (mulAddRecFN_preMul_io_toPostMul_highExpA),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNA  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highExpB           (mulAddRecFN_preMul_io_toPostMul_highExpB),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNB  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_signProd           (mulAddRecFN_preMul_io_toPostMul_signProd),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isZeroProd         (mulAddRecFN_preMul_io_toPostMul_isZeroProd),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_opSignC            (mulAddRecFN_preMul_io_toPostMul_opSignC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highExpC           (mulAddRecFN_preMul_io_toPostMul_highExpC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNC  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isCDominant        (mulAddRecFN_preMul_io_toPostMul_isCDominant),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_CAlignDist_0       (mulAddRecFN_preMul_io_toPostMul_CAlignDist_0),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_CAlignDist         (mulAddRecFN_preMul_io_toPostMul_CAlignDist),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_bit0AlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highAlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_sExpSum            (mulAddRecFN_preMul_io_toPostMul_sExpSum),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_roundingMode       (mulAddRecFN_preMul_io_toPostMul_roundingMode),	// MulAddRecFN.scala:598:15
    .io_mulAddResult                  ({1'h0, {24'h0, mulAddRecFN_preMul_io_mulAddA} * {24'h0, mulAddRecFN_preMul_io_mulAddB}} +
                {1'h0, mulAddRecFN_preMul_io_mulAddC}),	// Cat.scala:30:58, MulAddRecFN.scala:598:15, :610:{39,71}
    .io_out                           (io_out),
    .io_exceptionFlags                (io_exceptionFlags)
  );
endmodule

module CompareRecFN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2028:10
  input  [64:0] io_a, io_b,
  input         io_signaling,
  output        io_lt, io_eq, io_gt,
  output [4:0]  io_exceptionFlags);

  wire [11:0] _io_a_63to52 = io_a[63:52];	// rawFNFromRecFN.scala:50:21
  wire [2:0] _io_a_63to61 = io_a[63:61];	// rawFNFromRecFN.scala:51:29
  wire [1:0] _io_a_63to62 = io_a[63:62];	// rawFNFromRecFN.scala:52:29
  wire _io_a_64 = io_a[64];	// rawFNFromRecFN.scala:55:23
  wire _io_a_61 = io_a[61];	// rawFNFromRecFN.scala:56:40
  wire _T = &_io_a_63to62 & _io_a_61;	// rawFNFromRecFN.scala:52:54, :56:32
  wire [51:0] _io_a_51to0 = io_a[51:0];	// rawFNFromRecFN.scala:60:48
  wire [11:0] _io_b_63to52 = io_b[63:52];	// rawFNFromRecFN.scala:50:21
  wire [2:0] _io_b_63to61 = io_b[63:61];	// rawFNFromRecFN.scala:51:29
  wire [1:0] _io_b_63to62 = io_b[63:62];	// rawFNFromRecFN.scala:52:29
  wire _io_b_64 = io_b[64];	// rawFNFromRecFN.scala:55:23
  wire _io_b_61 = io_b[61];	// rawFNFromRecFN.scala:56:40
  wire _T_0 = &_io_b_63to62 & _io_b_61;	// rawFNFromRecFN.scala:52:54, :56:32
  wire [51:0] _io_b_51to0 = io_b[51:0];	// rawFNFromRecFN.scala:60:48
  wire _ordered = ~_T & ~_T_0;	// CompareRecFN.scala:57:{19,32,35}
  wire _T_1 = &_io_a_63to62 & ~_io_a_61 & &_io_b_63to62 & ~_io_b_61;	// CompareRecFN.scala:58:33, rawFNFromRecFN.scala:52:54, :57:35
  wire _bothZeros = ~(|_io_a_63to61) & ~(|_io_b_63to61);	// CompareRecFN.scala:59:33, rawFNFromRecFN.scala:51:54
  wire _T_2 = _io_a_63to52 == _io_b_63to52;	// CompareRecFN.scala:60:29
  wire _common_ltMags = $signed({1'h0, _io_a_63to52}) < $signed({1'h0, _io_b_63to52}) | _T_2 & {|_io_a_63to61,
                _io_a_51to0} < {|_io_b_63to61, _io_b_51to0};	// Cat.scala:30:58, CompareRecFN.scala:62:{20,33,44,57}, rawFNFromRecFN.scala:51:54, :59:25
  wire _common_eqMags = _T_2 & {|_io_a_63to61, _io_a_51to0} == {|_io_b_63to61, _io_b_51to0};	// Cat.scala:30:58, CompareRecFN.scala:63:{32,45}, rawFNFromRecFN.scala:51:54
  wire _ordered_lt = ~_bothZeros & (_io_a_64 & ~_io_b_64 | ~_T_1 & (_io_a_64 & ~_common_ltMags & ~_common_eqMags
                | ~_io_b_64 & _common_ltMags));	// CompareRecFN.scala:66:{9,21}, :67:{25,28,41}, :68:{19,30}, :69:{38,54,57,74}, :70:41
  wire _ordered_eq = _bothZeros | _io_a_64 == _io_b_64 & (_T_1 | _common_eqMags);	// CompareRecFN.scala:72:{19,34,49,62}
  assign io_lt = _ordered & _ordered_lt;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2028:10, CompareRecFN.scala:78:22
  assign io_eq = _ordered & _ordered_eq;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2028:10, CompareRecFN.scala:79:22
  assign io_gt = _ordered & ~_ordered_lt & ~_ordered_eq;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2028:10, CompareRecFN.scala:80:{25,38,41}
  assign io_exceptionFlags = {_T & ~(io_a[51]) | _T_0 & ~(io_b[51]) | io_signaling & ~_ordered, 4'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2028:10, Cat.scala:30:58, CompareRecFN.scala:75:52, :76:{27,30}, RoundRawFNToRecFN.scala:61:{46,49,57}
endmodule

module RecFNToIN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2137:10
  input  [64:0] io_in,
  input  [1:0]  io_roundingMode,
  input         io_signedOut,
  output [31:0] io_out,
  output [2:0]  io_intExceptionFlags);

  wire _sign = io_in[64];	// RecFNToIN.scala:54:21
  wire [1:0] _io_in_63to62 = io_in[63:62];	// RecFNToIN.scala:59:25
  wire _notSpecial_magGeOne = io_in[63];	// RecFNToIN.scala:61:34
  wire [83:0] _T = {31'h0, _notSpecial_magGeOne, io_in[51:0]} << (_notSpecial_magGeOne ? io_in[56:52] : 5'h0);	// RecFNToIN.scala:56:22, :72:40, :73:16, :74:20, :127:12
  wire [1:0] _T_0 = {_T[51], |(_T[50:0])};	// RecFNToIN.scala:86:{23,41}, :88:58
  wire _roundInexact = _notSpecial_magGeOne ? |_T_0 : |(io_in[63:61]);	// RecFNToIN.scala:58:{22,47}, :88:{27,65}
  wire [10:0] _posExp = io_in[62:52];	// RecFNToIN.scala:92:20
  wire _T_1 = io_roundingMode == 2'h0 & (_notSpecial_magGeOne ? &(_T[52:51]) | &_T_0 : &_posExp & |_T_0)
                | io_roundingMode == 2'h2 & _sign & _roundInexact | &io_roundingMode & ~_sign &
                _roundInexact;	// RecFNToIN.scala:85:23, :88:65, :90:12, :91:{29,34,53}, :92:{16,38}, :95:{27,51}, :96:{27,49,78}, :97:{27,49,53}
  wire [31:0] _T_2 = {32{_sign}} ^ _T[83:52];	// RecFNToIN.scala:82:24, :98:32
  wire _roundCarryBut2 = &(_T[81:52]) & _T_1;	// RecFNToIN.scala:103:{38,56,61}
  wire [5:0] _io_in_62to57 = io_in[62:57];	// RecFNToIN.scala:108:21
  wire _T_3 = _posExp == 11'h1F;	// RecFNToIN.scala:109:26
  wire _overflow = io_signedOut ? _notSpecial_magGeOne & (|_io_in_62to57 | _T_3 & (~_sign | |(_T[82:52]) |
                _T_1) | ~_sign & _posExp == 11'h1E & _roundCarryBut2) : _notSpecial_magGeOne ? _sign |
                |_io_in_62to57 | _T_3 & _T[82] & _roundCarryBut2 : _sign & _T_1;	// RecFNToIN.scala:97:53, :107:12, :108:21, :109:50, :110:{45,63}, :111:{27,42}, :112:{36,60}, :116:12, :117:48, :119:{34,49}, :120:18, :122:23
  wire _excSign = _sign & ~(&_io_in_63to62 & io_in[61]);	// RecFNToIN.scala:59:50, :60:{27,33}, :124:{24,27}
  assign io_out = &_io_in_63to62 | _overflow ? {io_signedOut & _excSign, {31{io_signedOut & ~_excSign}}} |
                {32{~io_signedOut & ~_excSign}} : _T_1 ^ _sign ? _T_2 + 32'h1 : _T_2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2137:10, RecFNToIN.scala:59:50, :100:{12,23,49}, :126:{26,72}, :127:{12,26,29}, :130:11, :131:{12,13,28}, :137:{18,27}
  assign io_intExceptionFlags = {&_io_in_63to62, _overflow, _roundInexact & ~(&_io_in_63to62) & ~_overflow};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2137:10, Cat.scala:30:58, RecFNToIN.scala:59:50, :135:{35,45,48}
endmodule

module RecFNToIN_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2253:10
  input  [64:0] io_in,
  input  [1:0]  io_roundingMode,
  input         io_signedOut,
  output [63:0] io_out,
  output [2:0]  io_intExceptionFlags);

  wire _sign = io_in[64];	// RecFNToIN.scala:54:21
  wire [1:0] _io_in_63to62 = io_in[63:62];	// RecFNToIN.scala:59:25
  wire _notSpecial_magGeOne = io_in[63];	// RecFNToIN.scala:61:34
  wire [115:0] _T = {63'h0, _notSpecial_magGeOne, io_in[51:0]} << (_notSpecial_magGeOne ? io_in[57:52] : 6'h0);	// RecFNToIN.scala:56:22, :72:40, :73:16, :74:20, :127:12
  wire [1:0] _T_0 = {_T[51], |(_T[50:0])};	// RecFNToIN.scala:86:{23,41}, :88:58
  wire _roundInexact = _notSpecial_magGeOne ? |_T_0 : |(io_in[63:61]);	// RecFNToIN.scala:58:{22,47}, :88:{27,65}
  wire [10:0] _posExp = io_in[62:52];	// RecFNToIN.scala:92:20
  wire _T_1 = io_roundingMode == 2'h0 & (_notSpecial_magGeOne ? &(_T[52:51]) | &_T_0 : &_posExp & |_T_0)
                | io_roundingMode == 2'h2 & _sign & _roundInexact | &io_roundingMode & ~_sign &
                _roundInexact;	// RecFNToIN.scala:85:23, :88:65, :90:12, :91:{29,34,53}, :92:{16,38}, :95:{27,51}, :96:{27,49,78}, :97:{27,49,53}
  wire [63:0] _T_2 = {64{_sign}} ^ _T[115:52];	// RecFNToIN.scala:82:24, :98:32
  wire _roundCarryBut2 = &(_T[113:52]) & _T_1;	// RecFNToIN.scala:103:{38,56,61}
  wire [4:0] _io_in_62to58 = io_in[62:58];	// RecFNToIN.scala:108:21
  wire _T_3 = _posExp == 11'h3F;	// RecFNToIN.scala:109:26
  wire _overflow = io_signedOut ? _notSpecial_magGeOne & (|_io_in_62to58 | _T_3 & (~_sign | |(_T[114:52]) |
                _T_1) | ~_sign & _posExp == 11'h3E & _roundCarryBut2) : _notSpecial_magGeOne ? _sign |
                |_io_in_62to58 | _T_3 & _T[114] & _roundCarryBut2 : _sign & _T_1;	// RecFNToIN.scala:97:53, :107:12, :108:21, :109:50, :110:{45,63}, :111:{27,42}, :112:{36,60}, :116:12, :117:48, :119:{34,49}, :120:18, :122:23
  wire _excSign = _sign & ~(&_io_in_63to62 & io_in[61]);	// RecFNToIN.scala:59:50, :60:{27,33}, :124:{24,27}
  assign io_out = &_io_in_63to62 | _overflow ? {io_signedOut & _excSign, {63{io_signedOut & ~_excSign}}} |
                {64{~io_signedOut & ~_excSign}} : _T_1 ^ _sign ? _T_2 + 64'h1 : _T_2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2253:10, RecFNToIN.scala:59:50, :100:{12,23,49}, :126:{26,72}, :127:{12,26,29}, :130:11, :131:{12,13,28}, :137:{18,27}
  assign io_intExceptionFlags = {&_io_in_63to62, _overflow, _roundInexact & ~(&_io_in_63to62) & ~_overflow};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2253:10, Cat.scala:30:58, RecFNToIN.scala:59:50, :135:{35,45,48}
endmodule

module INToRecFN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2369:10
  input         io_signedIn,
  input  [63:0] io_in,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire _sign = io_signedIn & io_in[63];	// INToRecFN.scala:55:{28,36}
  wire [63:0] _absIn = _sign ? 64'h0 - io_in : io_in;	// INToRecFN.scala:56:{20,27}
  wire [31:0] _absIn_63to32 = _absIn[63:32];	// CircuitMath.scala:35:17
  wire [15:0] _absIn_63to48 = _absIn[63:48];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_63to56 = _absIn[63:56];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_63to60 = _absIn[63:60];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_55to52 = _absIn[55:52];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_47to40 = _absIn[47:40];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_47to44 = _absIn[47:44];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_39to36 = _absIn[39:36];	// CircuitMath.scala:35:17
  wire [15:0] _absIn_31to16 = _absIn[31:16];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_31to24 = _absIn[31:24];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_31to28 = _absIn[31:28];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_23to20 = _absIn[23:20];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_15to8 = _absIn[15:8];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_15to12 = _absIn[15:12];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_7to4 = _absIn[7:4];	// CircuitMath.scala:35:17
  wire [4:0] _T = |_absIn_63to32 ? {|_absIn_63to48, |_absIn_63to48 ? {|_absIn_63to56, |_absIn_63to56 ?
                {|_absIn_63to60, |_absIn_63to60 ? (_absIn[63] ? 2'h3 : _absIn[62] ? 2'h2 : {1'h0,
                _absIn[61]}) : _absIn[59] ? 2'h3 : _absIn[58] ? 2'h2 : {1'h0, _absIn[57]}} :
                {|_absIn_55to52, |_absIn_55to52 ? (_absIn[55] ? 2'h3 : _absIn[54] ? 2'h2 : {1'h0,
                _absIn[53]}) : _absIn[51] ? 2'h3 : _absIn[50] ? 2'h2 : {1'h0, _absIn[49]}}} :
                {|_absIn_47to40, |_absIn_47to40 ? {|_absIn_47to44, |_absIn_47to44 ? (_absIn[47] ? 2'h3 :
                _absIn[46] ? 2'h2 : {1'h0, _absIn[45]}) : _absIn[43] ? 2'h3 : _absIn[42] ? 2'h2 : {1'h0,
                _absIn[41]}} : {|_absIn_39to36, |_absIn_39to36 ? (_absIn[39] ? 2'h3 : _absIn[38] ? 2'h2 :
                {1'h0, _absIn[37]}) : _absIn[35] ? 2'h3 : _absIn[34] ? 2'h2 : {1'h0, _absIn[33]}}}} :
                {|_absIn_31to16, |_absIn_31to16 ? {|_absIn_31to24, |_absIn_31to24 ? {|_absIn_31to28,
                |_absIn_31to28 ? (_absIn[31] ? 2'h3 : _absIn[30] ? 2'h2 : {1'h0, _absIn[29]}) : _absIn[27]
                ? 2'h3 : _absIn[26] ? 2'h2 : {1'h0, _absIn[25]}} : {|_absIn_23to20, |_absIn_23to20 ?
                (_absIn[23] ? 2'h3 : _absIn[22] ? 2'h2 : {1'h0, _absIn[21]}) : _absIn[19] ? 2'h3 :
                _absIn[18] ? 2'h2 : {1'h0, _absIn[17]}}} : {|_absIn_15to8, |_absIn_15to8 ? {|_absIn_15to12,
                |_absIn_15to12 ? (_absIn[15] ? 2'h3 : _absIn[14] ? 2'h2 : {1'h0, _absIn[13]}) : _absIn[11]
                ? 2'h3 : _absIn[10] ? 2'h2 : {1'h0, _absIn[9]}} : {|_absIn_7to4, |_absIn_7to4 ? (_absIn[7]
                ? 2'h3 : _absIn[6] ? 2'h2 : {1'h0, _absIn[5]}) : _absIn[3] ? 2'h3 : _absIn[2] ? 2'h2 :
                {1'h0, _absIn[1]}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, INToRecFN.scala:56:27
  wire [126:0] _T_0 = {63'h0, _absIn} << ~{|_absIn_63to32, _T};	// Cat.scala:30:58, CircuitMath.scala:37:22, INToRecFN.scala:57:21, :58:27
  wire [1:0] _T_1 = {_T_0[39], |(_T_0[38:0])};	// INToRecFN.scala:64:{26,55}, :72:33
  wire [23:0] _T_2 = _T_0[63:40];	// INToRecFN.scala:89:34
  wire [24:0] _roundedNorm = io_roundingMode == 2'h0 & (&(_T_0[40:39]) | &_T_1) | io_roundingMode == 2'h2 & _sign &
                |_T_1 | &io_roundingMode & ~_sign & |_T_1 ? {1'h0, _T_2} + 25'h1 : {1'h0, _T_2};	// Cat.scala:30:58, CircuitMath.scala:32:10, INToRecFN.scala:56:27, :63:26, :72:40, :74:{12,30}, :75:{29,34,53}, :78:{12,30}, :81:11, :82:{12,30}, :83:13, :94:{26,48}
  assign io_out = {_sign, _T_0[63], {2'h0, |_absIn_63to32, _T} + {7'h0, _roundedNorm[24]},
                _roundedNorm[22:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2369:10, Cat.scala:30:58, CircuitMath.scala:37:22, INToRecFN.scala:72:40, :106:{52,65}, :112:22, :122:44
  assign io_exceptionFlags = {4'h0, |_T_1};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2369:10, Cat.scala:30:58, CircuitMath.scala:37:22, INToRecFN.scala:72:40
endmodule

module INToRecFN_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2591:10
  input         io_signedIn,
  input  [63:0] io_in,
  input  [1:0]  io_roundingMode,
  output [64:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire _sign = io_signedIn & io_in[63];	// INToRecFN.scala:55:{28,36}
  wire [63:0] _absIn = _sign ? 64'h0 - io_in : io_in;	// INToRecFN.scala:56:{20,27}
  wire [31:0] _absIn_63to32 = _absIn[63:32];	// CircuitMath.scala:35:17
  wire [15:0] _absIn_63to48 = _absIn[63:48];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_63to56 = _absIn[63:56];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_63to60 = _absIn[63:60];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_55to52 = _absIn[55:52];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_47to40 = _absIn[47:40];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_47to44 = _absIn[47:44];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_39to36 = _absIn[39:36];	// CircuitMath.scala:35:17
  wire [15:0] _absIn_31to16 = _absIn[31:16];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_31to24 = _absIn[31:24];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_31to28 = _absIn[31:28];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_23to20 = _absIn[23:20];	// CircuitMath.scala:35:17
  wire [7:0] _absIn_15to8 = _absIn[15:8];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_15to12 = _absIn[15:12];	// CircuitMath.scala:35:17
  wire [3:0] _absIn_7to4 = _absIn[7:4];	// CircuitMath.scala:35:17
  wire [4:0] _T = |_absIn_63to32 ? {|_absIn_63to48, |_absIn_63to48 ? {|_absIn_63to56, |_absIn_63to56 ?
                {|_absIn_63to60, |_absIn_63to60 ? (_absIn[63] ? 2'h3 : _absIn[62] ? 2'h2 : {1'h0,
                _absIn[61]}) : _absIn[59] ? 2'h3 : _absIn[58] ? 2'h2 : {1'h0, _absIn[57]}} :
                {|_absIn_55to52, |_absIn_55to52 ? (_absIn[55] ? 2'h3 : _absIn[54] ? 2'h2 : {1'h0,
                _absIn[53]}) : _absIn[51] ? 2'h3 : _absIn[50] ? 2'h2 : {1'h0, _absIn[49]}}} :
                {|_absIn_47to40, |_absIn_47to40 ? {|_absIn_47to44, |_absIn_47to44 ? (_absIn[47] ? 2'h3 :
                _absIn[46] ? 2'h2 : {1'h0, _absIn[45]}) : _absIn[43] ? 2'h3 : _absIn[42] ? 2'h2 : {1'h0,
                _absIn[41]}} : {|_absIn_39to36, |_absIn_39to36 ? (_absIn[39] ? 2'h3 : _absIn[38] ? 2'h2 :
                {1'h0, _absIn[37]}) : _absIn[35] ? 2'h3 : _absIn[34] ? 2'h2 : {1'h0, _absIn[33]}}}} :
                {|_absIn_31to16, |_absIn_31to16 ? {|_absIn_31to24, |_absIn_31to24 ? {|_absIn_31to28,
                |_absIn_31to28 ? (_absIn[31] ? 2'h3 : _absIn[30] ? 2'h2 : {1'h0, _absIn[29]}) : _absIn[27]
                ? 2'h3 : _absIn[26] ? 2'h2 : {1'h0, _absIn[25]}} : {|_absIn_23to20, |_absIn_23to20 ?
                (_absIn[23] ? 2'h3 : _absIn[22] ? 2'h2 : {1'h0, _absIn[21]}) : _absIn[19] ? 2'h3 :
                _absIn[18] ? 2'h2 : {1'h0, _absIn[17]}}} : {|_absIn_15to8, |_absIn_15to8 ? {|_absIn_15to12,
                |_absIn_15to12 ? (_absIn[15] ? 2'h3 : _absIn[14] ? 2'h2 : {1'h0, _absIn[13]}) : _absIn[11]
                ? 2'h3 : _absIn[10] ? 2'h2 : {1'h0, _absIn[9]}} : {|_absIn_7to4, |_absIn_7to4 ? (_absIn[7]
                ? 2'h3 : _absIn[6] ? 2'h2 : {1'h0, _absIn[5]}) : _absIn[3] ? 2'h3 : _absIn[2] ? 2'h2 :
                {1'h0, _absIn[1]}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, INToRecFN.scala:56:27
  wire [126:0] _T_0 = {63'h0, _absIn} << ~{|_absIn_63to32, _T};	// Cat.scala:30:58, CircuitMath.scala:37:22, INToRecFN.scala:57:21, :58:27
  wire [1:0] _T_1 = {_T_0[10], |(_T_0[9:0])};	// INToRecFN.scala:64:{26,55}, :72:33
  wire [52:0] _T_2 = _T_0[63:11];	// INToRecFN.scala:89:34
  wire [53:0] _roundedNorm = io_roundingMode == 2'h0 & (&(_T_0[11:10]) | &_T_1) | io_roundingMode == 2'h2 & _sign &
                |_T_1 | &io_roundingMode & ~_sign & |_T_1 ? {1'h0, _T_2} + 54'h1 : {1'h0, _T_2};	// Cat.scala:30:58, CircuitMath.scala:32:10, INToRecFN.scala:56:27, :63:26, :72:40, :74:{12,30}, :75:{29,34,53}, :78:{12,30}, :81:11, :82:{12,30}, :83:13, :94:{26,48}
  assign io_out = {_sign, _T_0[63], {5'h0, |_absIn_63to32, _T} + {10'h0, _roundedNorm[53]},
                _roundedNorm[51:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2591:10, Cat.scala:30:58, CircuitMath.scala:37:22, INToRecFN.scala:64:55, :106:{52,65}, :112:22, :122:44
  assign io_exceptionFlags = {4'h0, |_T_1};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2591:10, Cat.scala:30:58, INToRecFN.scala:72:40
endmodule

module RecFNToRecFN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2813:10
  input  [64:0] io_in,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [2:0] _io_in_63to61 = io_in[63:61];	// rawFNFromRecFN.scala:51:29
  wire [1:0] _io_in_63to62 = io_in[63:62];	// rawFNFromRecFN.scala:52:29
  wire _io_in_61 = io_in[61];	// rawFNFromRecFN.scala:56:40
  wire _T = &_io_in_63to62 & _io_in_61;	// rawFNFromRecFN.scala:52:54, :56:32
  wire [13:0] _T_0 = {2'h0, io_in[63:52]} - 14'h700;	// Cat.scala:30:58, rawFNFromRecFN.scala:50:21, resizeRawFN.scala:49:31
  RoundRawFNToRecFN RoundRawFNToRecFN (	// RecFNToRecFN.scala:102:19
    .io_invalidExc     (_T & ~(io_in[51])),	// RoundRawFNToRecFN.scala:61:{46,49,57}
    .io_in_sign        (io_in[64]),	// rawFNFromRecFN.scala:55:23
    .io_in_isNaN       (_T),
    .io_in_isInf       (&_io_in_63to62 & ~_io_in_61),	// rawFNFromRecFN.scala:52:54, :57:{32,35}
    .io_in_isZero      (~(|_io_in_63to61)),	// rawFNFromRecFN.scala:51:54
    .io_in_sExp        ({$signed(_T_0) < 14'sh0, |(_T_0[12:9]) ? 9'h1FC : _T_0[8:0]}),	// Cat.scala:30:58, resizeRawFN.scala:60:31, :61:{25,33,65}, :63:33
    .io_in_sig         ({1'h0, |_io_in_63to61, io_in[51:28], |(io_in[27:0])}),	// Cat.scala:30:58, rawFNFromRecFN.scala:51:54, resizeRawFN.scala:71:28, :72:{28,56}
    .io_roundingMode   (io_roundingMode),
    .io_out            (io_out),
    .io_exceptionFlags (io_exceptionFlags)
  );
endmodule

module RecFNToRecFN_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2881:10
  input  [32:0] io_in,
  output [64:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [2:0] _io_in_31to29 = io_in[31:29];	// rawFNFromRecFN.scala:51:29
  wire [1:0] _io_in_31to30 = io_in[31:30];	// rawFNFromRecFN.scala:52:29
  wire _io_in_29 = io_in[29];	// rawFNFromRecFN.scala:56:40
  wire _T = &_io_in_31to30 & _io_in_29;	// rawFNFromRecFN.scala:52:54, :56:32
  wire _T_0 = &_io_in_31to30 & ~_io_in_29;	// rawFNFromRecFN.scala:52:54, :57:{32,35}
  assign io_out = {io_in[32] & ~_T, {3'h0, io_in[31:23]} + 12'h700 & ~(|_io_in_31to29 ? 12'h0 : 12'hC00) &
                {2'h3, ~(~(|_io_in_31to29) | _T_0), 9'h1FF} | (_T_0 ? 12'hC00 : 12'h0) | (_T ? 12'hE00 :
                12'h0), _T ? 52'h8000000000000 : {io_in[22:0], 29'h0}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2881:10, Cat.scala:30:58, RecFNToRecFN.scala:69:{37,40}, :72:{18,22}, :75:21, :76:{18,22,42}, :80:20, :83:19, :84:20, :89:16, :90:24, :91:32, rawFNFromRecFN.scala:50:21, :51:54, :52:54, :55:23, :60:48, resizeRawFN.scala:49:31
  assign io_exceptionFlags = {_T & ~(io_in[22]), 4'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2881:10, Cat.scala:30:58, RoundRawFNToRecFN.scala:61:{46,49,57}
endmodule

module MulAddRecFN_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2949:10
  input  [1:0]  io_op,
  input  [64:0] io_a, io_b, io_c,
  input  [1:0]  io_roundingMode,
  output [64:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [52:0]  mulAddRecFN_preMul_io_mulAddA;	// MulAddRecFN.scala:598:15
  wire [52:0]  mulAddRecFN_preMul_io_mulAddB;	// MulAddRecFN.scala:598:15
  wire [105:0] mulAddRecFN_preMul_io_mulAddC;	// MulAddRecFN.scala:598:15
  wire [2:0]   mulAddRecFN_preMul_io_toPostMul_highExpA;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA;	// MulAddRecFN.scala:598:15
  wire [2:0]   mulAddRecFN_preMul_io_toPostMul_highExpB;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_signProd;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_isZeroProd;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_opSignC;	// MulAddRecFN.scala:598:15
  wire [2:0]   mulAddRecFN_preMul_io_toPostMul_highExpC;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_isCDominant;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_CAlignDist_0;	// MulAddRecFN.scala:598:15
  wire [7:0]   mulAddRecFN_preMul_io_toPostMul_CAlignDist;	// MulAddRecFN.scala:598:15
  wire         mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC;	// MulAddRecFN.scala:598:15
  wire [54:0]  mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC;	// MulAddRecFN.scala:598:15
  wire [13:0]  mulAddRecFN_preMul_io_toPostMul_sExpSum;	// MulAddRecFN.scala:598:15
  wire [1:0]   mulAddRecFN_preMul_io_toPostMul_roundingMode;	// MulAddRecFN.scala:598:15

  MulAddRecFN_preMul_1 mulAddRecFN_preMul (	// MulAddRecFN.scala:598:15
    .io_op                           (io_op),
    .io_a                            (io_a),
    .io_b                            (io_b),
    .io_c                            (io_c),
    .io_roundingMode                 (io_roundingMode),
    .io_mulAddA                      (mulAddRecFN_preMul_io_mulAddA),
    .io_mulAddB                      (mulAddRecFN_preMul_io_mulAddB),
    .io_mulAddC                      (mulAddRecFN_preMul_io_mulAddC),
    .io_toPostMul_highExpA           (mulAddRecFN_preMul_io_toPostMul_highExpA),
    .io_toPostMul_isNaN_isQuietNaNA  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA),
    .io_toPostMul_highExpB           (mulAddRecFN_preMul_io_toPostMul_highExpB),
    .io_toPostMul_isNaN_isQuietNaNB  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB),
    .io_toPostMul_signProd           (mulAddRecFN_preMul_io_toPostMul_signProd),
    .io_toPostMul_isZeroProd         (mulAddRecFN_preMul_io_toPostMul_isZeroProd),
    .io_toPostMul_opSignC            (mulAddRecFN_preMul_io_toPostMul_opSignC),
    .io_toPostMul_highExpC           (mulAddRecFN_preMul_io_toPostMul_highExpC),
    .io_toPostMul_isNaN_isQuietNaNC  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC),
    .io_toPostMul_isCDominant        (mulAddRecFN_preMul_io_toPostMul_isCDominant),
    .io_toPostMul_CAlignDist_0       (mulAddRecFN_preMul_io_toPostMul_CAlignDist_0),
    .io_toPostMul_CAlignDist         (mulAddRecFN_preMul_io_toPostMul_CAlignDist),
    .io_toPostMul_bit0AlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC),
    .io_toPostMul_highAlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC),
    .io_toPostMul_sExpSum            (mulAddRecFN_preMul_io_toPostMul_sExpSum),
    .io_toPostMul_roundingMode       (mulAddRecFN_preMul_io_toPostMul_roundingMode)
  );
  MulAddRecFN_postMul_1 mulAddRecFN_postMul (	// MulAddRecFN.scala:600:15
    .io_fromPreMul_highExpA           (mulAddRecFN_preMul_io_toPostMul_highExpA),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNA  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNA),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highExpB           (mulAddRecFN_preMul_io_toPostMul_highExpB),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNB  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNB),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_signProd           (mulAddRecFN_preMul_io_toPostMul_signProd),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isZeroProd         (mulAddRecFN_preMul_io_toPostMul_isZeroProd),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_opSignC            (mulAddRecFN_preMul_io_toPostMul_opSignC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highExpC           (mulAddRecFN_preMul_io_toPostMul_highExpC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isNaN_isQuietNaNC  (mulAddRecFN_preMul_io_toPostMul_isNaN_isQuietNaNC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_isCDominant        (mulAddRecFN_preMul_io_toPostMul_isCDominant),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_CAlignDist_0       (mulAddRecFN_preMul_io_toPostMul_CAlignDist_0),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_CAlignDist         (mulAddRecFN_preMul_io_toPostMul_CAlignDist),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_bit0AlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_bit0AlignedNegSigC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_highAlignedNegSigC (mulAddRecFN_preMul_io_toPostMul_highAlignedNegSigC),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_sExpSum            (mulAddRecFN_preMul_io_toPostMul_sExpSum),	// MulAddRecFN.scala:598:15
    .io_fromPreMul_roundingMode       (mulAddRecFN_preMul_io_toPostMul_roundingMode),	// MulAddRecFN.scala:598:15
    .io_mulAddResult                  ({1'h0, {53'h0, mulAddRecFN_preMul_io_mulAddA} * {53'h0, mulAddRecFN_preMul_io_mulAddB}} +
                {1'h0, mulAddRecFN_preMul_io_mulAddC}),	// Cat.scala:30:58, MulAddRecFN.scala:598:15, :610:{39,71}
    .io_out                           (io_out),
    .io_exceptionFlags                (io_exceptionFlags)
  );
endmodule

module DivSqrtRecF64_mulAddZ31(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10
  input          clock, reset, io_inValid, io_sqrtOp,
  input  [64:0]  io_a, io_b,
  input  [1:0]   io_roundingMode,
  input  [104:0] io_mulAddResult_3,
  output         io_inReady_div, io_inReady_sqrt, io_outValid_div, io_outValid_sqrt,
  output [64:0]  io_out,
  output [4:0]   io_exceptionFlags,
  output [3:0]   io_usingMulAdd,
  output         io_latchMulAddA_0,
  output [53:0]  io_mulAddA_0,
  output         io_latchMulAddB_0,
  output [53:0]  io_mulAddB_0,
  output [104:0] io_mulAddC_2);

  wire [53:0] _T;	// DivSqrtRecF64_mulAddZ31.scala:716:12
  wire [53:0] _T_0;	// DivSqrtRecF64_mulAddZ31.scala:710:11
  wire [45:0] _T_1;	// DivSqrtRecF64_mulAddZ31.scala:702:22
  wire        _T_2;	// DivSqrtRecF64_mulAddZ31.scala:481:27
  wire        _T_3;	// DivSqrtRecF64_mulAddZ31.scala:458:27
  wire        _T_4;	// DivSqrtRecF64_mulAddZ31.scala:457:27
  wire        _T_5;	// DivSqrtRecF64_mulAddZ31.scala:456:27
  wire        _T_6;	// DivSqrtRecF64_mulAddZ31.scala:447:27
  wire        _T_7;	// DivSqrtRecF64_mulAddZ31.scala:444:39
  wire        _T_8;	// DivSqrtRecF64_mulAddZ31.scala:443:39
  wire        _T_9;	// DivSqrtRecF64_mulAddZ31.scala:442:39
  wire        _T_10;	// DivSqrtRecF64_mulAddZ31.scala:439:26
  wire        _T_11;	// DivSqrtRecF64_mulAddZ31.scala:437:38
  wire        _T_12;	// DivSqrtRecF64_mulAddZ31.scala:432:27
  wire        _T_13;	// DivSqrtRecF64_mulAddZ31.scala:431:27
  wire        _T_14;	// DivSqrtRecF64_mulAddZ31.scala:426:33
  wire        _T_15;	// DivSqrtRecF64_mulAddZ31.scala:382:28
  wire        _T_16;	// DivSqrtRecF64_mulAddZ31.scala:322:28
  wire        _T_17;	// DivSqrtRecF64_mulAddZ31.scala:284:28
  reg         valid_PA;	// DivSqrtRecF64_mulAddZ31.scala:78:30
  reg         sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:79:30
  reg         sign_PA;	// DivSqrtRecF64_mulAddZ31.scala:80:30
  reg  [2:0]  specialCodeB_PA;	// DivSqrtRecF64_mulAddZ31.scala:82:30
  reg         fractB_51_PA;	// DivSqrtRecF64_mulAddZ31.scala:83:30
  reg  [1:0]  roundingMode_PA;	// DivSqrtRecF64_mulAddZ31.scala:84:30
  reg  [2:0]  specialCodeA_PA;	// DivSqrtRecF64_mulAddZ31.scala:85:30
  reg         fractA_51_PA;	// DivSqrtRecF64_mulAddZ31.scala:86:30
  reg  [13:0] exp_PA;	// DivSqrtRecF64_mulAddZ31.scala:87:30
  reg  [50:0] fractB_other_PA;	// DivSqrtRecF64_mulAddZ31.scala:88:30
  reg  [50:0] fractA_other_PA;	// DivSqrtRecF64_mulAddZ31.scala:89:30
  reg         valid_PB;	// DivSqrtRecF64_mulAddZ31.scala:91:30
  reg         sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:92:30
  reg         sign_PB;	// DivSqrtRecF64_mulAddZ31.scala:93:30
  reg  [2:0]  specialCodeA_PB;	// DivSqrtRecF64_mulAddZ31.scala:95:30
  reg         fractA_51_PB;	// DivSqrtRecF64_mulAddZ31.scala:96:30
  reg  [2:0]  specialCodeB_PB;	// DivSqrtRecF64_mulAddZ31.scala:97:30
  reg         fractB_51_PB;	// DivSqrtRecF64_mulAddZ31.scala:98:30
  reg  [1:0]  roundingMode_PB;	// DivSqrtRecF64_mulAddZ31.scala:99:30
  reg  [13:0] exp_PB;	// DivSqrtRecF64_mulAddZ31.scala:100:30
  reg         fractA_0_PB;	// DivSqrtRecF64_mulAddZ31.scala:101:30
  reg  [50:0] fractB_other_PB;	// DivSqrtRecF64_mulAddZ31.scala:102:30
  reg         valid_PC;	// DivSqrtRecF64_mulAddZ31.scala:104:30
  reg         sqrtOp_PC;	// DivSqrtRecF64_mulAddZ31.scala:105:30
  reg         sign_PC;	// DivSqrtRecF64_mulAddZ31.scala:106:30
  reg  [2:0]  specialCodeA_PC;	// DivSqrtRecF64_mulAddZ31.scala:108:30
  reg         fractA_51_PC;	// DivSqrtRecF64_mulAddZ31.scala:109:30
  reg  [2:0]  specialCodeB_PC;	// DivSqrtRecF64_mulAddZ31.scala:110:30
  reg         fractB_51_PC;	// DivSqrtRecF64_mulAddZ31.scala:111:30
  reg  [1:0]  roundingMode_PC;	// DivSqrtRecF64_mulAddZ31.scala:112:30
  reg  [13:0] exp_PC;	// DivSqrtRecF64_mulAddZ31.scala:113:30
  reg         fractA_0_PC;	// DivSqrtRecF64_mulAddZ31.scala:114:30
  reg  [50:0] fractB_other_PC;	// DivSqrtRecF64_mulAddZ31.scala:115:30
  reg  [2:0]  cycleNum_A;	// DivSqrtRecF64_mulAddZ31.scala:117:30
  reg  [3:0]  cycleNum_B;	// DivSqrtRecF64_mulAddZ31.scala:118:30
  reg  [2:0]  cycleNum_C;	// DivSqrtRecF64_mulAddZ31.scala:119:30
  reg  [2:0]  cycleNum_E;	// DivSqrtRecF64_mulAddZ31.scala:120:30
  reg  [8:0]  fractR0_A;	// DivSqrtRecF64_mulAddZ31.scala:122:30
  reg  [9:0]  hiSqrR0_A_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:124:30
  reg  [20:0] partNegSigma0_A;	// DivSqrtRecF64_mulAddZ31.scala:125:30
  reg  [8:0]  nextMulAdd9A_A;	// DivSqrtRecF64_mulAddZ31.scala:126:30
  reg  [8:0]  nextMulAdd9B_A;	// DivSqrtRecF64_mulAddZ31.scala:127:30
  reg  [16:0] ER1_B_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:128:30
  reg  [31:0] ESqrR1_B_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:130:30
  reg  [57:0] sigX1_B;	// DivSqrtRecF64_mulAddZ31.scala:131:30
  reg  [32:0] sqrSigma1_C;	// DivSqrtRecF64_mulAddZ31.scala:132:30
  reg  [57:0] sigXN_C;	// DivSqrtRecF64_mulAddZ31.scala:133:30
  reg  [30:0] u_C_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:134:30
  reg         E_E_div;	// DivSqrtRecF64_mulAddZ31.scala:135:30
  reg  [52:0] sigT_E;	// DivSqrtRecF64_mulAddZ31.scala:136:30
  reg         extraT_E;	// DivSqrtRecF64_mulAddZ31.scala:137:30
  reg         isNegRemT_E;	// DivSqrtRecF64_mulAddZ31.scala:138:30
  reg         isZeroRemT_E;	// DivSqrtRecF64_mulAddZ31.scala:139:30

  `ifndef SYNTHESIS	// DivSqrtRecF64_mulAddZ31.scala:78:30
    `ifdef RANDOMIZE_REG_INIT	// DivSqrtRecF64_mulAddZ31.scala:78:30
      reg [31:0] _RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:78:30
      reg [31:0] _RANDOM_18;	// DivSqrtRecF64_mulAddZ31.scala:88:30
      reg [31:0] _RANDOM_19;	// DivSqrtRecF64_mulAddZ31.scala:88:30
      reg [31:0] _RANDOM_20;	// DivSqrtRecF64_mulAddZ31.scala:89:30
      reg [31:0] _RANDOM_21;	// DivSqrtRecF64_mulAddZ31.scala:89:30
      reg [31:0] _RANDOM_22;	// DivSqrtRecF64_mulAddZ31.scala:102:30
      reg [31:0] _RANDOM_23;	// DivSqrtRecF64_mulAddZ31.scala:102:30
      reg [31:0] _RANDOM_24;	// DivSqrtRecF64_mulAddZ31.scala:113:30
      reg [31:0] _RANDOM_25;	// DivSqrtRecF64_mulAddZ31.scala:115:30
      reg [31:0] _RANDOM_26;	// DivSqrtRecF64_mulAddZ31.scala:117:30
      reg [31:0] _RANDOM_27;	// DivSqrtRecF64_mulAddZ31.scala:125:30
      reg [31:0] _RANDOM_28;	// DivSqrtRecF64_mulAddZ31.scala:127:30
      reg [31:0] _RANDOM_29;	// DivSqrtRecF64_mulAddZ31.scala:130:30
      reg [31:0] _RANDOM_30;	// DivSqrtRecF64_mulAddZ31.scala:131:30
      reg [31:0] _RANDOM_31;	// DivSqrtRecF64_mulAddZ31.scala:131:30
      reg [31:0] _RANDOM_32;	// DivSqrtRecF64_mulAddZ31.scala:132:30
      reg [31:0] _RANDOM_33;	// DivSqrtRecF64_mulAddZ31.scala:133:30
      reg [31:0] _RANDOM_34;	// DivSqrtRecF64_mulAddZ31.scala:133:30
      reg [31:0] _RANDOM_35;	// DivSqrtRecF64_mulAddZ31.scala:134:30
      reg [31:0] _RANDOM_36;	// DivSqrtRecF64_mulAddZ31.scala:136:30
      reg [31:0] _RANDOM_37;	// DivSqrtRecF64_mulAddZ31.scala:136:30

    `endif
    initial begin	// DivSqrtRecF64_mulAddZ31.scala:78:30
      `INIT_RANDOM_PROLOG_	// DivSqrtRecF64_mulAddZ31.scala:78:30
      `ifdef RANDOMIZE_REG_INIT	// DivSqrtRecF64_mulAddZ31.scala:78:30
        _RANDOM = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:78:30
        valid_PA = _RANDOM[0];	// DivSqrtRecF64_mulAddZ31.scala:78:30
        sqrtOp_PA = _RANDOM[1];	// DivSqrtRecF64_mulAddZ31.scala:79:30
        sign_PA = _RANDOM[2];	// DivSqrtRecF64_mulAddZ31.scala:80:30
        specialCodeB_PA = _RANDOM[5:3];	// DivSqrtRecF64_mulAddZ31.scala:82:30
        fractB_51_PA = _RANDOM[6];	// DivSqrtRecF64_mulAddZ31.scala:83:30
        roundingMode_PA = _RANDOM[8:7];	// DivSqrtRecF64_mulAddZ31.scala:84:30
        specialCodeA_PA = _RANDOM[11:9];	// DivSqrtRecF64_mulAddZ31.scala:85:30
        fractA_51_PA = _RANDOM[12];	// DivSqrtRecF64_mulAddZ31.scala:86:30
        exp_PA = _RANDOM[26:13];	// DivSqrtRecF64_mulAddZ31.scala:87:30
        _RANDOM_18 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:88:30
        _RANDOM_19 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:88:30
        fractB_other_PA = {_RANDOM_19[13:0], _RANDOM_18, _RANDOM[31:27]};	// DivSqrtRecF64_mulAddZ31.scala:88:30
        _RANDOM_20 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:89:30
        _RANDOM_21 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:89:30
        fractA_other_PA = {_RANDOM_21[0], _RANDOM_20, _RANDOM_19[31:14]};	// DivSqrtRecF64_mulAddZ31.scala:89:30
        valid_PB = _RANDOM_21[1];	// DivSqrtRecF64_mulAddZ31.scala:91:30
        sqrtOp_PB = _RANDOM_21[2];	// DivSqrtRecF64_mulAddZ31.scala:92:30
        sign_PB = _RANDOM_21[3];	// DivSqrtRecF64_mulAddZ31.scala:93:30
        specialCodeA_PB = _RANDOM_21[6:4];	// DivSqrtRecF64_mulAddZ31.scala:95:30
        fractA_51_PB = _RANDOM_21[7];	// DivSqrtRecF64_mulAddZ31.scala:96:30
        specialCodeB_PB = _RANDOM_21[10:8];	// DivSqrtRecF64_mulAddZ31.scala:97:30
        fractB_51_PB = _RANDOM_21[11];	// DivSqrtRecF64_mulAddZ31.scala:98:30
        roundingMode_PB = _RANDOM_21[13:12];	// DivSqrtRecF64_mulAddZ31.scala:99:30
        exp_PB = _RANDOM_21[27:14];	// DivSqrtRecF64_mulAddZ31.scala:100:30
        fractA_0_PB = _RANDOM_21[28];	// DivSqrtRecF64_mulAddZ31.scala:101:30
        _RANDOM_22 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:102:30
        _RANDOM_23 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:102:30
        fractB_other_PB = {_RANDOM_23[15:0], _RANDOM_22, _RANDOM_21[31:29]};	// DivSqrtRecF64_mulAddZ31.scala:102:30
        valid_PC = _RANDOM_23[16];	// DivSqrtRecF64_mulAddZ31.scala:104:30
        sqrtOp_PC = _RANDOM_23[17];	// DivSqrtRecF64_mulAddZ31.scala:105:30
        sign_PC = _RANDOM_23[18];	// DivSqrtRecF64_mulAddZ31.scala:106:30
        specialCodeA_PC = _RANDOM_23[21:19];	// DivSqrtRecF64_mulAddZ31.scala:108:30
        fractA_51_PC = _RANDOM_23[22];	// DivSqrtRecF64_mulAddZ31.scala:109:30
        specialCodeB_PC = _RANDOM_23[25:23];	// DivSqrtRecF64_mulAddZ31.scala:110:30
        fractB_51_PC = _RANDOM_23[26];	// DivSqrtRecF64_mulAddZ31.scala:111:30
        roundingMode_PC = _RANDOM_23[28:27];	// DivSqrtRecF64_mulAddZ31.scala:112:30
        _RANDOM_24 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:113:30
        exp_PC = {_RANDOM_24[10:0], _RANDOM_23[31:29]};	// DivSqrtRecF64_mulAddZ31.scala:113:30
        fractA_0_PC = _RANDOM_24[11];	// DivSqrtRecF64_mulAddZ31.scala:114:30
        _RANDOM_25 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:115:30
        fractB_other_PC = {_RANDOM_25[30:0], _RANDOM_24[31:12]};	// DivSqrtRecF64_mulAddZ31.scala:115:30
        _RANDOM_26 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:117:30
        cycleNum_A = {_RANDOM_26[1:0], _RANDOM_25[31]};	// DivSqrtRecF64_mulAddZ31.scala:117:30
        cycleNum_B = _RANDOM_26[5:2];	// DivSqrtRecF64_mulAddZ31.scala:118:30
        cycleNum_C = _RANDOM_26[8:6];	// DivSqrtRecF64_mulAddZ31.scala:119:30
        cycleNum_E = _RANDOM_26[11:9];	// DivSqrtRecF64_mulAddZ31.scala:120:30
        fractR0_A = _RANDOM_26[20:12];	// DivSqrtRecF64_mulAddZ31.scala:122:30
        hiSqrR0_A_sqrt = _RANDOM_26[30:21];	// DivSqrtRecF64_mulAddZ31.scala:124:30
        _RANDOM_27 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:125:30
        partNegSigma0_A = {_RANDOM_27[19:0], _RANDOM_26[31]};	// DivSqrtRecF64_mulAddZ31.scala:125:30
        nextMulAdd9A_A = _RANDOM_27[28:20];	// DivSqrtRecF64_mulAddZ31.scala:126:30
        _RANDOM_28 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:127:30
        nextMulAdd9B_A = {_RANDOM_28[5:0], _RANDOM_27[31:29]};	// DivSqrtRecF64_mulAddZ31.scala:127:30
        ER1_B_sqrt = _RANDOM_28[22:6];	// DivSqrtRecF64_mulAddZ31.scala:128:30
        _RANDOM_29 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:130:30
        ESqrR1_B_sqrt = {_RANDOM_29[22:0], _RANDOM_28[31:23]};	// DivSqrtRecF64_mulAddZ31.scala:130:30
        _RANDOM_30 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:131:30
        _RANDOM_31 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:131:30
        sigX1_B = {_RANDOM_31[16:0], _RANDOM_30, _RANDOM_29[31:23]};	// DivSqrtRecF64_mulAddZ31.scala:131:30
        _RANDOM_32 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:132:30
        sqrSigma1_C = {_RANDOM_32[17:0], _RANDOM_31[31:17]};	// DivSqrtRecF64_mulAddZ31.scala:132:30
        _RANDOM_33 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:133:30
        _RANDOM_34 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:133:30
        sigXN_C = {_RANDOM_34[11:0], _RANDOM_33, _RANDOM_32[31:18]};	// DivSqrtRecF64_mulAddZ31.scala:133:30
        _RANDOM_35 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:134:30
        u_C_sqrt = {_RANDOM_35[10:0], _RANDOM_34[31:12]};	// DivSqrtRecF64_mulAddZ31.scala:134:30
        E_E_div = _RANDOM_35[11];	// DivSqrtRecF64_mulAddZ31.scala:135:30
        _RANDOM_36 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:136:30
        _RANDOM_37 = `RANDOM;	// DivSqrtRecF64_mulAddZ31.scala:136:30
        sigT_E = {_RANDOM_37[0], _RANDOM_36, _RANDOM_35[31:12]};	// DivSqrtRecF64_mulAddZ31.scala:136:30
        extraT_E = _RANDOM_37[1];	// DivSqrtRecF64_mulAddZ31.scala:137:30
        isNegRemT_E = _RANDOM_37[2];	// DivSqrtRecF64_mulAddZ31.scala:138:30
        isZeroRemT_E = _RANDOM_37[3];	// DivSqrtRecF64_mulAddZ31.scala:139:30
      `endif
    end // initial
  `endif
      wire _T_18 = _T_17 & ~_T_14 & ~_T_9 & ~_T_8 & ~_T_7 & ~_T_13 & ~_T_12 & ~_T_6 & ~_T_5 & ~_T_4;	// DivSqrtRecF64_mulAddZ31.scala:197:{21,38,55}, :198:{13,30,42,54}, :199:{13,22,25}, :284:28, :426:33, :431:27, :432:27, :442:39, :443:39, :444:39, :447:27, :456:27, :457:27
      wire _T_19 = _T_17 & ~_T_9 & ~_T_8 & ~_T_7 & ~_T_10 & ~_T_6;	// DivSqrtRecF64_mulAddZ31.scala:197:{38,55}, :198:{13,54}, :202:{13,26}, :284:28, :439:26, :442:39, :443:39, :444:39, :447:27
      wire _T_20 = _T_18 & io_inValid & ~io_sqrtOp;	// DivSqrtRecF64_mulAddZ31.scala:203:{52,55}
      wire _T_21 = _T_19 & io_inValid & io_sqrtOp;	// DivSqrtRecF64_mulAddZ31.scala:204:52
      wire _cyc_S = _T_20 | _T_21;	// DivSqrtRecF64_mulAddZ31.scala:205:27
      wire [2:0] _specialCodeA_S = io_a[63:61];	// DivSqrtRecF64_mulAddZ31.scala:210:32
      wire _signB_S = io_b[64];	// DivSqrtRecF64_mulAddZ31.scala:214:24
      wire [2:0] _specialCodeB_S = io_b[63:61];	// DivSqrtRecF64_mulAddZ31.scala:217:32
      wire _T_22 = io_b[63:62] != 2'h3;	// DivSqrtRecF64_mulAddZ31.scala:212:46, :219:{39,46}
      wire _T_23 = io_a[63:62] != 2'h3 & _T_22 & |_specialCodeA_S & |_specialCodeB_S;	// DivSqrtRecF64_mulAddZ31.scala:211:40, :212:{39,46}, :218:40, :224:57
      wire _T_24 = _T_22 & |_specialCodeB_S & ~_signB_S;	// DivSqrtRecF64_mulAddZ31.scala:218:40, :225:{59,62}
      wire _cyc_A4_div = _T_20 & _T_23;	// DivSqrtRecF64_mulAddZ31.scala:228:50
      wire _cyc_A7_sqrt = _T_21 & _T_24;	// DivSqrtRecF64_mulAddZ31.scala:229:50
      wire _io_b_51 = io_b[51];	// DivSqrtRecF64_mulAddZ31.scala:247:36
      wire _T_25 = specialCodeB_PA[2:1] != 2'h3;	// DivSqrtRecF64_mulAddZ31.scala:212:46, :246:25, :271:{41,48}
      wire _normalCase_PA = sqrtOp_PA ? _T_25 & |specialCodeB_PA & ~sign_PA : specialCodeA_PA[2:1] != 2'h3 & _T_25 &
                |specialCodeA_PA & |specialCodeB_PA;	// DivSqrtRecF64_mulAddZ31.scala:212:46, :244:25, :245:25, :246:25, :251:25, :266:42, :267:{41,48}, :270:42, :275:12, :276:{45,48}, :277:64
      wire _valid_normalCase_leaving_PA = _T_11 | _T_14;	// DivSqrtRecF64_mulAddZ31.scala:280:50, :426:33, :437:38
      wire _valid_leaving_PA = _normalCase_PA ? _valid_normalCase_leaving_PA : _T_16;	// DivSqrtRecF64_mulAddZ31.scala:282:12, :322:28
      wire _T_26 = valid_PA & _valid_leaving_PA;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :283:28
  assign _T_17 = ~valid_PA | _valid_leaving_PA;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :235:36, :284:28
      wire _T_27 = specialCodeB_PB[2:1] != 2'h3;	// DivSqrtRecF64_mulAddZ31.scala:212:46, :298:25, :311:{41,48}
      wire _normalCase_PB = sqrtOp_PB ? _T_27 & |specialCodeB_PB & ~sign_PB : specialCodeA_PB[2:1] != 2'h3 & _T_27 &
                |specialCodeA_PB & |specialCodeB_PB;	// DivSqrtRecF64_mulAddZ31.scala:212:46, :294:25, :295:25, :296:25, :298:25, :308:42, :309:{41,48}, :310:42, :313:12, :314:{45,48}, :315:64
      wire _valid_leaving_PB = _normalCase_PB ? _T_3 : _T_15;	// DivSqrtRecF64_mulAddZ31.scala:320:12, :382:28, :458:27
      wire _T_28 = valid_PB & _valid_leaving_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :321:28
  assign _T_16 = ~valid_PB | _valid_leaving_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :322:28
      wire [1:0] _specialCodeA_PC_2to1 = specialCodeA_PC[2:1];	// DivSqrtRecF64_mulAddZ31.scala:334:25, :347:41
      wire _specialCodeA_PC_0 = specialCodeA_PC[0];	// DivSqrtRecF64_mulAddZ31.scala:334:25, :348:59
      wire _isInfA_PC = &_specialCodeA_PC_2to1 & ~_specialCodeA_PC_0;	// DivSqrtRecF64_mulAddZ31.scala:347:48, :348:{39,42}
      wire _isNaNA_PC = &_specialCodeA_PC_2to1 & _specialCodeA_PC_0;	// DivSqrtRecF64_mulAddZ31.scala:347:48, :349:39
      wire [1:0] _specialCodeB_PC_2to1 = specialCodeB_PC[2:1];	// DivSqrtRecF64_mulAddZ31.scala:336:25, :353:41
      wire _specialCodeB_PC_0 = specialCodeB_PC[0];	// DivSqrtRecF64_mulAddZ31.scala:336:25, :354:59
      wire _isInfB_PC = &_specialCodeB_PC_2to1 & ~_specialCodeB_PC_0;	// DivSqrtRecF64_mulAddZ31.scala:353:48, :354:{39,42}
      wire _isNaNB_PC = &_specialCodeB_PC_2to1 & _specialCodeB_PC_0;	// DivSqrtRecF64_mulAddZ31.scala:353:48, :355:39
      wire _normalCase_PC = sqrtOp_PC ? ~(&_specialCodeB_PC_2to1) & |specialCodeB_PC & ~sign_PC :
                ~(&_specialCodeA_PC_2to1) & ~(&_specialCodeB_PC_2to1) & |specialCodeA_PC & |specialCodeB_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :333:25, :334:25, :336:25, :346:42, :347:48, :352:42, :353:48, :360:{12,24,56,59}, :361:{13,64}
      wire [13:0] _T_29 = exp_PC + 14'h2;	// DivSqrtRecF64_mulAddZ31.scala:341:25, :363:27
      wire _exp_PC_0 = exp_PC[0];	// DivSqrtRecF64_mulAddZ31.scala:341:25, :365:19
      wire [12:0] _T_30 = _T_29[13:1];	// DivSqrtRecF64_mulAddZ31.scala:366:25
      wire [12:0] _exp_PC_13to1 = exp_PC[13:1];	// DivSqrtRecF64_mulAddZ31.scala:341:25, :367:23
      wire [13:0] _expP1_PC = _exp_PC_0 ? {_T_30, 1'h0} : {_exp_PC_13to1, 1'h1};	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :365:12
      wire _roundMagUp_PC = sign_PC ? roundingMode_PC == 2'h2 : &roundingMode_PC;	// DivSqrtRecF64_mulAddZ31.scala:333:25, :338:25, :363:27, :372:54, :373:54, :376:12
      wire _overflowY_roundMagUp_PC = ~(|roundingMode_PC) | _roundMagUp_PC;	// DivSqrtRecF64_mulAddZ31.scala:338:25, :370:54, :377:61
      wire _valid_leaving_PC = ~_normalCase_PC | _T_2;	// DivSqrtRecF64_mulAddZ31.scala:380:{28,44}, :481:27
      wire _T_31 = valid_PC & _valid_leaving_PC;	// DivSqrtRecF64_mulAddZ31.scala:329:18, :381:28
  assign _T_15 = ~valid_PC | _valid_leaving_PC;	// DivSqrtRecF64_mulAddZ31.scala:329:18, :382:{17,28}
      wire _cyc_A6_sqrt = cycleNum_A == 3'h6;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :391:16, :396:35
      wire _cyc_A5_sqrt = cycleNum_A == 3'h5;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :397:35
      wire _cyc_A4_sqrt = cycleNum_A == 3'h4;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :398:35
      wire _cyc_A4 = _cyc_A4_sqrt | _cyc_A4_div;	// DivSqrtRecF64_mulAddZ31.scala:402:30
      wire _cyc_A3 = cycleNum_A == 3'h3;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :403:30, :827:59
      wire _cyc_A2 = cycleNum_A == 3'h2;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :404:30
      wire _cyc_A1 = cycleNum_A == 3'h1;	// DivSqrtRecF64_mulAddZ31.scala:388:49, :405:30
      wire _cyc_A3_div = _cyc_A3 & ~sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:244:25, :407:{29,32}
      wire _cyc_A1_div = _cyc_A1 & ~sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:244:25, :407:32, :409:29
      wire _cyc_A1_sqrt = _cyc_A1 & sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:244:25, :413:30
      wire _T_32 = cycleNum_B == 4'h9;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :424:33
      wire _T_33 = cycleNum_B == 4'h8;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :425:33
  assign _T_14 = cycleNum_B == 4'h7;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :426:33
      wire _T_34 = cycleNum_B == 4'h6;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :418:20, :428:27
      wire _T_35 = cycleNum_B == 4'h5;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :429:27
      wire _T_36 = cycleNum_B == 4'h4;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :430:27
  assign _T_13 = cycleNum_B == 4'h3;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :431:27
  assign _T_12 = cycleNum_B == 4'h2;	// DivSqrtRecF64_mulAddZ31.scala:415:33, :432:27
      wire _T_37 = cycleNum_B == 4'h1;	// DivSqrtRecF64_mulAddZ31.scala:392:54, :415:33, :433:27
      wire _T_38 = _T_34 & valid_PA & ~sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :244:25, :407:32, :435:38
  assign _T_11 = _T_36 & valid_PA & ~sqrtOp_PA;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :244:25, :407:32, :437:38
  assign _T_10 = _T_12 & ~sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :432:27, :438:29, :439:26
  assign _T_9 = _T_34 & valid_PB & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :294:25, :442:39
  assign _T_8 = _T_35 & valid_PB & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :294:25, :443:39
  assign _T_7 = _T_36 & valid_PB & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :294:25, :444:39
      wire _T_39 = _T_13 & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :431:27, :445:27
      wire _T_40 = _T_12 & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :432:27, :446:27
  assign _T_6 = _T_37 & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :447:27
      wire _cyc_C6_sqrt = cycleNum_C == 3'h6;	// DivSqrtRecF64_mulAddZ31.scala:391:16, :449:33, :454:35
  assign _T_5 = cycleNum_C == 3'h5;	// DivSqrtRecF64_mulAddZ31.scala:397:35, :449:33, :456:27
  assign _T_4 = cycleNum_C == 3'h4;	// DivSqrtRecF64_mulAddZ31.scala:398:35, :449:33, :457:27
  assign _T_3 = cycleNum_C == 3'h3;	// DivSqrtRecF64_mulAddZ31.scala:449:33, :458:27, :827:59
      wire _T_41 = cycleNum_C == 3'h2;	// DivSqrtRecF64_mulAddZ31.scala:404:30, :449:33, :459:27
      wire _T_42 = cycleNum_C == 3'h1;	// DivSqrtRecF64_mulAddZ31.scala:405:30, :449:33, :460:27
      wire _cyc_C1_div = _T_42 & ~sqrtOp_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :383:39, :466:29
      wire _cyc_C4_sqrt = _T_4 & sqrtOp_PB;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :457:27, :469:30
      wire _cyc_C1_sqrt = _T_42 & sqrtOp_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :472:30
      wire _T_43 = cycleNum_E == 3'h3;	// DivSqrtRecF64_mulAddZ31.scala:474:33, :479:27, :827:59
  assign _T_2 = cycleNum_E == 3'h1;	// DivSqrtRecF64_mulAddZ31.scala:405:30, :474:33, :481:27
      wire [13:0] _T_44 = _cyc_A4_div ? io_b[48:35] : 14'h0;	// DivSqrtRecF64_mulAddZ31.scala:496:29, :592:12
      wire [2:0] _io_b_51to49 = io_b[51:49];	// DivSqrtRecF64_mulAddZ31.scala:498:53
      wire _zLinPiece_0_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h0;	// DivSqrtRecF64_mulAddZ31.scala:117:30, :498:{41,62}
      wire _zLinPiece_1_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h1;	// DivSqrtRecF64_mulAddZ31.scala:405:30, :499:{41,62}
      wire _zLinPiece_2_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h2;	// DivSqrtRecF64_mulAddZ31.scala:404:30, :500:{41,62}
      wire _zLinPiece_3_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h3;	// DivSqrtRecF64_mulAddZ31.scala:501:{41,62}, :827:59
      wire _zLinPiece_4_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h4;	// DivSqrtRecF64_mulAddZ31.scala:398:35, :502:{41,62}
      wire _zLinPiece_5_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h5;	// DivSqrtRecF64_mulAddZ31.scala:397:35, :503:{41,62}
      wire _zLinPiece_6_A4_div = _cyc_A4_div & _io_b_51to49 == 3'h6;	// DivSqrtRecF64_mulAddZ31.scala:391:16, :504:{41,62}
      wire _zLinPiece_7_A4_div = _cyc_A4_div & &_io_b_51to49;	// DivSqrtRecF64_mulAddZ31.scala:505:{41,62}
      wire [8:0] _T_45 = _cyc_A7_sqrt ? io_b[50:42] : 9'h0;	// DivSqrtRecF64_mulAddZ31.scala:507:12, :525:30
      wire _io_b_52 = io_b[52];	// DivSqrtRecF64_mulAddZ31.scala:527:55
      wire _T_46 = _cyc_A7_sqrt & ~_io_b_52;	// DivSqrtRecF64_mulAddZ31.scala:527:{44,47}
      wire _zQuadPiece_0_A7_sqrt = _T_46 & ~_io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:527:{59,62}
      wire _zQuadPiece_1_A7_sqrt = _T_46 & _io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:528:59
      wire _T_47 = _cyc_A7_sqrt & _io_b_52;	// DivSqrtRecF64_mulAddZ31.scala:529:44
      wire _zQuadPiece_2_A7_sqrt = _T_47 & ~_io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:527:62, :529:59
      wire _zQuadPiece_3_A7_sqrt = _T_47 & _io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:530:59
      wire _exp_PA_0 = exp_PA[0];	// DivSqrtRecF64_mulAddZ31.scala:255:16, :542:55
      wire _T_48 = _cyc_A6_sqrt & ~_exp_PA_0;	// DivSqrtRecF64_mulAddZ31.scala:542:{44,47}
      wire _T_49 = _cyc_A6_sqrt & _exp_PA_0;	// DivSqrtRecF64_mulAddZ31.scala:544:44
      wire [19:0] _T_50 = {(_zQuadPiece_0_A7_sqrt ? 10'h2F : 10'h0) | (_zQuadPiece_1_A7_sqrt ? 10'h1DF : 10'h0) |
                (_zQuadPiece_2_A7_sqrt ? 10'h14D : 10'h0) | (_zQuadPiece_3_A7_sqrt ? 10'h27E : 10'h0),
                {10{_cyc_A7_sqrt}}} | {_cyc_A6_sqrt, (_T_48 & ~fractB_51_PA ? 13'h1A : 13'h0) | (_T_48 &
                fractB_51_PA ? 13'hBCA : 13'h0) | (_T_49 & ~fractB_51_PA ? 13'h12D3 : 13'h0) | (_T_49 &
                fractB_51_PA ? 13'h1B17 : 13'h0), {6{_cyc_A6_sqrt}}};	// Bitwise.scala:71:12, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:247:25, :537:{12,35}, :538:{12,35}, :539:{12,35,63}, :540:{12,35}, :542:{59,62}, :543:59, :544:59, :545:59, :547:{12,35}, :548:{12,35}, :549:{12,35,64}, :550:{12,35}, :559:71
      wire [19:0] _T_51 = _cyc_A5_sqrt ? {1'h0, fractR0_A, 10'h0} + 20'h40000 : 20'h0;	// DivSqrtRecF64_mulAddZ31.scala:78:30, :537:12, :563:{12,42,54}
      wire [11:0] _T_52 = _T_50[19:8] | (_zLinPiece_0_A4_div ? 12'h1C : 12'h0) | (_zLinPiece_1_A4_div ? 12'h3A2 :
                12'h0) | (_zLinPiece_2_A4_div ? 12'h675 : 12'h0) | (_zLinPiece_3_A4_div ? 12'h8C6 : 12'h0)
                | (_zLinPiece_4_A4_div ? 12'hAB4 : 12'h0) | (_zLinPiece_5_A4_div ? 12'hC56 : 12'h0) |
                (_zLinPiece_6_A4_div ? 12'hDBD : 12'h0) | (_zLinPiece_7_A4_div ? 12'hEF4 : 12'h0) |
                _T_51[19:8];	// DivSqrtRecF64_mulAddZ31.scala:516:{12,33}, :517:{12,33}, :518:{12,33}, :519:{12,33}, :520:{12,33}, :521:{12,33}, :522:{12,33}, :523:{12,33}, :560:71, :561:71
      wire _hiSqrR0_A_sqrt_9 = hiSqrR0_A_sqrt[9];	// DivSqrtRecF64_mulAddZ31.scala:564:44
      wire [23:0] _T_53 = _cyc_A1_div ? {fractR0_A, 15'h0} : 24'h0;	// DivSqrtRecF64_mulAddZ31.scala:563:54, :571:{12,45}, :583:12
      wire [24:0] _T_54 = (_cyc_A1_sqrt ? {fractR0_A, 16'h0} : 25'h0) | {1'h0, _T_53[23:21], {_cyc_A4_div,
                _T_52[11:3], _T_52[2] | _cyc_A4_sqrt & ~_hiSqrR0_A_sqrt_9, _T_52[1:0], _T_50[7:0] |
                {8{_cyc_A4_div}} | _T_51[7:0]} | (_cyc_A4_sqrt & _hiSqrR0_A_sqrt_9 | _cyc_A3_div ?
                fractB_other_PA[46:26] + 21'h400 : 21'h0) | (_cyc_A3 & sqrtOp_PA | _cyc_A2 ?
                partNegSigma0_A : 21'h0) | _T_53[20:0]};	// Bitwise.scala:71:12, DivSqrtRecF64_mulAddZ31.scala:78:30, :244:25, :260:25, :411:30, :560:71, :561:71, :563:{54,70}, :564:{25,28}, :565:{12,26,48}, :566:{20,29}, :569:{12,25}, :570:{12,45,62}
      wire [18:0] _T_55 = {1'h0, {9'h0, _T_44[13:5] | (_zQuadPiece_0_A7_sqrt ? 9'h1C8 : 9'h0) |
                (_zQuadPiece_1_A7_sqrt ? 9'hC1 : 9'h0) | (_zQuadPiece_2_A7_sqrt ? 9'h143 : 9'h0) |
                (_zQuadPiece_3_A7_sqrt ? 9'h89 : 9'h0) | (_cyc_S ? 9'h0 : nextMulAdd9A_A)} * {9'h0,
                (_zLinPiece_0_A4_div ? 9'h1C7 : 9'h0) | (_zLinPiece_1_A4_div ? 9'h16C : 9'h0) |
                (_zLinPiece_2_A4_div ? 9'h12A : 9'h0) | (_zLinPiece_3_A4_div ? 9'hF8 : 9'h0) |
                (_zLinPiece_4_A4_div ? 9'hD2 : 9'h0) | (_zLinPiece_5_A4_div ? 9'hB4 : 9'h0) |
                (_zLinPiece_6_A4_div ? 9'h9C : 9'h0) | (_zLinPiece_7_A4_div ? 9'h89 : 9'h0) | _T_45 |
                (_cyc_S ? 9'h0 : nextMulAdd9B_A)}} + {1'h0, _T_54[17:0]};	// DivSqrtRecF64_mulAddZ31.scala:78:30, :507:12, :508:12, :509:12, :510:12, :511:12, :512:12, :513:12, :514:12, :532:12, :533:12, :534:12, :535:12, :553:{23,46}, :554:16, :556:46, :557:16, :573:{20,33,61}
      wire [6:0] _T_56 = _T_54[24:18];	// DivSqrtRecF64_mulAddZ31.scala:576:27
      wire [6:0] _T_57 = _T_55[18] ? _T_56 + 7'h1 : _T_56;	// DivSqrtRecF64_mulAddZ31.scala:575:{16,31}, :576:36
      wire [8:0] _T_58 = _T_55[17:9];	// DivSqrtRecF64_mulAddZ31.scala:601:54
      wire [14:0] _T_59 = sqrtOp_PA ? {_T_57, _T_55[17:10]} : {_T_57[5:0], _T_58};	// DivSqrtRecF64_mulAddZ31.scala:244:25, :601:{12,36}
      wire [16:0] _ER1_A1_sqrt = _exp_PA_0 ? {1'h1, _T_59, 1'h0} : {2'h1, _T_59};	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :603:{26,43}
      wire _T_60 = _cyc_A1 | _T_14;	// DivSqrtRecF64_mulAddZ31.scala:426:33, :647:16
      wire _T_61 = _T_60 | _T_38 | _T_36 | _T_13 | _cyc_C6_sqrt | _T_4 | _T_42;	// DivSqrtRecF64_mulAddZ31.scala:431:27, :457:27, :648:35
      wire [45:0] _T_62 = _T_13 | _cyc_C6_sqrt ? io_mulAddResult_3[104:59] : 46'h0;	// DivSqrtRecF64_mulAddZ31.scala:431:27, :655:{12,20,48}
      wire [45:0] _T_63 = _T_4 & ~sqrtOp_PB ? {sigXN_C[57:25], 13'h0} : 46'h0;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :438:29, :457:27, :463:29, :547:12, :655:12, :656:{12,43,51}
      wire [45:0] _T_64 = _cyc_C4_sqrt ? {u_C_sqrt, 15'h0} : 46'h0;	// DivSqrtRecF64_mulAddZ31.scala:583:12, :655:12, :657:{12,44}
      wire [51:0] _T_65 = (_cyc_A1 ? {1'h1, _T_59, 36'h0} : 52'h0) | {1'h0, _T_14 ? {ESqrR1_B_sqrt, 19'h0} : 51'h0};	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :426:33, :496:29, :650:51, :652:67, :664:{12,31,55}, :665:{12,39}
      wire [45:0] _T_66 = _T_65[45:0] | _T_1;	// DivSqrtRecF64_mulAddZ31.scala:666:55, :702:22
      wire [32:0] _T_67 = _T_4 ? sqrSigma1_C : 33'h0;	// DivSqrtRecF64_mulAddZ31.scala:457:27, :668:37, :669:12
      wire [103:0] _T_68 = _cyc_C6_sqrt ? {sigX1_B, 46'h0} : 104'h0;	// DivSqrtRecF64_mulAddZ31.scala:655:12, :688:45, :689:{12,45}
      wire _fractB_other_PC_0 = fractB_other_PC[0];	// DivSqrtRecF64_mulAddZ31.scala:343:25, :694:29
  assign _T_1 = _T_36 ? ~(io_mulAddResult_3[90:45]) : 46'h0;	// DivSqrtRecF64_mulAddZ31.scala:655:12, :702:{22,31,49}
      wire _io_mulAddResult_3_104 = io_mulAddResult_3[104];	// DivSqrtRecF64_mulAddZ31.scala:705:39
      wire [53:0] _io_mulAddResult_3_104to51 = io_mulAddResult_3[104:51];	// DivSqrtRecF64_mulAddZ31.scala:708:31
  assign _T_0 = (_cyc_C1_div & _io_mulAddResult_3_104 | _cyc_C1_sqrt ? ~_io_mulAddResult_3_104to51 : 54'h0)
                | (_cyc_C1_div & ~_io_mulAddResult_3_104 ? {1'h0, ~(io_mulAddResult_3[102:50])} : 54'h0);	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :691:12, :705:20, :707:{12,25,40}, :708:13, :710:11, :711:{12,24}, :712:{29,47}
  assign _T = _cyc_C1_sqrt ? ~_io_mulAddResult_3_104to51 : 54'h0;	// DivSqrtRecF64_mulAddZ31.scala:691:12, :708:13, :716:12
      wire [53:0] _sigT_C1 = ~_T_0;	// DivSqrtRecF64_mulAddZ31.scala:710:11, :720:19
      wire _T_69 = ~io_sqrtOp & io_a[64] ^ _signB_S;	// DivSqrtRecF64_mulAddZ31.scala:207:24, :221:21
      wire _entering_PA_normalCase = _cyc_A4_div | _cyc_A7_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:231:36
      wire _entering_PA = _entering_PA_normalCase | _cyc_S & (valid_PA | ~_T_16);	// DivSqrtRecF64_mulAddZ31.scala:233:{32,42,55,58}, :322:28
      wire _T_70 = _cyc_S & ~(io_sqrtOp ? _T_24 : _T_23) & ~valid_PA;	// DivSqrtRecF64_mulAddZ31.scala:226:27, :233:55, :235:{18,33,36}
      wire _io_a_51 = io_a[51];	// DivSqrtRecF64_mulAddZ31.scala:252:36
      wire _entering_PB = _T_70 & (_T_28 | ~valid_PB & ~_T_15) | _T_26;	// DivSqrtRecF64_mulAddZ31.scala:235:47, :236:{25,29,40,43}, :288:37, :382:28
      wire _entering_PC = _T_70 & ~valid_PB & _T_15 | _T_28;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :238:61, :326:37, :382:28
      wire [17:0] _T_71 = _T_55[17:0];	// DivSqrtRecF64_mulAddZ31.scala:579:27
      wire [17:0] _T_72 = ~{_T_57[1:0], _T_55[17:2]};	// DivSqrtRecF64_mulAddZ31.scala:584:13
      wire [8:0] _T_73 = _cyc_A6_sqrt & _T_57[1] ? _T_72[16:8] : 9'h0;	// DivSqrtRecF64_mulAddZ31.scala:507:12, :583:{12,25,40}
      wire [18:0] _T_74 = _exp_PA_0 ? {_T_57[0], _T_71} : {_T_57[1:0], _T_55[17:1]};	// DivSqrtRecF64_mulAddZ31.scala:590:28
      wire [8:0] _T_75 = _cyc_A4_div & _T_57[2] ? _T_72[17:9] : 9'h0;	// DivSqrtRecF64_mulAddZ31.scala:507:12, :592:{12,24,39}
      wire _T_76 = _cyc_A7_sqrt | _cyc_A6_sqrt | _cyc_A5_sqrt | _cyc_A4;	// DivSqrtRecF64_mulAddZ31.scala:620:51
      wire [57:0] _io_mulAddResult_3_104to47 = io_mulAddResult_3[104:47];	// DivSqrtRecF64_mulAddZ31.scala:704:38
  always @(posedge clock) begin	// DivSqrtRecF64_mulAddZ31.scala:244:25
    if (reset) begin	// DivSqrtRecF64_mulAddZ31.scala:78:30
      valid_PA <= 1'h0;	// DivSqrtRecF64_mulAddZ31.scala:78:30
      valid_PB <= 1'h0;	// DivSqrtRecF64_mulAddZ31.scala:78:30, :91:30
      valid_PC <= 1'h0;	// DivSqrtRecF64_mulAddZ31.scala:78:30, :104:30
      cycleNum_A <= 3'h0;	// DivSqrtRecF64_mulAddZ31.scala:117:30
      cycleNum_B <= 4'h0;	// DivSqrtRecF64_mulAddZ31.scala:118:30
      cycleNum_C <= 3'h0;	// DivSqrtRecF64_mulAddZ31.scala:117:30, :119:30
      cycleNum_E <= 3'h0;	// DivSqrtRecF64_mulAddZ31.scala:117:30, :120:30
    end
    else begin	// DivSqrtRecF64_mulAddZ31.scala:78:30
      valid_PA <= _entering_PA | _T_26 ? _entering_PA : valid_PA;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :240:23, :241:18
      valid_PB <= _entering_PB | _T_28 ? _entering_PB : valid_PB;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :290:23, :291:18
      valid_PC <= _entering_PC | _T_31 ? _entering_PC : valid_PC;	// DivSqrtRecF64_mulAddZ31.scala:328:23, :329:18
      cycleNum_A <= _entering_PA_normalCase | |cycleNum_A ? {1'h0, {2{_cyc_A4_div}}} | (_cyc_A7_sqrt ? 3'h6 :
                                                3'h0) | (_entering_PA_normalCase ? 3'h0 : cycleNum_A - 3'h1) : cycleNum_A;	// DivSqrtRecF64_mulAddZ31.scala:78:30, :117:30, :388:{34,49}, :389:20, :390:{16,74}, :391:{16,74}, :392:{16,54}
      cycleNum_B <= _cyc_A1 | |cycleNum_B ? (_cyc_A1 ? (sqrtOp_PA ? 4'hA : 4'h6) : cycleNum_B - 4'h1) :
                                                cycleNum_B;	// DivSqrtRecF64_mulAddZ31.scala:244:25, :415:{18,33}, :416:20, :417:16, :418:20, :419:28
      cycleNum_C <= _T_37 | |cycleNum_C ? (_T_37 ? (sqrtOp_PB ? 3'h6 : 3'h5) : cycleNum_C - 3'h1) : cycleNum_C;	// DivSqrtRecF64_mulAddZ31.scala:294:25, :391:16, :397:35, :449:{18,33}, :450:20, :451:{16,28,70}
      cycleNum_E <= _T_42 | |cycleNum_E ? (_T_42 ? 3'h4 : cycleNum_E - 3'h1) : cycleNum_E;	// DivSqrtRecF64_mulAddZ31.scala:398:35, :474:{18,33}, :475:{20,26,55}
    end
    if (_entering_PA) begin	// DivSqrtRecF64_mulAddZ31.scala:248:25
      sqrtOp_PA <= io_sqrtOp;	// DivSqrtRecF64_mulAddZ31.scala:244:25
      sign_PA <= _T_69;	// DivSqrtRecF64_mulAddZ31.scala:245:25
      specialCodeB_PA <= _specialCodeB_S;	// DivSqrtRecF64_mulAddZ31.scala:246:25
      fractB_51_PA <= _io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:247:25
      roundingMode_PA <= io_roundingMode;	// DivSqrtRecF64_mulAddZ31.scala:248:25
    end
    if (_entering_PA & ~io_sqrtOp) begin	// DivSqrtRecF64_mulAddZ31.scala:203:55, :250:23, :252:25
      specialCodeA_PA <= _specialCodeA_S;	// DivSqrtRecF64_mulAddZ31.scala:251:25
      fractA_51_PA <= _io_a_51;	// DivSqrtRecF64_mulAddZ31.scala:252:25
    end
    if (_entering_PA_normalCase) begin	// DivSqrtRecF64_mulAddZ31.scala:260:25
      exp_PA <= io_sqrtOp ? {2'h0, io_b[63:52]} : {2'h0, io_a[63:52]} + {{3{io_b[63]}}, ~(io_b[62:52])};	// Bitwise.scala:71:12, DivSqrtRecF64_mulAddZ31.scala:208:24, :215:24, :255:16, :256:16, :258:{24,44,51,58}, :390:16
      fractB_other_PA <= io_b[50:0];	// DivSqrtRecF64_mulAddZ31.scala:260:{25,36}
    end
    if (_cyc_A4_div)	// DivSqrtRecF64_mulAddZ31.scala:263:25
      fractA_other_PA <= io_a[50:0];	// DivSqrtRecF64_mulAddZ31.scala:263:{25,36}
    if (_entering_PB) begin	// DivSqrtRecF64_mulAddZ31.scala:300:25
      sqrtOp_PB <= valid_PA ? sqrtOp_PA : io_sqrtOp;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :244:25, :294:{25,31}
      sign_PB <= valid_PA ? sign_PA : _T_69;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :245:25, :295:{25,31}
      specialCodeA_PB <= valid_PA ? specialCodeA_PA : _specialCodeA_S;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :251:25, :296:{25,31}
      fractA_51_PB <= valid_PA ? fractA_51_PA : _io_a_51;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :252:25, :297:{25,31}
      specialCodeB_PB <= valid_PA ? specialCodeB_PA : _specialCodeB_S;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :246:25, :298:{25,31}
      fractB_51_PB <= valid_PA ? fractB_51_PA : _io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :247:25, :299:{25,31}
      roundingMode_PB <= valid_PA ? roundingMode_PA : io_roundingMode;	// DivSqrtRecF64_mulAddZ31.scala:233:55, :248:25, :300:{25,31}
    end
    if (valid_PA & _normalCase_PA & _valid_normalCase_leaving_PA) begin	// DivSqrtRecF64_mulAddZ31.scala:233:55, :287:35, :305:25
      exp_PB <= exp_PA;	// DivSqrtRecF64_mulAddZ31.scala:255:16, :303:25
      fractA_0_PB <= fractA_other_PA[0];	// DivSqrtRecF64_mulAddZ31.scala:263:25, :304:{25,43}
      fractB_other_PB <= fractB_other_PA;	// DivSqrtRecF64_mulAddZ31.scala:260:25, :305:25
    end
    if (_entering_PC) begin	// DivSqrtRecF64_mulAddZ31.scala:338:25
      sqrtOp_PC <= valid_PB ? sqrtOp_PB : io_sqrtOp;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :294:25, :332:{25,31}
      sign_PC <= valid_PB ? sign_PB : _T_69;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :295:25, :333:{25,31}
      specialCodeA_PC <= valid_PB ? specialCodeA_PB : _specialCodeA_S;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :296:25, :334:{25,31}
      fractA_51_PC <= valid_PB ? fractA_51_PB : _io_a_51;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :297:25, :335:{25,31}
      specialCodeB_PC <= valid_PB ? specialCodeB_PB : _specialCodeB_S;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :298:25, :336:{25,31}
      fractB_51_PC <= valid_PB ? fractB_51_PB : _io_b_51;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :299:25, :337:{25,31}
      roundingMode_PC <= valid_PB ? roundingMode_PB : io_roundingMode;	// DivSqrtRecF64_mulAddZ31.scala:236:29, :300:25, :338:{25,31}
    end
    if (valid_PB & _normalCase_PB & _T_3) begin	// DivSqrtRecF64_mulAddZ31.scala:236:29, :325:35, :343:25, :458:27
      exp_PC <= exp_PB;	// DivSqrtRecF64_mulAddZ31.scala:303:25, :341:25
      fractA_0_PC <= fractA_0_PB;	// DivSqrtRecF64_mulAddZ31.scala:304:25, :342:25
      fractB_other_PC <= fractB_other_PB;	// DivSqrtRecF64_mulAddZ31.scala:305:25, :343:25
    end
    if (_cyc_A6_sqrt | _cyc_A4_div)	// DivSqrtRecF64_mulAddZ31.scala:605:23, :606:19
      fractR0_A <= _T_73 | _T_75;	// DivSqrtRecF64_mulAddZ31.scala:606:{19,39}
    if (_cyc_A5_sqrt)	// DivSqrtRecF64_mulAddZ31.scala:610:24
      hiSqrR0_A_sqrt <= _T_74[18:9];	// DivSqrtRecF64_mulAddZ31.scala:610:24
    if (_cyc_A4_sqrt | _cyc_A3)	// DivSqrtRecF64_mulAddZ31.scala:613:23, :615:25
      partNegSigma0_A <= _cyc_A4_sqrt ? {_T_57[2:0], _T_71} : {5'h0, _T_57, _T_58};	// DivSqrtRecF64_mulAddZ31.scala:615:25, :616:16, :623:68
    if (_T_76 | _cyc_A3 | _cyc_A2)	// DivSqrtRecF64_mulAddZ31.scala:620:71, :622:24
      nextMulAdd9A_A <= (_cyc_A7_sqrt ? _T_72[17:9] : 9'h0) | _T_73 | (_cyc_A4_sqrt ? fractB_other_PA[43:35] :
                                                9'h0) | _T_44[8:0] | (_cyc_A5_sqrt | _cyc_A3 ? {1'h1, fractB_51_PA, fractB_other_PA[50:44]}
                                                : 9'h0) | (_cyc_A2 & _T_55[11] ? _T_72[8:0] : 9'h0);	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:247:25, :260:25, :507:12, :598:{12,20,35}, :622:24, :623:16, :625:{16,47}, :626:27, :627:{16,29,47,68}
    if (_T_76 | _cyc_A2)	// DivSqrtRecF64_mulAddZ31.scala:630:63, :631:24
      nextMulAdd9B_A <= _T_45 | _T_73 | (_cyc_A5_sqrt ? _T_74[8:0] : 9'h0) | _T_75 | (_cyc_A4_sqrt ?
                                                hiSqrR0_A_sqrt[8:0] : 9'h0) | (_cyc_A2 ? {1'h1, fractR0_A[8:1]} : 9'h0);	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:507:12, :563:54, :564:44, :631:24, :634:{16,43}, :636:{16,44,73}, :637:{16,55}
    if (_cyc_A1_sqrt)	// DivSqrtRecF64_mulAddZ31.scala:641:20
      ER1_B_sqrt <= _ER1_A1_sqrt;	// DivSqrtRecF64_mulAddZ31.scala:641:20
    if (_T_33)	// DivSqrtRecF64_mulAddZ31.scala:724:23
      ESqrR1_B_sqrt <= io_mulAddResult_3[103:72];	// DivSqrtRecF64_mulAddZ31.scala:701:43, :724:23
    if (_T_13)	// DivSqrtRecF64_mulAddZ31.scala:431:27, :727:17
      sigX1_B <= _io_mulAddResult_3_104to47;	// DivSqrtRecF64_mulAddZ31.scala:727:17
    if (_T_37)	// DivSqrtRecF64_mulAddZ31.scala:730:21
      sqrSigma1_C <= io_mulAddResult_3[79:47];	// DivSqrtRecF64_mulAddZ31.scala:703:41, :730:21
    if (_cyc_C6_sqrt | _T_5 & ~sqrtOp_PB | _T_3 & sqrtOp_PB)	// DivSqrtRecF64_mulAddZ31.scala:294:25, :438:29, :456:27, :458:27, :462:29, :470:30, :733:37, :734:17
      sigXN_C <= _io_mulAddResult_3_104to47;	// DivSqrtRecF64_mulAddZ31.scala:734:17
    if (_T_5 & sqrtOp_PB)	// DivSqrtRecF64_mulAddZ31.scala:294:25, :456:27, :468:30, :737:18
      u_C_sqrt <= io_mulAddResult_3[103:73];	// DivSqrtRecF64_mulAddZ31.scala:737:{18,33}
    if (_T_42) begin	// DivSqrtRecF64_mulAddZ31.scala:742:18
      E_E_div <= ~_io_mulAddResult_3_104;	// DivSqrtRecF64_mulAddZ31.scala:705:20, :740:18
      sigT_E <= _sigT_C1[53:1];	// DivSqrtRecF64_mulAddZ31.scala:741:{18,28}
      extraT_E <= _sigT_C1[0];	// DivSqrtRecF64_mulAddZ31.scala:742:{18,28}
    end
    if (cycleNum_E == 3'h2) begin	// DivSqrtRecF64_mulAddZ31.scala:404:30, :474:33, :480:27, :747:22
      isNegRemT_E <= sqrtOp_PC ? io_mulAddResult_3[55] : io_mulAddResult_3[53];	// DivSqrtRecF64_mulAddZ31.scala:332:25, :746:{21,27,47,61}
      isZeroRemT_E <= io_mulAddResult_3[53:0] == 54'h0 & (~sqrtOp_PC | io_mulAddResult_3[55:54] == 2'h0);	// DivSqrtRecF64_mulAddZ31.scala:332:25, :383:39, :390:16, :691:12, :747:22, :748:{21,29,42}, :749:{30,41,50}
    end
  end // always @(posedge)
      wire [13:0] _T_77 = (~sqrtOp_PC & E_E_div ? exp_PC : 14'h0) | (sqrtOp_PC | E_E_div ? 14'h0 : _expP1_PC) |
                {1'h0, sqrtOp_PC ? _exp_PC_13to1 + 13'h400 : 13'h0};	// DivSqrtRecF64_mulAddZ31.scala:78:30, :332:25, :341:25, :383:39, :547:12, :592:12, :691:27, :755:{12,25}, :756:{12,76}, :757:{12,47}
      wire [12:0] _posExpX_E = _T_77[12:0];	// DivSqrtRecF64_mulAddZ31.scala:759:28
      wire [12:0] _T_78 = ~_posExpX_E;	// primitives.scala:50:21
      wire _T_79 = _T_78[9];	// primitives.scala:56:25
      wire _T_80 = _T_78[8];	// primitives.scala:56:25
      wire _T_81 = _T_78[7];	// primitives.scala:56:25
      wire _T_82 = _T_78[6];	// primitives.scala:56:25
      wire [64:0] _T_83 = $signed(65'sh10000000000000000 >>> _T_78[5:0]);	// primitives.scala:57:26, :68:52
      wire [52:0] _T_84 = _T_78[12] & _T_78[11] ? (_T_78[10] ? {~(_T_79 | _T_80 | _T_81 | _T_82 ? 50'h0 :
                ~{_T_83[14], _T_83[15], _T_83[16], _T_83[17], _T_83[18], _T_83[19], _T_83[20], _T_83[21],
                _T_83[22], _T_83[23], _T_83[24], _T_83[25], _T_83[26], _T_83[27], _T_83[28], _T_83[29],
                _T_83[30], _T_83[31], _T_83[32], _T_83[33], _T_83[34], _T_83[35], _T_83[36], _T_83[37],
                _T_83[38], _T_83[39], _T_83[40], _T_83[41], _T_83[42], _T_83[43], _T_83[44], _T_83[45],
                _T_83[46], _T_83[47], _T_83[48], _T_83[49], _T_83[50], _T_83[51], _T_83[52], _T_83[53],
                _T_83[54], _T_83[55], _T_83[56], _T_83[57], _T_83[58], _T_83[59], _T_83[60], _T_83[61],
                _T_83[62], _T_83[63]}), 3'h7} : {50'h0, _T_79 & _T_80 & _T_81 & _T_82 ? {_T_83[0],
                _T_83[1], _T_83[2]} : 3'h0}) : 53'h0;	// Bitwise.scala:71:12, :102:{21,46}, :108:{18,44}, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:117:30, :650:12, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
      wire [53:0] _incrPosMask_E = {1'h1, ~_T_84} & {_T_84, 1'h1};	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:763:{9,39}
      wire [52:0] _T_85 = sigT_E & _incrPosMask_E[53:1];	// DivSqrtRecF64_mulAddZ31.scala:741:18, :765:{36,51}
      wire [51:0] _T_86 = ~(sigT_E[51:0]) & _T_84[52:1];	// DivSqrtRecF64_mulAddZ31.scala:741:18, :766:55, :767:{34,42}
      wire _T_87 = _T_84[0];	// DivSqrtRecF64_mulAddZ31.scala:769:23
      wire _all1sHiRoundT_E = (~_T_87 | |_T_85) & ~(|_T_86);	// DivSqrtRecF64_mulAddZ31.scala:765:56, :767:60, :769:{10,27,48}
      wire [53:0] _T_88 = {1'h0, sigT_E} + {53'h0, _roundMagUp_PC};	// DivSqrtRecF64_mulAddZ31.scala:78:30, :650:12, :741:18, :773:42
      wire _trueLtX_E1 = sqrtOp_PC ? ~isNegRemT_E & ~isZeroRemT_E : isNegRemT_E;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :746:21, :747:22, :783:{12,24,38,41}
      wire _hiRoundPosBit_E1 = |_T_85 ^ _T_87 & ~_trueLtX_E1 & ~(|_T_86) & extraT_E;	// DivSqrtRecF64_mulAddZ31.scala:696:22, :765:56, :767:60, :792:26, :793:{32,69}
      wire _T_89 = ~isZeroRemT_E | ~extraT_E | |_T_86;	// DivSqrtRecF64_mulAddZ31.scala:696:22, :747:22, :767:60, :783:41, :795:55
      wire _T_90 = extraT_E & ~_trueLtX_E1;	// DivSqrtRecF64_mulAddZ31.scala:696:22, :793:32, :806:29
      wire [53:0] _sigY_E1 = (~_roundMagUp_PC & |roundingMode_PC & extraT_E & ~_trueLtX_E1 & _all1sHiRoundT_E |
                _roundMagUp_PC & (_T_90 & ~isZeroRemT_E | ~_all1sHiRoundT_E) | ~(|roundingMode_PC) &
                (|_T_85 | (extraT_E | ~_trueLtX_E1) & ~_T_87 | _T_90 & ~(|_T_86)) ? (_T_88 | {1'h0, _T_84})
                + 54'h1 : _T_88 & {1'h1, ~_T_84}) & ~(~(|roundingMode_PC) & _hiRoundPosBit_E1 & ~_T_89 ?
                _incrPosMask_E : 54'h0);	// Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :338:25, :370:54, :378:27, :691:12, :696:22, :747:22, :765:56, :767:60, :769:10, :774:{29,47}, :775:{30,62}, :783:41, :793:32, :797:{12,59}, :798:17, :804:{12,58}, :805:28, :806:{45,62}, :807:{23,43}, :808:40, :810:{34,51,72}, :811:49, :814:{11,13}
      wire _inexactY_E1 = _hiRoundPosBit_E1 | _T_89;	// DivSqrtRecF64_mulAddZ31.scala:816:40
      wire _sigY_E1_53 = _sigY_E1[53];	// DivSqrtRecF64_mulAddZ31.scala:818:22
      wire _T_91 = _sigY_E1_53 & ~sqrtOp_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :383:39, :819:25
      wire [13:0] _T_92 = (_sigY_E1_53 ? 14'h0 : _T_77) | (_T_91 & E_E_div ? _expP1_PC : 14'h0) | (_T_91 & ~E_E_div ?
                _T_29 : 14'h0) | {1'h0, _sigY_E1_53 & sqrtOp_PC ? _T_30 + 13'h400 : 13'h0};	// DivSqrtRecF64_mulAddZ31.scala:78:30, :332:25, :547:12, :592:12, :691:27, :757:47, :818:12, :819:{12,40}, :820:{12,40,73}, :821:{12,25}, :822:27
      wire _T_93 = _T_92[13];	// DivSqrtRecF64_mulAddZ31.scala:827:34
      wire _totalUnderflowY_E1 = _T_93 | _T_92[12:0] < 13'h3CE;	// DivSqrtRecF64_mulAddZ31.scala:830:{22,34,42}
      wire _notSigNaN_invalid_PC = sqrtOp_PC ? ~_isNaNB_PC & |specialCodeB_PC & sign_PC : ~(|specialCodeA_PC) &
                ~(|specialCodeB_PC) | _isInfA_PC & _isInfB_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :333:25, :334:25, :336:25, :346:42, :352:42, :838:12, :839:{13,41}, :840:{25,40,54}
      wire _T_94 = _normalCase_PC & ~_T_93 & _T_92[12:10] > 3'h2;	// DivSqrtRecF64_mulAddZ31.scala:404:30, :827:{24,59,70}, :847:37
      wire _underflow_E1 = _normalCase_PC & (_totalUnderflowY_E1 | _posExpX_E < 13'h402 & _inexactY_E1);	// DivSqrtRecF64_mulAddZ31.scala:832:28, :833:{25,56}, :848:38
      wire _notSpecial_isZeroOut_E1 = sqrtOp_PC ? ~(|specialCodeB_PC) : ~(|specialCodeA_PC) | _isInfB_PC | _totalUnderflowY_E1 &
                ~_roundMagUp_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :334:25, :336:25, :346:42, :352:42, :378:27, :855:12, :857:{37,60}
      wire _T_95 = _normalCase_PC & _totalUnderflowY_E1 & _roundMagUp_PC;	// DivSqrtRecF64_mulAddZ31.scala:860:45
      wire _pegMaxFiniteMagOut_E1 = _T_94 & ~_overflowY_roundMagUp_PC;	// DivSqrtRecF64_mulAddZ31.scala:861:{45,48}
      wire _notNaN_isInfOut_E1 = sqrtOp_PC ? _isInfB_PC : _isInfA_PC | ~(|specialCodeB_PC) | _T_94 &
                _overflowY_roundMagUp_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :336:25, :352:42, :863:12, :865:{37,53}
      wire _T_96 = ~sqrtOp_PC & _isNaNA_PC | _isNaNB_PC | _notSigNaN_invalid_PC;	// DivSqrtRecF64_mulAddZ31.scala:332:25, :383:39, :868:{22,49}
  assign io_inReady_div = _T_18;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10
  assign io_inReady_sqrt = _T_19;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10
  assign io_outValid_div = _T_31 & ~sqrtOp_PC;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, DivSqrtRecF64_mulAddZ31.scala:332:25, :383:{36,39}
  assign io_outValid_sqrt = _T_31 & sqrtOp_PC;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, DivSqrtRecF64_mulAddZ31.scala:332:25, :384:36
  assign io_out = {~_T_96 & (~sqrtOp_PC | ~(|specialCodeB_PC)) & sign_PC, _T_92[11:0] &
                ~(_notSpecial_isZeroOut_E1 ? 12'hE00 : 12'h0) & ~(_T_95 ? 12'hC31 : 12'h0) & {1'h1,
                ~_pegMaxFiniteMagOut_E1, 10'h3FF} & {2'h3, ~_notNaN_isInfOut_E1, 9'h1FF} | (_T_95 ? 12'h3CE
                : 12'h0) | (_pegMaxFiniteMagOut_E1 ? 12'hBFF : 12'h0) | (_notNaN_isInfOut_E1 ? 12'hC00 :
                12'h0) | (_T_96 ? 12'hE00 : 12'h0), (_notSpecial_isZeroOut_E1 | _totalUnderflowY_E1 | _T_96
                ? {_T_96, 51'h0} : _sigY_E1[51:0]) | {52{_pegMaxFiniteMagOut_E1}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, Bitwise.scala:71:12, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:212:46, :332:25, :333:25, :336:25, :352:42, :516:12, :665:12, :815:28, :825:27, :871:{9,23,29}, :874:{14,18}, :875:19, :878:{14,18}, :879:19, :882:{14,18}, :883:19, :885:16, :886:{14,18}, :890:16, :891:16, :892:{16,76}, :893:16, :895:{12,59}, :896:16, :898:11
  assign io_exceptionFlags = {~sqrtOp_PC & _isNaNA_PC & ~fractA_51_PC | _isNaNB_PC & ~fractB_51_PC |
                _notSigNaN_invalid_PC, ~sqrtOp_PC & ~(&_specialCodeA_PC_2to1) & |specialCodeA_PC &
                ~(|specialCodeB_PC), _T_94, _underflow_E1, _T_94 | _underflow_E1 | _normalCase_PC &
                _inexactY_E1};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:332:25, :334:25, :335:25, :336:25, :337:25, :346:42, :347:48, :350:38, :352:42, :356:{35,38}, :361:13, :383:39, :843:{22,55}, :845:56, :852:{37,55}
  assign io_usingMulAdd = {_cyc_A4 | _cyc_A3_div | _cyc_A1_div | cycleNum_B == 4'hA | _T_32 | _T_14 | _T_34 | _T_8 |
                _T_39 | _T_10 | _T_6 | _T_4, _cyc_A3 | _cyc_A2 & ~sqrtOp_PA | _T_32 | _T_33 | _T_34 | _T_35
                | _T_7 | _T_40 | _T_37 & ~sqrtOp_PB | _cyc_C6_sqrt | _T_3, _cyc_A2 | _cyc_A1_div | _T_33 |
                _T_14 | _T_35 | _T_36 | _T_39 | _T_6 | _T_5 | _T_41, _T_61 | _T_34 | _T_40};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:244:25, :294:25, :407:32, :408:29, :415:33, :418:20, :423:33, :426:33, :438:29, :439:26, :440:26, :443:39, :444:39, :447:27, :456:27, :457:27, :458:27, :674:73, :678:73, :682:54, :684:41
  assign io_latchMulAddA_0 = _T_61;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10
  assign io_mulAddA_0 = {1'h0, (_cyc_A1_sqrt ? {_ER1_A1_sqrt, 36'h0} : 53'h0) | (_T_14 | _cyc_A1_div ? {1'h1,
                fractB_51_PA, fractB_other_PA} : 53'h0) | (_T_38 ? {1'h1, fractA_51_PA, fractA_other_PA} :
                53'h0) | {7'h0, _T_62[45:34] | _T_63[45:34] | _T_64[45:34], _T_1[45:12] | _T_62[33:0] |
                _T_63[33:0] | _T_64[33:0]} | (_cyc_C1_div ? {1'h1, fractB_51_PC, fractB_other_PC} : 53'h0)}
                | _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :247:25, :252:25, :260:25, :263:25, :337:25, :343:25, :426:33, :576:36, :650:{12,51}, :651:{12,25}, :652:12, :653:{19,67}, :655:67, :656:67, :657:67, :658:{12,67}, :702:22, :716:12
  assign io_latchMulAddB_0 = _T_60 | _T_9 | _T_36 | _cyc_C6_sqrt | _T_4 | _T_42;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, DivSqrtRecF64_mulAddZ31.scala:442:39, :457:27, :662:35
  assign io_mulAddB_0 = {1'h0, (_T_9 ? {ER1_B_sqrt, 36'h0} : 53'h0) | {1'h0, _T_65[51:46], _T_66[45:33],
                _T_66[32:30] | _T_67[32:30], _T_66[29:0] | (_cyc_C6_sqrt ? sqrSigma1_C[30:1] : 30'h0) |
                _T_67[29:0]}} | _T_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, DivSqrtRecF64_mulAddZ31.scala:78:30, :442:39, :641:20, :650:{12,51}, :666:{12,36,55}, :667:55, :668:{12,37,55}, :669:55, :710:11
  assign io_mulAddC_2 = (_T_37 ? {sigX1_B, 47'h0} : 105'h0) | (_cyc_C4_sqrt | _T_41 ? {sigXN_C, 47'h0} : 105'h0) |
                {1'h0, _T_68[103:56], _T_68[55:54] | (_T_43 & sqrtOp_PC ? (_exp_PC_0 ? {_fractB_other_PC_0,
                1'h0} : {fractB_other_PC[1] ^ _fractB_other_PC_0, _fractB_other_PC_0}) ^ {~extraT_E, 1'h0}
                : 2'h0), _T_68[53:0] | (_T_43 & ~sqrtOp_PC & ~E_E_div ? {fractA_0_PC, 53'h0} : 54'h0)};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:2978:10, Cat.scala:30:58, DivSqrtRecF64_mulAddZ31.scala:78:30, :332:25, :342:25, :343:25, :383:39, :390:16, :489:30, :650:12, :656:43, :688:{12,45}, :690:{12,25,45,64}, :691:{12,24,27,49,64}, :692:12, :693:17, :695:{29,33}, :696:{16,22}
endmodule

module Mul54(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4423:10
  input          clock, io_val_s0, io_latch_a_s0,
  input  [53:0]  io_a_s0,
  input          io_latch_b_s0,
  input  [53:0]  io_b_s0,
  input  [104:0] io_c_s2,
  output [104:0] io_result_s3);

  reg         val_s1;	// DivSqrtRecF64.scala:96:21
  reg         val_s2;	// DivSqrtRecF64.scala:97:21
  reg [53:0]  reg_a_s1;	// DivSqrtRecF64.scala:98:23
  reg [53:0]  reg_b_s1;	// DivSqrtRecF64.scala:99:23
  reg [53:0]  reg_a_s2;	// DivSqrtRecF64.scala:100:23
  reg [53:0]  reg_b_s2;	// DivSqrtRecF64.scala:101:23
  reg [104:0] reg_result_s3;	// DivSqrtRecF64.scala:102:28

  `ifndef SYNTHESIS	// DivSqrtRecF64.scala:96:21
    `ifdef RANDOMIZE_REG_INIT	// DivSqrtRecF64.scala:96:21
      reg [31:0] _RANDOM;	// DivSqrtRecF64.scala:96:21
      reg [31:0] _RANDOM_0;	// DivSqrtRecF64.scala:98:23
      reg [31:0] _RANDOM_1;	// DivSqrtRecF64.scala:99:23
      reg [31:0] _RANDOM_2;	// DivSqrtRecF64.scala:99:23
      reg [31:0] _RANDOM_3;	// DivSqrtRecF64.scala:100:23
      reg [31:0] _RANDOM_4;	// DivSqrtRecF64.scala:100:23
      reg [31:0] _RANDOM_5;	// DivSqrtRecF64.scala:101:23
      reg [31:0] _RANDOM_6;	// DivSqrtRecF64.scala:102:28
      reg [31:0] _RANDOM_7;	// DivSqrtRecF64.scala:102:28
      reg [31:0] _RANDOM_8;	// DivSqrtRecF64.scala:102:28
      reg [31:0] _RANDOM_9;	// DivSqrtRecF64.scala:102:28

    `endif
    initial begin	// DivSqrtRecF64.scala:96:21
      `INIT_RANDOM_PROLOG_	// DivSqrtRecF64.scala:96:21
      `ifdef RANDOMIZE_REG_INIT	// DivSqrtRecF64.scala:96:21
        _RANDOM = `RANDOM;	// DivSqrtRecF64.scala:96:21
        val_s1 = _RANDOM[0];	// DivSqrtRecF64.scala:96:21
        val_s2 = _RANDOM[1];	// DivSqrtRecF64.scala:97:21
        _RANDOM_0 = `RANDOM;	// DivSqrtRecF64.scala:98:23
        reg_a_s1 = {_RANDOM_0[23:0], _RANDOM[31:2]};	// DivSqrtRecF64.scala:98:23
        _RANDOM_1 = `RANDOM;	// DivSqrtRecF64.scala:99:23
        _RANDOM_2 = `RANDOM;	// DivSqrtRecF64.scala:99:23
        reg_b_s1 = {_RANDOM_2[13:0], _RANDOM_1, _RANDOM_0[31:24]};	// DivSqrtRecF64.scala:99:23
        _RANDOM_3 = `RANDOM;	// DivSqrtRecF64.scala:100:23
        _RANDOM_4 = `RANDOM;	// DivSqrtRecF64.scala:100:23
        reg_a_s2 = {_RANDOM_4[3:0], _RANDOM_3, _RANDOM_2[31:14]};	// DivSqrtRecF64.scala:100:23
        _RANDOM_5 = `RANDOM;	// DivSqrtRecF64.scala:101:23
        reg_b_s2 = {_RANDOM_5[25:0], _RANDOM_4[31:4]};	// DivSqrtRecF64.scala:101:23
        _RANDOM_6 = `RANDOM;	// DivSqrtRecF64.scala:102:28
        _RANDOM_7 = `RANDOM;	// DivSqrtRecF64.scala:102:28
        _RANDOM_8 = `RANDOM;	// DivSqrtRecF64.scala:102:28
        _RANDOM_9 = `RANDOM;	// DivSqrtRecF64.scala:102:28
        reg_result_s3 = {_RANDOM_9[2:0], _RANDOM_8, _RANDOM_7, _RANDOM_6, _RANDOM_5[31:26]};	// DivSqrtRecF64.scala:102:28
      `endif
    end // initial
  `endif
  always @(posedge clock) begin	// DivSqrtRecF64.scala:104:12
    val_s1 <= io_val_s0;	// DivSqrtRecF64.scala:104:12
    val_s2 <= val_s1;	// DivSqrtRecF64.scala:105:12
    if (io_val_s0 & io_latch_a_s0)	// DivSqrtRecF64.scala:109:22
      reg_a_s1 <= io_a_s0;	// DivSqrtRecF64.scala:109:22
    if (io_val_s0 & io_latch_b_s0)	// DivSqrtRecF64.scala:112:22
      reg_b_s1 <= io_b_s0;	// DivSqrtRecF64.scala:112:22
    if (val_s1) begin	// DivSqrtRecF64.scala:105:12, :118:18
      reg_a_s2 <= reg_a_s1;	// DivSqrtRecF64.scala:109:22, :117:18
      reg_b_s2 <= reg_b_s1;	// DivSqrtRecF64.scala:112:22, :118:18
    end
    if (val_s2)	// DivSqrtRecF64.scala:122:23
      reg_result_s3 <= {51'h0, reg_a_s2} * {51'h0, reg_b_s2} + io_c_s2;	// DivSqrtRecF64.scala:117:18, :118:18, :122:{23,36,55}
  end // always @(posedge)
  assign io_result_s3 = reg_result_s3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4423:10, DivSqrtRecF64.scala:122:23
endmodule

module RoundRawFNToRecFN_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4460:10
  input         io_invalidExc, io_in_sign, io_in_isNaN, io_in_isInf, io_in_isZero,
  input  [9:0]  io_in_sExp,
  input  [26:0] io_in_sig,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire _T = io_roundingMode == 2'h0;	// RoundRawFNToRecFN.scala:88:54
  wire _roundMagUp = io_roundingMode == 2'h2 & io_in_sign | &io_roundingMode & ~io_in_sign;	// RoundRawFNToRecFN.scala:90:54, :91:54, :94:{27,42,63,66}
  wire _doShiftSigDown1 = io_in_sig[26];	// RoundRawFNToRecFN.scala:98:36
  wire _isNegExp = $signed(io_in_sExp) < 10'sh0;	// RoundRawFNToRecFN.scala:99:32
  wire [8:0] _T_0 = ~(io_in_sExp[8:0]);	// RoundRawFNToRecFN.scala:103:31, primitives.scala:50:21
  wire _T_1 = _T_0[6];	// primitives.scala:56:25
  wire [64:0] _T_2 = $signed(65'sh10000000000000000 >>> _T_0[5:0]);	// primitives.scala:57:26, :68:52
  wire [24:0] _T_3 = {25{_isNegExp}} | (_T_0[8] ? (_T_0[7] ? {~(_T_1 ? 22'h0 : ~{_T_2[42], _T_2[43], _T_2[44],
                _T_2[45], _T_2[46], _T_2[47], _T_2[48], _T_2[49], _T_2[50], _T_2[51], _T_2[52], _T_2[53],
                _T_2[54], _T_2[55], _T_2[56], _T_2[57], _T_2[58], _T_2[59], _T_2[60], _T_2[61], _T_2[62],
                _T_2[63]}), 3'h7} : {22'h0, _T_1 ? {_T_2[0], _T_2[1], _T_2[2]} : 3'h0}) : 25'h0) | {24'h0,
                _doShiftSigDown1};	// Bitwise.scala:71:12, :102:{21,46}, :108:{18,44}, Cat.scala:30:58, RoundRawFNToRecFN.scala:106:19, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
  wire [25:0] _T_4 = io_in_sig[26:1] & {~_isNegExp, ~_T_3} & {_T_3, 1'h1};	// RoundRawFNToRecFN.scala:110:24, :111:34, :118:43
  wire [26:0] _T_5 = io_in_sig & {_isNegExp, _T_3, 1'h1};	// RoundRawFNToRecFN.scala:109:52, :112:36, :118:43
  wire _anyRound = |_T_4 | |_T_5;	// RoundRawFNToRecFN.scala:111:50, :112:56, :113:32
  wire _T_6 = _T & |_T_4;	// RoundRawFNToRecFN.scala:111:50, :116:40
  wire [25:0] _roundedSig = _T_6 | _roundMagUp & _anyRound ? {1'h0, io_in_sig[26:2] | _T_3} + 26'h1 & ~(_T_6 & ~(|_T_5)
                ? {_T_3, 1'h1} : 26'h0) : {1'h0, io_in_sig[26:2] & ~_T_3};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4460:10, RoundRawFNToRecFN.scala:112:56, :116:{12,56}, :117:29, :118:{26,43,55}, :119:{17,21,63}, :120:26, :121:31, :124:{24,26}
  wire [10:0] _sRoundedExp = {io_in_sExp[9], io_in_sExp} + {9'h0, _roundedSig[25:24]};	// RoundRawFNToRecFN.scala:127:{34,48}, :163:18
  wire _common_totalUnderflow = $signed(_sRoundedExp) < 11'sh6B;	// RoundRawFNToRecFN.scala:138:46
  wire _isNaNOut = io_invalidExc | io_in_isNaN;	// RoundRawFNToRecFN.scala:147:34
  wire _T_7 = ~_isNaNOut & ~io_in_isInf & ~io_in_isZero;	// RoundRawFNToRecFN.scala:149:{22,36,61,64}
  wire _overflow = _T_7 & $signed(_sRoundedExp[10:7]) > 4'sh2;	// RoundRawFNToRecFN.scala:136:{39,56}, :150:32
  wire _overflow_roundMagUp = _T | _roundMagUp;	// RoundRawFNToRecFN.scala:154:57
  wire _T_8 = _T_7 & _common_totalUnderflow & _roundMagUp;	// RoundRawFNToRecFN.scala:155:67
  wire _T_9 = _T_7 & _overflow & ~_overflow_roundMagUp;	// RoundRawFNToRecFN.scala:156:{53,56}
  wire _notNaN_isInfOut = io_in_isInf | _overflow & _overflow_roundMagUp;	// RoundRawFNToRecFN.scala:158:{32,45}
  assign io_out = {~_isNaNOut & io_in_sign, _sRoundedExp[8:0] & ~(io_in_isZero | _common_totalUnderflow ?
                9'h1C0 : 9'h0) & ~(_T_8 ? 9'h194 : 9'h0) & {1'h1, ~_T_9, 7'h7F} & {2'h3, ~_notNaN_isInfOut,
                6'h3F} | (_T_8 ? 9'h6B : 9'h0) | (_T_9 ? 9'h17F : 9'h0) | (_notNaN_isInfOut ? 9'h180 :
                9'h0) | (_isNaNOut ? 9'h1C0 : 9'h0), (_common_totalUnderflow | _isNaNOut ? {_isNaNOut,
                22'h0} : _doShiftSigDown1 ? _roundedSig[23:1] : _roundedSig[22:0]) | {23{_T_9}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4460:10, Bitwise.scala:71:12, Cat.scala:30:58, RoundRawFNToRecFN.scala:91:54, :118:43, :129:36, :131:12, :132:23, :133:23, :160:22, :163:{14,18,32}, :167:{14,18}, :168:19, :171:{14,18}, :174:17, :175:{14,18}, :179:16, :183:16, :187:{16,71}, :188:16, :190:{12,35}, :191:16, :193:11, primitives.scala:65:21
  assign io_exceptionFlags = {io_invalidExc, 1'h0, _overflow, _T_7 & _anyRound & $signed(io_in_sExp) < $signed({1'h0,
                _doShiftSigDown1 ? 9'h81 : 9'h82}), _overflow | _T_7 & _anyRound};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4460:10, Cat.scala:30:58, RoundRawFNToRecFN.scala:141:25, :142:21, :151:32, :152:{28,43}
endmodule

module MulAddRecFN_preMul(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  input  [1:0]  io_op,
  input  [32:0] io_a, io_b, io_c,
  input  [1:0]  io_roundingMode,
  output [23:0] io_mulAddA, io_mulAddB,
  output [47:0] io_mulAddC,
  output [2:0]  io_toPostMul_highExpA,
  output        io_toPostMul_isNaN_isQuietNaNA,
  output [2:0]  io_toPostMul_highExpB,
  output        io_toPostMul_isNaN_isQuietNaNB, io_toPostMul_signProd,
  output        io_toPostMul_isZeroProd, io_toPostMul_opSignC,
  output [2:0]  io_toPostMul_highExpC,
  output        io_toPostMul_isNaN_isQuietNaNC, io_toPostMul_isCDominant,
  output        io_toPostMul_CAlignDist_0,
  output [6:0]  io_toPostMul_CAlignDist,
  output        io_toPostMul_bit0AlignedNegSigC,
  output [25:0] io_toPostMul_highAlignedNegSigC,
  output [10:0] io_toPostMul_sExpSum,
  output [1:0]  io_toPostMul_roundingMode);

  wire [2:0] _io_a_31to29 = io_a[31:29];	// MulAddRecFN.scala:105:24
  wire [2:0] _io_b_31to29 = io_b[31:29];	// MulAddRecFN.scala:111:24
  wire _opSignC = io_c[32] ^ io_op[0];	// MulAddRecFN.scala:114:{23,45,52}
  wire [8:0] _expC = io_c[31:23];	// MulAddRecFN.scala:115:22
  wire [2:0] _io_c_31to29 = io_c[31:29];	// MulAddRecFN.scala:117:24
  wire [23:0] _sigC = {|_io_c_31to29, io_c[22:0]};	// Cat.scala:30:58, MulAddRecFN.scala:116:22, :117:49
  wire _T = io_a[32] ^ io_b[32] ^ io_op[1];	// MulAddRecFN.scala:102:22, :108:22, :122:{34,41}
  wire _isZeroProd = ~(|_io_a_31to29) | ~(|_io_b_31to29);	// MulAddRecFN.scala:105:49, :111:49, :123:30
  wire [10:0] _T_0 = {2'h0, io_a[31:23]} + {{3{~(io_b[31])}}, io_b[30:23]} + 11'h1B;	// Bitwise.scala:71:12, MulAddRecFN.scala:103:22, :125:{14,28,34,51,70}, :148:22
  wire _doSubMags = _T ^ _opSignC;	// MulAddRecFN.scala:130:30
  wire [10:0] _T_1 = _T_0 - {2'h0, _expC};	// MulAddRecFN.scala:132:42, :148:22
  wire _CAlignDist_floor = _isZeroProd | _T_1[10];	// MulAddRecFN.scala:133:{39,56}
  wire [9:0] _T_2 = _T_1[9:0];	// MulAddRecFN.scala:135:44
  wire [6:0] _CAlignDist = _CAlignDist_floor ? 7'h0 : _T_2 < 10'h4A ? _T_1[6:0] : 7'h4A;	// MulAddRecFN.scala:141:12, :143:{16,49}, :144:31
  wire [64:0] _T_3 = $signed(65'sh10000000000000000 >>> _CAlignDist[5:0]);	// primitives.scala:57:26, :68:52
  wire [74:0] _T_4 = $signed($signed({_doSubMags, {24{_doSubMags}} ^ _sigC, {50{_doSubMags}}}) >>> _CAlignDist);	// Bitwise.scala:71:12, Cat.scala:30:58, MulAddRecFN.scala:151:22, :154:70
  assign io_mulAddA = {|_io_a_31to29, io_a[22:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, Cat.scala:30:58, MulAddRecFN.scala:104:22, :105:49
  assign io_mulAddB = {|_io_b_31to29, io_b[22:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, Cat.scala:30:58, MulAddRecFN.scala:110:22, :111:49
  assign io_mulAddC = _T_4[47:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:161:33
  assign io_toPostMul_highExpA = _io_a_31to29;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_isNaN_isQuietNaNA = io_a[22];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:164:46
  assign io_toPostMul_highExpB = _io_b_31to29;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_isNaN_isQuietNaNB = io_b[22];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:166:46
  assign io_toPostMul_signProd = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_isZeroProd = _isZeroProd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_opSignC = _opSignC;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_highExpC = _io_c_31to29;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_isNaN_isQuietNaNC = io_c[22];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:171:46
  assign io_toPostMul_isCDominant = |_io_c_31to29 & (_CAlignDist_floor | _T_2 < 10'h19);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:117:49, :137:19, :138:31, :139:51
  assign io_toPostMul_CAlignDist_0 = _CAlignDist_floor | _T_2 == 10'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:135:{26,62}
  assign io_toPostMul_CAlignDist = _CAlignDist;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
  assign io_toPostMul_bit0AlignedNegSigC = |(_sigC & (_CAlignDist[6] ? {_T_3[54], _T_3[55], _T_3[56], _T_3[57], _T_3[58], _T_3[59],
                _T_3[60], _T_3[61], _T_3[62], _T_3[63], 14'h3FFF} : {10'h0, _T_3[0], _T_3[1], _T_3[2],
                _T_3[3], _T_3[4], _T_3[5], _T_3[6], _T_3[7], _T_3[8], _T_3[9], _T_3[10], _T_3[11],
                _T_3[12], _T_3[13]})) ^ _doSubMags;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, Bitwise.scala:102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:135:62, :156:{19,33,37}, primitives.scala:56:25, :61:20
  assign io_toPostMul_highAlignedNegSigC = _T_4[73:48];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:177:23
  assign io_toPostMul_sExpSum = _CAlignDist_floor ? {2'h0, _expC} : _T_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10, MulAddRecFN.scala:148:22
  assign io_toPostMul_roundingMode = io_roundingMode;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4663:10
endmodule

module MulAddRecFN_postMul(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4860:10
  input  [2:0]  io_fromPreMul_highExpA,
  input         io_fromPreMul_isNaN_isQuietNaNA,
  input  [2:0]  io_fromPreMul_highExpB,
  input         io_fromPreMul_isNaN_isQuietNaNB, io_fromPreMul_signProd,
  input         io_fromPreMul_isZeroProd, io_fromPreMul_opSignC,
  input  [2:0]  io_fromPreMul_highExpC,
  input         io_fromPreMul_isNaN_isQuietNaNC, io_fromPreMul_isCDominant,
  input         io_fromPreMul_CAlignDist_0,
  input  [6:0]  io_fromPreMul_CAlignDist,
  input         io_fromPreMul_bit0AlignedNegSigC,
  input  [25:0] io_fromPreMul_highAlignedNegSigC,
  input  [10:0] io_fromPreMul_sExpSum,
  input  [1:0]  io_fromPreMul_roundingMode,
  input  [48:0] io_mulAddResult,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire [1:0] _io_fromPreMul_highExpA_2to1 = io_fromPreMul_highExpA[2:1];	// MulAddRecFN.scala:208:45
  wire _io_fromPreMul_highExpA_0 = io_fromPreMul_highExpA[0];	// MulAddRecFN.scala:209:56
  wire _isInfA = &_io_fromPreMul_highExpA_2to1 & ~_io_fromPreMul_highExpA_0;	// MulAddRecFN.scala:208:52, :209:{29,32}
  wire _isNaNA = &_io_fromPreMul_highExpA_2to1 & _io_fromPreMul_highExpA_0;	// MulAddRecFN.scala:208:52, :210:29
  wire [1:0] _io_fromPreMul_highExpB_2to1 = io_fromPreMul_highExpB[2:1];	// MulAddRecFN.scala:214:45
  wire _io_fromPreMul_highExpB_0 = io_fromPreMul_highExpB[0];	// MulAddRecFN.scala:215:56
  wire _isInfB = &_io_fromPreMul_highExpB_2to1 & ~_io_fromPreMul_highExpB_0;	// MulAddRecFN.scala:214:52, :215:{29,32}
  wire _isNaNB = &_io_fromPreMul_highExpB_2to1 & _io_fromPreMul_highExpB_0;	// MulAddRecFN.scala:214:52, :216:29
  wire [1:0] _io_fromPreMul_highExpC_2to1 = io_fromPreMul_highExpC[2:1];	// MulAddRecFN.scala:220:45
  wire _io_fromPreMul_highExpC_0 = io_fromPreMul_highExpC[0];	// MulAddRecFN.scala:221:56
  wire _isInfC = &_io_fromPreMul_highExpC_2to1 & ~_io_fromPreMul_highExpC_0;	// MulAddRecFN.scala:220:52, :221:{29,32}
  wire _isNaNC = &_io_fromPreMul_highExpC_2to1 & _io_fromPreMul_highExpC_0;	// MulAddRecFN.scala:220:52, :222:29
  wire _T = io_fromPreMul_roundingMode == 2'h0;	// MulAddRecFN.scala:226:37
  wire _roundingMode_min = io_fromPreMul_roundingMode == 2'h2;	// MulAddRecFN.scala:228:59
  wire _doSubMags = io_fromPreMul_signProd ^ io_fromPreMul_opSignC;	// MulAddRecFN.scala:232:44
  wire [25:0] _T_0 = io_mulAddResult[48] ? io_fromPreMul_highAlignedNegSigC + 26'h1 :
                io_fromPreMul_highAlignedNegSigC;	// MulAddRecFN.scala:237:{16,32}, :238:50
  wire [47:0] _io_mulAddResult_47to0 = io_mulAddResult[47:0];	// MulAddRecFN.scala:241:28
  wire [48:0] _T_1 = {_T_0[1:0], io_mulAddResult[47:1]} ^ {_T_0[0], _io_mulAddResult_47to0};	// MulAddRecFN.scala:191:32, :248:38
  wire [17:0] _T_2 = _T_1[48:31];	// CircuitMath.scala:35:17
  wire [1:0] _T_3 = _T_1[48:47];	// CircuitMath.scala:35:17
  wire [7:0] _T_4 = _T_1[46:39];	// CircuitMath.scala:35:17
  wire [3:0] _T_5 = _T_1[46:43];	// CircuitMath.scala:35:17
  wire [3:0] _T_6 = _T_1[38:35];	// CircuitMath.scala:35:17
  wire [15:0] _T_7 = _T_1[30:15];	// CircuitMath.scala:35:17
  wire [7:0] _T_8 = _T_1[30:23];	// CircuitMath.scala:35:17
  wire [3:0] _T_9 = _T_1[30:27];	// CircuitMath.scala:35:17
  wire [3:0] _T_10 = _T_1[22:19];	// CircuitMath.scala:35:17
  wire [7:0] _T_11 = _T_1[14:7];	// CircuitMath.scala:35:17
  wire [3:0] _T_12 = _T_1[14:11];	// CircuitMath.scala:35:17
  wire [3:0] _T_13 = _T_1[6:3];	// CircuitMath.scala:35:17
  wire [6:0] _T_14 = 7'h49 - {1'h0, |_T_2, |_T_2 ? {|_T_3, |_T_3 ? {3'h0, _T_1[48]} : {|_T_4, |_T_4 ? {|_T_5,
                |_T_5 ? (_T_1[46] ? 2'h3 : _T_1[45] ? 2'h2 : {1'h0, _T_1[44]}) : _T_1[42] ? 2'h3 : _T_1[41]
                ? 2'h2 : {1'h0, _T_1[40]}} : {|_T_6, |_T_6 ? (_T_1[38] ? 2'h3 : _T_1[37] ? 2'h2 : {1'h0,
                _T_1[36]}) : _T_1[34] ? 2'h3 : _T_1[33] ? 2'h2 : {1'h0, _T_1[32]}}}} : {|_T_7, |_T_7 ?
                {|_T_8, |_T_8 ? {|_T_9, |_T_9 ? (_T_1[30] ? 2'h3 : _T_1[29] ? 2'h2 : {1'h0, _T_1[28]}) :
                _T_1[26] ? 2'h3 : _T_1[25] ? 2'h2 : {1'h0, _T_1[24]}} : {|_T_10, |_T_10 ? (_T_1[22] ? 2'h3
                : _T_1[21] ? 2'h2 : {1'h0, _T_1[20]}) : _T_1[18] ? 2'h3 : _T_1[17] ? 2'h2 : {1'h0,
                _T_1[16]}}} : {|_T_11, |_T_11 ? {|_T_12, |_T_12 ? (_T_1[14] ? 2'h3 : _T_1[13] ? 2'h2 :
                {1'h0, _T_1[12]}) : _T_1[10] ? 2'h3 : _T_1[9] ? 2'h2 : {1'h0, _T_1[8]}} : {|_T_13, |_T_13 ?
                (_T_1[6] ? 2'h3 : _T_1[5] ? 2'h2 : {1'h0, _T_1[4]}) : _T_1[2] ? 2'h3 : _T_1[1] ? 2'h2 :
                {1'h0, _T_1[0]}}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, MulAddRecFN.scala:207:46, :208:52, :228:59, primitives.scala:79:25
  wire [17:0] _T_15 = {io_mulAddResult[16:0], io_fromPreMul_bit0AlignedNegSigC};	// MulAddRecFN.scala:255:19
  wire [74:0] _complSigSum = ~{_T_0, _io_mulAddResult_47to0, io_fromPreMul_bit0AlignedNegSigC};	// Cat.scala:30:58, MulAddRecFN.scala:257:23
  wire [17:0] _complSigSum_17to0 = _complSigSum[17:0];	// MulAddRecFN.scala:262:24
  wire [6:0] _CDom_estNormDist = io_fromPreMul_CAlignDist_0 | _doSubMags ? io_fromPreMul_CAlignDist : {2'h0,
                io_fromPreMul_CAlignDist[4:0] - 5'h1};	// MulAddRecFN.scala:226:37, :266:{12,40}, :268:39
  wire _CDom_estNormDist_4 = _CDom_estNormDist[4];	// MulAddRecFN.scala:271:46
  wire [41:0] _T_16 = (_doSubMags | _CDom_estNormDist_4 ? 42'h0 : {_T_0, io_mulAddResult[47:33],
                |{|(io_mulAddResult[32:17]), |_T_15}}) | (~_doSubMags & _CDom_estNormDist_4 ? {_T_0[9:0],
                io_mulAddResult[47:17], |_T_15} : 42'h0) | (_doSubMags & ~_CDom_estNormDist_4 ?
                {_complSigSum[74:34], |{|(_complSigSum[33:18]), |_complSigSum_17to0}} : 42'h0) |
                (_doSubMags & _CDom_estNormDist_4 ? {_complSigSum[58:18], |_complSigSum_17to0} : 42'h0);	// Cat.scala:30:58, MulAddRecFN.scala:252:19, :254:15, :255:57, :259:24, :261:15, :262:62, :271:{12,13,28}, :272:23, :273:35, :277:{12,25}, :278:23, :286:{12,23}, :287:28, :288:40, :291:11, :292:{12,23}, :293:28
  wire _T_17 = _T_14[5];	// MulAddRecFN.scala:338:28
  wire _T_18 = _T_14[4];	// MulAddRecFN.scala:339:33
  wire _T_19 = _T_0[2];	// MulAddRecFN.scala:392:36
  wire [6:0] _estNormDist = io_fromPreMul_isCDominant ? _CDom_estNormDist : _T_14;	// MulAddRecFN.scala:399:12
  wire [42:0] _cFirstNormAbsSigSum = _T_19 ? (io_fromPreMul_isCDominant ? {1'h0, _T_16} : _T_17 ? (_T_18 ? {_complSigSum[27:1],
                16'h0} : {1'h0, _complSigSum[42:1]}) : _T_18 ? {10'h0, _complSigSum[49:18],
                |_complSigSum_17to0} : {_complSigSum[11:1], 32'h0}) : {1'h0, io_fromPreMul_isCDominant ?
                _T_16 : _T_17 ? (_T_18 ? {io_mulAddResult[25:0], {16{_doSubMags}}} : io_mulAddResult[41:0])
                : _T_18 ? {8'h0, _T_0[1:0], io_mulAddResult[47:17], _doSubMags ? ~(|_complSigSum_17to0) :
                |_T_15} : {io_mulAddResult[9:0], {32{_doSubMags}}}};	// Bitwise.scala:71:12, Cat.scala:30:58, CircuitMath.scala:37:22, MulAddRecFN.scala:207:46, :255:57, :262:62, :308:23, :309:20, :310:21, :314:24, :338:12, :339:17, :340:28, :345:17, :347:28, :360:28, :363:29, :379:12, :380:17, :381:{29,64}, :385:17, :387:{29,64}, :407:12, :408:16, :412:16
  wire _T_20 = ~io_fromPreMul_isCDominant & ~_T_19 & _doSubMags;	// MulAddRecFN.scala:418:{9,40,61}
  wire [3:0] _estNormDist_5 = _estNormDist[3:0];	// MulAddRecFN.scala:419:36
  wire [16:0] _T_21 = $signed(17'sh10000 >>> ~_estNormDist_5);	// MulAddRecFN.scala:420:28, primitives.scala:68:52
  wire [15:0] _T_22 = {_T_21[1], _T_21[2], _T_21[3], _T_21[4], _T_21[5], _T_21[6], _T_21[7], _T_21[8], _T_21[9],
                _T_21[10], _T_21[11], _T_21[12], _T_21[13], _T_21[14], _T_21[15], 1'h1};	// Bitwise.scala:102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:231:35
  wire [41:0] _T_23 = _cFirstNormAbsSigSum[42:1] >> ~_estNormDist_5;	// MulAddRecFN.scala:420:28, :424:{32,65}
  wire [15:0] _cFirstNormAbsSigSum_15to0 = _cFirstNormAbsSigSum[15:0];	// MulAddRecFN.scala:427:39
  wire _T_24 = _T_20 ? (~_cFirstNormAbsSigSum_15to0 & _T_22) == 16'h0 : |(_cFirstNormAbsSigSum_15to0 &
                _T_22);	// Bitwise.scala:71:12, MulAddRecFN.scala:426:16, :427:{19,62}, :428:43, :430:61, :431:43
  wire _T_25 = _T_23[26:25] == 2'h0;	// MulAddRecFN.scala:226:37, :436:{29,58}
  wire [10:0] _T_26 = io_fromPreMul_sExpSum - {4'h0, _estNormDist};	// MulAddRecFN.scala:437:40, primitives.scala:59:20
  wire [2:0] _T_27 = _T_23[26:24];	// MulAddRecFN.scala:439:25
  wire _T_28 = |_T_27 ? io_fromPreMul_signProd ^ (io_fromPreMul_isCDominant ? _doSubMags &
                |io_fromPreMul_highExpC : _T_19) : _roundingMode_min;	// MulAddRecFN.scala:219:46, :394:12, :395:23, :439:54, :442:12, :444:36
  wire [9:0] _T_29 = _T_26[9:0];	// MulAddRecFN.scala:446:27
  wire _T_30 = _T_26[10];	// MulAddRecFN.scala:448:34
  wire [9:0] _T_31 = ~_T_29;	// primitives.scala:50:21
  wire _T_32 = _T_31[6];	// primitives.scala:56:25
  wire [64:0] _T_33 = $signed(65'sh10000000000000000 >>> _T_31[5:0]);	// primitives.scala:57:26, :68:52
  wire [26:0] _roundMask = {27{_T_30}} | {(_T_31[9] & _T_31[8] ? (_T_31[7] ? {~(_T_32 ? 21'h0 : ~{_T_33[43],
                _T_33[44], _T_33[45], _T_33[46], _T_33[47], _T_33[48], _T_33[49], _T_33[50], _T_33[51],
                _T_33[52], _T_33[53], _T_33[54], _T_33[55], _T_33[56], _T_33[57], _T_33[58], _T_33[59],
                _T_33[60], _T_33[61], _T_33[62], _T_33[63]}), 4'hF} : {21'h0, _T_32 ? {_T_33[0], _T_33[1],
                _T_33[2], _T_33[3]} : 4'h0}) : 25'h0) | {24'h0, _T_23[25]}, 2'h3};	// Bitwise.scala:71:12, :101:47, :102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:208:52, :448:50, :449:75, :450:26, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
  wire [25:0] _roundMask_26to1 = _roundMask[26:1];	// MulAddRecFN.scala:454:35
  wire [25:0] _T_34 = {_T_23[24:0], _T_24} & ~_roundMask_26to1 & _roundMask[25:0];	// MulAddRecFN.scala:454:{24,40}, :455:30
  wire [25:0] _T_35 = {_T_23[24:0], _T_24} & _roundMask_26to1;	// MulAddRecFN.scala:456:34
  wire _T_36 = (~{_T_23[24:0], _T_24} & _roundMask_26to1) == 26'h0;	// MulAddRecFN.scala:457:{27,34,50}, :477:12
  wire _anyRound = |_T_34 | |_T_35;	// MulAddRecFN.scala:455:46, :456:50, :458:32
  wire _allRound = |_T_34 & _T_36;	// MulAddRecFN.scala:455:46, :459:32
  wire _roundDirectUp = _T_28 ? _roundingMode_min : &io_fromPreMul_roundingMode;	// MulAddRecFN.scala:229:59, :460:28
  wire _T_37 = ~_T_20 & _T & |_T_34 & |_T_35 | ~_T_20 & _roundDirectUp & _anyRound | _T_20 & _allRound |
                _T_20 & _T & |_T_34 | _T_20 & _roundDirectUp;	// MulAddRecFN.scala:455:46, :456:50, :462:10, :463:60, :464:49, :465:49, :466:{49,65}, :467:20
  wire _roundEven = _T_20 ? _T & ~(|_T_34) & _T_36 : _T & |_T_34 & ~(|_T_35);	// MulAddRecFN.scala:455:46, :456:50, :469:12, :470:{42,56}, :471:{56,59}
  wire _inexactY = _T_20 ? ~_allRound : _anyRound;	// MulAddRecFN.scala:473:{27,39}
  wire [25:0] _T_38 = {_T_23[26], _T_23[25:1] | _roundMask[26:2]} + 26'h1;	// MulAddRecFN.scala:238:50, :475:{18,35}
  wire [25:0] _T_39 = (_T_37 | _roundEven ? 26'h0 : _T_23[26:1] & {1'h0, ~(_roundMask[26:2])}) | (_T_37 ? _T_38 :
                26'h0) | (_roundEven ? _T_38 & ~_roundMask_26to1 : 26'h0);	// MulAddRecFN.scala:207:46, :454:24, :477:{12,46,48}, :478:{12,79}, :479:{12,51}
  wire [9:0] _T_40 = (_T_39[25] ? _T_26[9:0] + 10'h1 : 10'h0) | (_T_39[24] ? _T_26[9:0] : 10'h0) | (_T_39[25:24]
                == 2'h0 ? _T_26[9:0] - 10'h1 : 10'h0);	// MulAddRecFN.scala:226:37, :385:17, :482:{12,18,41}, :483:{12,18,61}, :484:{12,19,44}, :485:20
  wire [8:0] _T_41 = _T_40[8:0];	// MulAddRecFN.scala:488:21
  wire _totalUnderflowY = |_T_27 & (_T_40[9] | _T_41 < 9'h6B);	// MulAddRecFN.scala:439:54, :495:19, :496:{19,34,57}, :557:16
  wire _roundMagUp = _roundingMode_min & _T_28 | &io_fromPreMul_roundingMode & ~_T_28;	// MulAddRecFN.scala:229:59, :506:{27,37,58,61}
  wire _overflowY_roundMagUp = _T | _roundMagUp;	// MulAddRecFN.scala:507:58
  wire _mulSpecial = &_io_fromPreMul_highExpA_2to1 | &_io_fromPreMul_highExpB_2to1;	// MulAddRecFN.scala:208:52, :214:52, :511:33
  wire _notSpecial_addZeros = io_fromPreMul_isZeroProd & ~(|io_fromPreMul_highExpC);	// MulAddRecFN.scala:219:46, :513:56
  wire _commonCase = ~(_mulSpecial | &_io_fromPreMul_highExpC_2to1) & ~_notSpecial_addZeros;	// MulAddRecFN.scala:220:52, :512:33, :514:{22,35,38}
  wire _T_42 = _isInfA | _isInfB;	// MulAddRecFN.scala:518:46
  wire _T_43 = _isInfA & io_fromPreMul_highExpB == 3'h0 | io_fromPreMul_highExpA == 3'h0 & _isInfB |
                ~_isNaNA & ~_isNaNB & _T_42 & _isInfC & _doSubMags;	// MulAddRecFN.scala:207:46, :213:46, :517:{17,41,52}, :518:{14,26,67}
  wire _overflow = _commonCase & _T_40[9:7] == 3'h3;	// MulAddRecFN.scala:492:{27,56}, :520:32
  wire _T_44 = _commonCase & _totalUnderflowY & _roundMagUp;	// MulAddRecFN.scala:526:60
  wire _pegMaxFiniteMagOut = _overflow & ~_overflowY_roundMagUp;	// MulAddRecFN.scala:527:{39,42}
  wire _T_45 = _T_42 | _isInfC | _overflow & _overflowY_roundMagUp;	// MulAddRecFN.scala:529:{36,49}
  wire _T_46 = _isNaNA | _isNaNB | _isNaNC | _T_43;	// MulAddRecFN.scala:530:47
  assign io_out = {~_T_46 & (~_doSubMags & io_fromPreMul_opSignC | _mulSpecial &
                ~(&_io_fromPreMul_highExpC_2to1) & io_fromPreMul_signProd | ~_mulSpecial &
                &_io_fromPreMul_highExpC_2to1 & io_fromPreMul_opSignC | ~_mulSpecial & _notSpecial_addZeros
                & _doSubMags & _roundingMode_min) | _commonCase & _T_28, _T_41 & ~(_notSpecial_addZeros |
                ~(|_T_27) | _totalUnderflowY ? 9'h1C0 : 9'h0) & ~(_T_44 ? 9'h194 : 9'h0) & {1'h1,
                ~_pegMaxFiniteMagOut, 7'h7F} & {2'h3, ~_T_45, 6'h3F} | (_T_44 ? 9'h6B : 9'h0) |
                (_pegMaxFiniteMagOut ? 9'h17F : 9'h0) | (_T_45 ? 9'h180 : 9'h0) | (_T_46 ? 9'h1C0 : 9'h0),
                (_totalUnderflowY & _roundMagUp | _T_46 ? {_T_46, 22'h0} : _T_25 ? _T_39[22:0] :
                _T_39[23:1]) | {23{_pegMaxFiniteMagOut}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4860:10, Bitwise.scala:71:12, Cat.scala:30:58, MulAddRecFN.scala:208:52, :220:52, :231:35, :271:13, :439:54, :490:{12,31,55}, :525:40, :533:51, :534:{24,51}, :535:{10,51,78}, :536:59, :538:{20,31,55,70}, :541:{14,18}, :545:{14,18}, :546:19, :549:{14,18}, :552:17, :553:{14,18}, :557:16, :558:16, :562:16, :565:15, :566:16, :568:{12,30,45}, :569:16, :571:11
  assign io_exceptionFlags = {_isNaNA & ~io_fromPreMul_isNaN_isQuietNaNA | _isNaNB & ~io_fromPreMul_isNaN_isQuietNaNB |
                _isNaNC & ~io_fromPreMul_isNaN_isQuietNaNC | _T_43, 1'h0, _overflow, _commonCase &
                _inexactY & (_T_30 | _T_29 <= {2'h0, _T_25 ? 8'h82 : 8'h81}), _overflow | _commonCase &
                _inexactY};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:4860:10, Cat.scala:30:58, MulAddRecFN.scala:207:46, :211:{28,31}, :217:{28,31}, :223:{28,31}, :226:37, :499:35, :500:29, :501:26, :519:55, :521:32, :522:{28,43}
endmodule

module RoundRawFNToRecFN(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5495:10
  input         io_invalidExc, io_in_sign, io_in_isNaN, io_in_isInf, io_in_isZero,
  input  [9:0]  io_in_sExp,
  input  [26:0] io_in_sig,
  input  [1:0]  io_roundingMode,
  output [32:0] io_out,
  output [4:0]  io_exceptionFlags);

  wire _T = io_roundingMode == 2'h0;	// RoundRawFNToRecFN.scala:88:54
  wire _roundMagUp = io_roundingMode == 2'h2 & io_in_sign | &io_roundingMode & ~io_in_sign;	// RoundRawFNToRecFN.scala:90:54, :91:54, :94:{27,42,63,66}
  wire _doShiftSigDown1 = io_in_sig[26];	// RoundRawFNToRecFN.scala:98:36
  wire _isNegExp = $signed(io_in_sExp) < 10'sh0;	// RoundRawFNToRecFN.scala:99:32
  wire [8:0] _T_0 = ~(io_in_sExp[8:0]);	// RoundRawFNToRecFN.scala:103:31, primitives.scala:50:21
  wire _T_1 = _T_0[6];	// primitives.scala:56:25
  wire [64:0] _T_2 = $signed(65'sh10000000000000000 >>> _T_0[5:0]);	// primitives.scala:57:26, :68:52
  wire [24:0] _T_3 = {25{_isNegExp}} | (_T_0[8] ? (_T_0[7] ? {~(_T_1 ? 22'h0 : ~{_T_2[42], _T_2[43], _T_2[44],
                _T_2[45], _T_2[46], _T_2[47], _T_2[48], _T_2[49], _T_2[50], _T_2[51], _T_2[52], _T_2[53],
                _T_2[54], _T_2[55], _T_2[56], _T_2[57], _T_2[58], _T_2[59], _T_2[60], _T_2[61], _T_2[62],
                _T_2[63]}), 3'h7} : {22'h0, _T_1 ? {_T_2[0], _T_2[1], _T_2[2]} : 3'h0}) : 25'h0) | {24'h0,
                _doShiftSigDown1};	// Bitwise.scala:71:12, :102:{21,46}, :108:{18,44}, Cat.scala:30:58, RoundRawFNToRecFN.scala:106:19, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
  wire [25:0] _T_4 = io_in_sig[26:1] & {~_isNegExp, ~_T_3} & {_T_3, 1'h1};	// RoundRawFNToRecFN.scala:110:24, :111:34, :118:43
  wire [26:0] _T_5 = io_in_sig & {_isNegExp, _T_3, 1'h1};	// RoundRawFNToRecFN.scala:109:52, :112:36, :118:43
  wire _anyRound = |_T_4 | |_T_5;	// RoundRawFNToRecFN.scala:111:50, :112:56, :113:32
  wire _T_6 = _T & |_T_4;	// RoundRawFNToRecFN.scala:111:50, :116:40
  wire [25:0] _roundedSig = _T_6 | _roundMagUp & _anyRound ? {1'h0, io_in_sig[26:2] | _T_3} + 26'h1 & ~(_T_6 & ~(|_T_5)
                ? {_T_3, 1'h1} : 26'h0) : {1'h0, io_in_sig[26:2] & ~_T_3};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5495:10, RoundRawFNToRecFN.scala:112:56, :116:{12,56}, :117:29, :118:{26,43,55}, :119:{17,21,63}, :120:26, :121:31, :124:{24,26}
  wire [10:0] _sRoundedExp = {io_in_sExp[9], io_in_sExp} + {9'h0, _roundedSig[25:24]};	// RoundRawFNToRecFN.scala:127:{34,48}, :163:18
  wire _common_totalUnderflow = $signed(_sRoundedExp) < 11'sh6B;	// RoundRawFNToRecFN.scala:138:46
  wire _isNaNOut = io_invalidExc | io_in_isNaN;	// RoundRawFNToRecFN.scala:147:34
  wire _T_7 = ~_isNaNOut & ~io_in_isInf & ~io_in_isZero;	// RoundRawFNToRecFN.scala:149:{22,36,61,64}
  wire _overflow = _T_7 & $signed(_sRoundedExp[10:7]) > 4'sh2;	// RoundRawFNToRecFN.scala:136:{39,56}, :150:32
  wire _overflow_roundMagUp = _T | _roundMagUp;	// RoundRawFNToRecFN.scala:154:57
  wire _T_8 = _T_7 & _common_totalUnderflow & _roundMagUp;	// RoundRawFNToRecFN.scala:155:67
  wire _T_9 = _T_7 & _overflow & ~_overflow_roundMagUp;	// RoundRawFNToRecFN.scala:156:{53,56}
  wire _notNaN_isInfOut = io_in_isInf | _overflow & _overflow_roundMagUp;	// RoundRawFNToRecFN.scala:158:{32,45}
  assign io_out = {~_isNaNOut & io_in_sign, _sRoundedExp[8:0] & ~(io_in_isZero | _common_totalUnderflow ?
                9'h1C0 : 9'h0) & ~(_T_8 ? 9'h194 : 9'h0) & {1'h1, ~_T_9, 7'h7F} & {2'h3, ~_notNaN_isInfOut,
                6'h3F} | (_T_8 ? 9'h6B : 9'h0) | (_T_9 ? 9'h17F : 9'h0) | (_notNaN_isInfOut ? 9'h180 :
                9'h0) | (_isNaNOut ? 9'h1C0 : 9'h0), (_common_totalUnderflow | _isNaNOut ? {_isNaNOut,
                22'h0} : _doShiftSigDown1 ? _roundedSig[23:1] : _roundedSig[22:0]) | {23{_T_9}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5495:10, Bitwise.scala:71:12, Cat.scala:30:58, RoundRawFNToRecFN.scala:91:54, :118:43, :129:36, :131:12, :132:23, :133:23, :160:22, :163:{14,18,32}, :167:{14,18}, :168:19, :171:{14,18}, :174:17, :175:{14,18}, :179:16, :183:16, :187:{16,71}, :188:16, :190:{12,35}, :191:16, :193:11, primitives.scala:65:21
  assign io_exceptionFlags = {io_invalidExc, 1'h0, _overflow, _T_7 & _anyRound & $signed(io_in_sExp) < $signed({1'h0,
                _doShiftSigDown1 ? 9'h81 : 9'h82}), _overflow | _T_7 & _anyRound};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5495:10, Cat.scala:30:58, RoundRawFNToRecFN.scala:141:25, :142:21, :151:32, :152:{28,43}
endmodule

module MulAddRecFN_preMul_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  input  [1:0]   io_op,
  input  [64:0]  io_a, io_b, io_c,
  input  [1:0]   io_roundingMode,
  output [52:0]  io_mulAddA, io_mulAddB,
  output [105:0] io_mulAddC,
  output [2:0]   io_toPostMul_highExpA,
  output         io_toPostMul_isNaN_isQuietNaNA,
  output [2:0]   io_toPostMul_highExpB,
  output         io_toPostMul_isNaN_isQuietNaNB, io_toPostMul_signProd,
  output         io_toPostMul_isZeroProd, io_toPostMul_opSignC,
  output [2:0]   io_toPostMul_highExpC,
  output         io_toPostMul_isNaN_isQuietNaNC, io_toPostMul_isCDominant,
  output         io_toPostMul_CAlignDist_0,
  output [7:0]   io_toPostMul_CAlignDist,
  output         io_toPostMul_bit0AlignedNegSigC,
  output [54:0]  io_toPostMul_highAlignedNegSigC,
  output [13:0]  io_toPostMul_sExpSum,
  output [1:0]   io_toPostMul_roundingMode);

  wire [2:0] _io_a_63to61 = io_a[63:61];	// MulAddRecFN.scala:105:24
  wire [2:0] _io_b_63to61 = io_b[63:61];	// MulAddRecFN.scala:111:24
  wire _opSignC = io_c[64] ^ io_op[0];	// MulAddRecFN.scala:114:{23,45,52}
  wire [11:0] _expC = io_c[63:52];	// MulAddRecFN.scala:115:22
  wire [2:0] _io_c_63to61 = io_c[63:61];	// MulAddRecFN.scala:117:24
  wire [52:0] _sigC = {|_io_c_63to61, io_c[51:0]};	// Cat.scala:30:58, MulAddRecFN.scala:116:22, :117:49
  wire _T = io_a[64] ^ io_b[64] ^ io_op[1];	// MulAddRecFN.scala:102:22, :108:22, :122:{34,41}
  wire _isZeroProd = ~(|_io_a_63to61) | ~(|_io_b_63to61);	// MulAddRecFN.scala:105:49, :111:49, :123:30
  wire [13:0] _T_0 = {2'h0, io_a[63:52]} + {{3{~(io_b[63])}}, io_b[62:52]} + 14'h38;	// Bitwise.scala:71:12, MulAddRecFN.scala:103:22, :125:{14,28,34,51,70}, :148:22
  wire _doSubMags = _T ^ _opSignC;	// MulAddRecFN.scala:130:30
  wire [13:0] _T_1 = _T_0 - {2'h0, _expC};	// MulAddRecFN.scala:132:42, :148:22
  wire _CAlignDist_floor = _isZeroProd | _T_1[13];	// MulAddRecFN.scala:133:{39,56}
  wire [12:0] _T_2 = _T_1[12:0];	// MulAddRecFN.scala:135:44
  wire [7:0] _CAlignDist = _CAlignDist_floor ? 8'h0 : _T_2 < 13'hA1 ? _T_1[7:0] : 8'hA1;	// MulAddRecFN.scala:141:12, :143:{16,49}, :144:31
  wire _CAlignDist_6 = _CAlignDist[6];	// primitives.scala:56:25
  wire [64:0] _T_3 = $signed(65'sh10000000000000000 >>> _CAlignDist[5:0]);	// primitives.scala:57:26, :68:52
  wire [161:0] _T_4 = $signed($signed({_doSubMags, {53{_doSubMags}} ^ _sigC, {108{_doSubMags}}}) >>> _CAlignDist);	// Bitwise.scala:71:12, Cat.scala:30:58, MulAddRecFN.scala:151:22, :154:70
  assign io_mulAddA = {|_io_a_63to61, io_a[51:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, Cat.scala:30:58, MulAddRecFN.scala:104:22, :105:49
  assign io_mulAddB = {|_io_b_63to61, io_b[51:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, Cat.scala:30:58, MulAddRecFN.scala:110:22, :111:49
  assign io_mulAddC = _T_4[105:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:161:33
  assign io_toPostMul_highExpA = _io_a_63to61;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_isNaN_isQuietNaNA = io_a[51];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:164:46
  assign io_toPostMul_highExpB = _io_b_63to61;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_isNaN_isQuietNaNB = io_b[51];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:166:46
  assign io_toPostMul_signProd = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_isZeroProd = _isZeroProd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_opSignC = _opSignC;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_highExpC = _io_c_63to61;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_isNaN_isQuietNaNC = io_c[51];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:171:46
  assign io_toPostMul_isCDominant = |_io_c_63to61 & (_CAlignDist_floor | _T_2 < 13'h36);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:117:49, :137:19, :138:31, :139:51
  assign io_toPostMul_CAlignDist_0 = _CAlignDist_floor | _T_2 == 13'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:135:{26,62}
  assign io_toPostMul_CAlignDist = _CAlignDist;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
  assign io_toPostMul_bit0AlignedNegSigC = |(_sigC & (_CAlignDist[7] ? {~(_CAlignDist_6 ? 33'h0 : ~{_T_3[31], _T_3[32], _T_3[33],
                _T_3[34], _T_3[35], _T_3[36], _T_3[37], _T_3[38], _T_3[39], _T_3[40], _T_3[41], _T_3[42],
                _T_3[43], _T_3[44], _T_3[45], _T_3[46], _T_3[47], _T_3[48], _T_3[49], _T_3[50], _T_3[51],
                _T_3[52], _T_3[53], _T_3[54], _T_3[55], _T_3[56], _T_3[57], _T_3[58], _T_3[59], _T_3[60],
                _T_3[61], _T_3[62], _T_3[63]}), 20'hFFFFF} : {33'h0, _CAlignDist_6 ? {_T_3[0], _T_3[1],
                _T_3[2], _T_3[3], _T_3[4], _T_3[5], _T_3[6], _T_3[7], _T_3[8], _T_3[9], _T_3[10], _T_3[11],
                _T_3[12], _T_3[13], _T_3[14], _T_3[15], _T_3[16], _T_3[17], _T_3[18], _T_3[19]} : 20'h0}))
                ^ _doSubMags;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, Bitwise.scala:102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:156:{19,33,37}, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
  assign io_toPostMul_highAlignedNegSigC = _T_4[160:106];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:177:23
  assign io_toPostMul_sExpSum = _CAlignDist_floor ? {2'h0, _expC} : _T_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10, MulAddRecFN.scala:148:22
  assign io_toPostMul_roundingMode = io_roundingMode;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5698:10
endmodule

module MulAddRecFN_postMul_1(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5924:10
  input  [2:0]   io_fromPreMul_highExpA,
  input          io_fromPreMul_isNaN_isQuietNaNA,
  input  [2:0]   io_fromPreMul_highExpB,
  input          io_fromPreMul_isNaN_isQuietNaNB, io_fromPreMul_signProd,
  input          io_fromPreMul_isZeroProd, io_fromPreMul_opSignC,
  input  [2:0]   io_fromPreMul_highExpC,
  input          io_fromPreMul_isNaN_isQuietNaNC, io_fromPreMul_isCDominant,
  input          io_fromPreMul_CAlignDist_0,
  input  [7:0]   io_fromPreMul_CAlignDist,
  input          io_fromPreMul_bit0AlignedNegSigC,
  input  [54:0]  io_fromPreMul_highAlignedNegSigC,
  input  [13:0]  io_fromPreMul_sExpSum,
  input  [1:0]   io_fromPreMul_roundingMode,
  input  [106:0] io_mulAddResult,
  output [64:0]  io_out,
  output [4:0]   io_exceptionFlags);

  wire [1:0] _io_fromPreMul_highExpA_2to1 = io_fromPreMul_highExpA[2:1];	// MulAddRecFN.scala:208:45
  wire _io_fromPreMul_highExpA_0 = io_fromPreMul_highExpA[0];	// MulAddRecFN.scala:209:56
  wire _isInfA = &_io_fromPreMul_highExpA_2to1 & ~_io_fromPreMul_highExpA_0;	// MulAddRecFN.scala:208:52, :209:{29,32}
  wire _isNaNA = &_io_fromPreMul_highExpA_2to1 & _io_fromPreMul_highExpA_0;	// MulAddRecFN.scala:208:52, :210:29
  wire [1:0] _io_fromPreMul_highExpB_2to1 = io_fromPreMul_highExpB[2:1];	// MulAddRecFN.scala:214:45
  wire _io_fromPreMul_highExpB_0 = io_fromPreMul_highExpB[0];	// MulAddRecFN.scala:215:56
  wire _isInfB = &_io_fromPreMul_highExpB_2to1 & ~_io_fromPreMul_highExpB_0;	// MulAddRecFN.scala:214:52, :215:{29,32}
  wire _isNaNB = &_io_fromPreMul_highExpB_2to1 & _io_fromPreMul_highExpB_0;	// MulAddRecFN.scala:214:52, :216:29
  wire [1:0] _io_fromPreMul_highExpC_2to1 = io_fromPreMul_highExpC[2:1];	// MulAddRecFN.scala:220:45
  wire _io_fromPreMul_highExpC_0 = io_fromPreMul_highExpC[0];	// MulAddRecFN.scala:221:56
  wire _isInfC = &_io_fromPreMul_highExpC_2to1 & ~_io_fromPreMul_highExpC_0;	// MulAddRecFN.scala:220:52, :221:{29,32}
  wire _isNaNC = &_io_fromPreMul_highExpC_2to1 & _io_fromPreMul_highExpC_0;	// MulAddRecFN.scala:220:52, :222:29
  wire _T = io_fromPreMul_roundingMode == 2'h0;	// MulAddRecFN.scala:226:37
  wire _roundingMode_min = io_fromPreMul_roundingMode == 2'h2;	// MulAddRecFN.scala:228:59
  wire _doSubMags = io_fromPreMul_signProd ^ io_fromPreMul_opSignC;	// MulAddRecFN.scala:232:44
  wire [54:0] _T_0 = io_mulAddResult[106] ? io_fromPreMul_highAlignedNegSigC + 55'h1 :
                io_fromPreMul_highAlignedNegSigC;	// MulAddRecFN.scala:237:{16,32}, :238:50
  wire [105:0] _io_mulAddResult_105to0 = io_mulAddResult[105:0];	// MulAddRecFN.scala:241:28
  wire [106:0] _T_1 = {_T_0[1:0], io_mulAddResult[105:1]} ^ {_T_0[0], _io_mulAddResult_105to0};	// MulAddRecFN.scala:191:32, :248:38
  wire [43:0] _T_2 = _T_1[106:63];	// CircuitMath.scala:35:17
  wire [11:0] _T_3 = _T_1[106:95];	// CircuitMath.scala:35:17
  wire [3:0] _T_4 = _T_1[106:103];	// CircuitMath.scala:35:17
  wire [3:0] _T_5 = _T_1[102:99];	// CircuitMath.scala:35:17
  wire [15:0] _T_6 = _T_1[94:79];	// CircuitMath.scala:35:17
  wire [7:0] _T_7 = _T_1[94:87];	// CircuitMath.scala:35:17
  wire [3:0] _T_8 = _T_1[94:91];	// CircuitMath.scala:35:17
  wire [3:0] _T_9 = _T_1[86:83];	// CircuitMath.scala:35:17
  wire [7:0] _T_10 = _T_1[78:71];	// CircuitMath.scala:35:17
  wire [3:0] _T_11 = _T_1[78:75];	// CircuitMath.scala:35:17
  wire [3:0] _T_12 = _T_1[70:67];	// CircuitMath.scala:35:17
  wire [31:0] _T_13 = _T_1[62:31];	// CircuitMath.scala:35:17
  wire [15:0] _T_14 = _T_1[62:47];	// CircuitMath.scala:35:17
  wire [7:0] _T_15 = _T_1[62:55];	// CircuitMath.scala:35:17
  wire [3:0] _T_16 = _T_1[62:59];	// CircuitMath.scala:35:17
  wire [3:0] _T_17 = _T_1[54:51];	// CircuitMath.scala:35:17
  wire [7:0] _T_18 = _T_1[46:39];	// CircuitMath.scala:35:17
  wire [3:0] _T_19 = _T_1[46:43];	// CircuitMath.scala:35:17
  wire [3:0] _T_20 = _T_1[38:35];	// CircuitMath.scala:35:17
  wire [15:0] _T_21 = _T_1[30:15];	// CircuitMath.scala:35:17
  wire [7:0] _T_22 = _T_1[30:23];	// CircuitMath.scala:35:17
  wire [3:0] _T_23 = _T_1[30:27];	// CircuitMath.scala:35:17
  wire [3:0] _T_24 = _T_1[22:19];	// CircuitMath.scala:35:17
  wire [7:0] _T_25 = _T_1[14:7];	// CircuitMath.scala:35:17
  wire [3:0] _T_26 = _T_1[14:11];	// CircuitMath.scala:35:17
  wire [3:0] _T_27 = _T_1[6:3];	// CircuitMath.scala:35:17
  wire [7:0] _T_28 = 8'hA0 - {1'h0, |_T_2, |_T_2 ? {|_T_3, |_T_3 ? {1'h0, |_T_4, |_T_4 ? {1'h0, _T_1[106] ? 2'h3
                : _T_1[105] ? 2'h2 : {1'h0, _T_1[104]}} : {|_T_5, |_T_5 ? (_T_1[102] ? 2'h3 : _T_1[101] ?
                2'h2 : {1'h0, _T_1[100]}) : _T_1[98] ? 2'h3 : _T_1[97] ? 2'h2 : {1'h0, _T_1[96]}}} :
                {|_T_6, |_T_6 ? {|_T_7, |_T_7 ? {|_T_8, |_T_8 ? (_T_1[94] ? 2'h3 : _T_1[93] ? 2'h2 : {1'h0,
                _T_1[92]}) : _T_1[90] ? 2'h3 : _T_1[89] ? 2'h2 : {1'h0, _T_1[88]}} : {|_T_9, |_T_9 ?
                (_T_1[86] ? 2'h3 : _T_1[85] ? 2'h2 : {1'h0, _T_1[84]}) : _T_1[82] ? 2'h3 : _T_1[81] ? 2'h2
                : {1'h0, _T_1[80]}}} : {|_T_10, |_T_10 ? {|_T_11, |_T_11 ? (_T_1[78] ? 2'h3 : _T_1[77] ?
                2'h2 : {1'h0, _T_1[76]}) : _T_1[74] ? 2'h3 : _T_1[73] ? 2'h2 : {1'h0, _T_1[72]}} : {|_T_12,
                |_T_12 ? (_T_1[70] ? 2'h3 : _T_1[69] ? 2'h2 : {1'h0, _T_1[68]}) : _T_1[66] ? 2'h3 :
                _T_1[65] ? 2'h2 : {1'h0, _T_1[64]}}}}} : {|_T_13, |_T_13 ? {|_T_14, |_T_14 ? {|_T_15,
                |_T_15 ? {|_T_16, |_T_16 ? (_T_1[62] ? 2'h3 : _T_1[61] ? 2'h2 : {1'h0, _T_1[60]}) :
                _T_1[58] ? 2'h3 : _T_1[57] ? 2'h2 : {1'h0, _T_1[56]}} : {|_T_17, |_T_17 ? (_T_1[54] ? 2'h3
                : _T_1[53] ? 2'h2 : {1'h0, _T_1[52]}) : _T_1[50] ? 2'h3 : _T_1[49] ? 2'h2 : {1'h0,
                _T_1[48]}}} : {|_T_18, |_T_18 ? {|_T_19, |_T_19 ? (_T_1[46] ? 2'h3 : _T_1[45] ? 2'h2 :
                {1'h0, _T_1[44]}) : _T_1[42] ? 2'h3 : _T_1[41] ? 2'h2 : {1'h0, _T_1[40]}} : {|_T_20, |_T_20
                ? (_T_1[38] ? 2'h3 : _T_1[37] ? 2'h2 : {1'h0, _T_1[36]}) : _T_1[34] ? 2'h3 : _T_1[33] ?
                2'h2 : {1'h0, _T_1[32]}}}} : {|_T_21, |_T_21 ? {|_T_22, |_T_22 ? {|_T_23, |_T_23 ?
                (_T_1[30] ? 2'h3 : _T_1[29] ? 2'h2 : {1'h0, _T_1[28]}) : _T_1[26] ? 2'h3 : _T_1[25] ? 2'h2
                : {1'h0, _T_1[24]}} : {|_T_24, |_T_24 ? (_T_1[22] ? 2'h3 : _T_1[21] ? 2'h2 : {1'h0,
                _T_1[20]}) : _T_1[18] ? 2'h3 : _T_1[17] ? 2'h2 : {1'h0, _T_1[16]}}} : {|_T_25, |_T_25 ?
                {|_T_26, |_T_26 ? (_T_1[14] ? 2'h3 : _T_1[13] ? 2'h2 : {1'h0, _T_1[12]}) : _T_1[10] ? 2'h3
                : _T_1[9] ? 2'h2 : {1'h0, _T_1[8]}} : {|_T_27, |_T_27 ? (_T_1[6] ? 2'h3 : _T_1[5] ? 2'h2 :
                {1'h0, _T_1[4]}) : _T_1[2] ? 2'h3 : _T_1[1] ? 2'h2 : {1'h0, _T_1[0]}}}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, MulAddRecFN.scala:207:46, :208:52, :228:59, primitives.scala:79:25
  wire [43:0] _T_29 = {io_mulAddResult[42:0], io_fromPreMul_bit0AlignedNegSigC};	// MulAddRecFN.scala:255:19
  wire [161:0] _complSigSum = ~{_T_0, _io_mulAddResult_105to0, io_fromPreMul_bit0AlignedNegSigC};	// Cat.scala:30:58, MulAddRecFN.scala:257:23
  wire [43:0] _complSigSum_43to0 = _complSigSum[43:0];	// MulAddRecFN.scala:262:24
  wire [7:0] _CDom_estNormDist = io_fromPreMul_CAlignDist_0 | _doSubMags ? io_fromPreMul_CAlignDist : {2'h0,
                io_fromPreMul_CAlignDist[5:0] - 6'h1};	// MulAddRecFN.scala:226:37, :266:{12,40}, :268:39
  wire _CDom_estNormDist_5 = _CDom_estNormDist[5];	// MulAddRecFN.scala:271:46
  wire [86:0] _T_30 = (_doSubMags | _CDom_estNormDist_5 ? 87'h0 : {_T_0, io_mulAddResult[105:75],
                |{|(io_mulAddResult[74:43]), |_T_29}}) | (~_doSubMags & _CDom_estNormDist_5 ? {_T_0[22:0],
                io_mulAddResult[105:43], |_T_29} : 87'h0) | (_doSubMags & ~_CDom_estNormDist_5 ?
                {_complSigSum[161:76], |{|(_complSigSum[75:44]), |_complSigSum_43to0}} : 87'h0) |
                (_doSubMags & _CDom_estNormDist_5 ? {_complSigSum[129:44], |_complSigSum_43to0} : 87'h0);	// Cat.scala:30:58, MulAddRecFN.scala:252:19, :254:15, :255:57, :259:24, :261:15, :262:62, :271:{12,13,28}, :272:23, :273:35, :277:{12,25}, :278:23, :286:{12,23}, :287:28, :288:40, :291:11, :292:{12,23}, :293:28
  wire _T_31 = _T_28[4];	// MulAddRecFN.scala:316:37
  wire [10:0] _complSigSum_11to1 = _complSigSum[11:1];	// MulAddRecFN.scala:329:39
  wire _T_32 = _T_28[6];	// MulAddRecFN.scala:338:28
  wire _T_33 = _T_28[5];	// MulAddRecFN.scala:339:33
  wire _T_34 = _T_0[2];	// MulAddRecFN.scala:392:36
  wire [7:0] _estNormDist = io_fromPreMul_isCDominant ? _CDom_estNormDist : _T_28;	// MulAddRecFN.scala:399:12
  wire [87:0] _cFirstNormAbsSigSum = _T_34 ? (io_fromPreMul_isCDominant ? {1'h0, _T_30} : _T_32 ? (_T_33 ? {_complSigSum[66:1],
                22'h0} : {_complSigSum[98:12], |_complSigSum_11to1}) : _T_33 ? (_T_31 ? {23'h0,
                _complSigSum[107:44], |_complSigSum_43to0} : {_complSigSum[2:1], 86'h0}) :
                {_complSigSum[34:1], 54'h0}) : {1'h0, io_fromPreMul_isCDominant ? _T_30 : _T_32 ? (_T_33 ?
                {io_mulAddResult[64:0], {22{_doSubMags}}} : {io_mulAddResult[96:11], _doSubMags ?
                ~(|_complSigSum_11to1) : |(io_mulAddResult[10:0])}) : _T_33 ? (_T_31 ? {21'h0, _T_0[1:0],
                io_mulAddResult[105:43], _doSubMags ? ~(|_complSigSum_43to0) : |_T_29} :
                {io_mulAddResult[0], {86{_doSubMags}}}) : {io_mulAddResult[32:0], {54{_doSubMags}}}};	// Bitwise.scala:71:12, Cat.scala:30:58, MulAddRecFN.scala:207:46, :255:57, :262:62, :308:23, :309:20, :310:21, :316:21, :318:32, :324:28, :328:26, :329:77, :331:{34,72}, :338:12, :339:17, :340:28, :345:17, :347:28, :360:28, :365:21, :367:{33,68}, :372:33, :379:12, :380:17, :381:{29,64}, :385:17, :387:{29,64}, :407:12, :408:16, :412:16
  wire _T_35 = ~io_fromPreMul_isCDominant & ~_T_34 & _doSubMags;	// MulAddRecFN.scala:418:{9,40,61}
  wire [4:0] _estNormDist_5 = _estNormDist[4:0];	// MulAddRecFN.scala:419:36
  wire [32:0] _T_36 = $signed(33'sh100000000 >>> ~_estNormDist_5);	// MulAddRecFN.scala:420:28, primitives.scala:68:52
  wire [31:0] _T_37 = {_T_36[1], _T_36[2], _T_36[3], _T_36[4], _T_36[5], _T_36[6], _T_36[7], _T_36[8], _T_36[9],
                _T_36[10], _T_36[11], _T_36[12], _T_36[13], _T_36[14], _T_36[15], _T_36[16], _T_36[17],
                _T_36[18], _T_36[19], _T_36[20], _T_36[21], _T_36[22], _T_36[23], _T_36[24], _T_36[25],
                _T_36[26], _T_36[27], _T_36[28], _T_36[29], _T_36[30], _T_36[31], 1'h1};	// Bitwise.scala:102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:231:35
  wire [86:0] _T_38 = _cFirstNormAbsSigSum[87:1] >> ~_estNormDist_5;	// MulAddRecFN.scala:420:28, :424:{32,65}
  wire [31:0] _cFirstNormAbsSigSum_31to0 = _cFirstNormAbsSigSum[31:0];	// MulAddRecFN.scala:427:39
  wire _T_39 = _T_35 ? (~_cFirstNormAbsSigSum_31to0 & _T_37) == 32'h0 : |(_cFirstNormAbsSigSum_31to0 &
                _T_37);	// CircuitMath.scala:37:22, MulAddRecFN.scala:426:16, :427:{19,62}, :428:43, :430:61, :431:43
  wire _T_40 = _T_38[55:54] == 2'h0;	// MulAddRecFN.scala:226:37, :436:{29,58}
  wire [13:0] _T_41 = io_fromPreMul_sExpSum - {6'h0, _estNormDist};	// MulAddRecFN.scala:437:40
  wire [2:0] _T_42 = _T_38[55:53];	// MulAddRecFN.scala:439:25
  wire _T_43 = |_T_42 ? io_fromPreMul_signProd ^ (io_fromPreMul_isCDominant ? _doSubMags &
                |io_fromPreMul_highExpC : _T_34) : _roundingMode_min;	// MulAddRecFN.scala:219:46, :394:12, :395:23, :439:54, :442:12, :444:36
  wire [12:0] _T_44 = _T_41[12:0];	// MulAddRecFN.scala:446:27
  wire _T_45 = _T_41[13];	// MulAddRecFN.scala:448:34
  wire [12:0] _T_46 = ~_T_44;	// primitives.scala:50:21
  wire _T_47 = _T_46[9];	// primitives.scala:56:25
  wire _T_48 = _T_46[8];	// primitives.scala:56:25
  wire _T_49 = _T_46[7];	// primitives.scala:56:25
  wire _T_50 = _T_46[6];	// primitives.scala:56:25
  wire [64:0] _T_51 = $signed(65'sh10000000000000000 >>> _T_46[5:0]);	// primitives.scala:57:26, :68:52
  wire [55:0] _roundMask = {56{_T_45}} | {(_T_46[12] & _T_46[11] ? (_T_46[10] ? {~(_T_47 | _T_48 | _T_49 | _T_50 ?
                50'h0 : ~{_T_51[14], _T_51[15], _T_51[16], _T_51[17], _T_51[18], _T_51[19], _T_51[20],
                _T_51[21], _T_51[22], _T_51[23], _T_51[24], _T_51[25], _T_51[26], _T_51[27], _T_51[28],
                _T_51[29], _T_51[30], _T_51[31], _T_51[32], _T_51[33], _T_51[34], _T_51[35], _T_51[36],
                _T_51[37], _T_51[38], _T_51[39], _T_51[40], _T_51[41], _T_51[42], _T_51[43], _T_51[44],
                _T_51[45], _T_51[46], _T_51[47], _T_51[48], _T_51[49], _T_51[50], _T_51[51], _T_51[52],
                _T_51[53], _T_51[54], _T_51[55], _T_51[56], _T_51[57], _T_51[58], _T_51[59], _T_51[60],
                _T_51[61], _T_51[62], _T_51[63]}), 4'hF} : {50'h0, _T_47 & _T_48 & _T_49 & _T_50 ?
                {_T_51[0], _T_51[1], _T_51[2], _T_51[3]} : 4'h0}) : 54'h0) | {53'h0, _T_38[54]}, 2'h3};	// Bitwise.scala:71:12, :101:47, :102:{21,46}, :108:{18,44}, Cat.scala:30:58, MulAddRecFN.scala:208:52, :448:50, :449:75, :450:26, primitives.scala:56:25, :59:20, :61:20, :65:{17,21,36}
  wire [54:0] _roundMask_55to1 = _roundMask[55:1];	// MulAddRecFN.scala:454:35
  wire [54:0] _T_52 = {_T_38[53:0], _T_39} & ~_roundMask_55to1 & _roundMask[54:0];	// MulAddRecFN.scala:454:{24,40}, :455:30
  wire [54:0] _T_53 = {_T_38[53:0], _T_39} & _roundMask_55to1;	// MulAddRecFN.scala:456:34
  wire _T_54 = (~{_T_38[53:0], _T_39} & _roundMask_55to1) == 55'h0;	// MulAddRecFN.scala:457:{27,34,50}, :477:12
  wire _anyRound = |_T_52 | |_T_53;	// MulAddRecFN.scala:455:46, :456:50, :458:32
  wire _allRound = |_T_52 & _T_54;	// MulAddRecFN.scala:455:46, :459:32
  wire _roundDirectUp = _T_43 ? _roundingMode_min : &io_fromPreMul_roundingMode;	// MulAddRecFN.scala:229:59, :460:28
  wire _T_55 = ~_T_35 & _T & |_T_52 & |_T_53 | ~_T_35 & _roundDirectUp & _anyRound | _T_35 & _allRound |
                _T_35 & _T & |_T_52 | _T_35 & _roundDirectUp;	// MulAddRecFN.scala:455:46, :456:50, :462:10, :463:60, :464:49, :465:49, :466:{49,65}, :467:20
  wire _roundEven = _T_35 ? _T & ~(|_T_52) & _T_54 : _T & |_T_52 & ~(|_T_53);	// MulAddRecFN.scala:455:46, :456:50, :469:12, :470:{42,56}, :471:{56,59}
  wire _inexactY = _T_35 ? ~_allRound : _anyRound;	// MulAddRecFN.scala:473:{27,39}
  wire [54:0] _T_56 = {_T_38[55], _T_38[54:1] | _roundMask[55:2]} + 55'h1;	// MulAddRecFN.scala:238:50, :475:{18,35}
  wire [54:0] _T_57 = (_T_55 | _roundEven ? 55'h0 : _T_38[55:1] & {1'h0, ~(_roundMask[55:2])}) | (_T_55 ? _T_56 :
                55'h0) | (_roundEven ? _T_56 & ~_roundMask_55to1 : 55'h0);	// MulAddRecFN.scala:207:46, :454:24, :477:{12,46,48}, :478:{12,79}, :479:{12,51}
  wire [12:0] _T_58 = (_T_57[54] ? _T_41[12:0] + 13'h1 : 13'h0) | (_T_57[53] ? _T_41[12:0] : 13'h0) |
                (_T_57[54:53] == 2'h0 ? _T_41[12:0] - 13'h1 : 13'h0);	// MulAddRecFN.scala:226:37, :482:{12,18,41}, :483:{12,18,61}, :484:{12,19,44}, :485:20
  wire [11:0] _T_59 = _T_58[11:0];	// MulAddRecFN.scala:488:21
  wire _totalUnderflowY = |_T_42 & (_T_58[12] | _T_59 < 12'h3CE);	// MulAddRecFN.scala:439:54, :495:19, :496:{19,34,57}, :557:16
  wire _roundMagUp = _roundingMode_min & _T_43 | &io_fromPreMul_roundingMode & ~_T_43;	// MulAddRecFN.scala:229:59, :506:{27,37,58,61}
  wire _overflowY_roundMagUp = _T | _roundMagUp;	// MulAddRecFN.scala:507:58
  wire _mulSpecial = &_io_fromPreMul_highExpA_2to1 | &_io_fromPreMul_highExpB_2to1;	// MulAddRecFN.scala:208:52, :214:52, :511:33
  wire _notSpecial_addZeros = io_fromPreMul_isZeroProd & ~(|io_fromPreMul_highExpC);	// MulAddRecFN.scala:219:46, :513:56
  wire _commonCase = ~(_mulSpecial | &_io_fromPreMul_highExpC_2to1) & ~_notSpecial_addZeros;	// MulAddRecFN.scala:220:52, :512:33, :514:{22,35,38}
  wire _T_60 = _isInfA | _isInfB;	// MulAddRecFN.scala:518:46
  wire _T_61 = _isInfA & io_fromPreMul_highExpB == 3'h0 | io_fromPreMul_highExpA == 3'h0 & _isInfB |
                ~_isNaNA & ~_isNaNB & _T_60 & _isInfC & _doSubMags;	// MulAddRecFN.scala:207:46, :213:46, :517:{17,41,52}, :518:{14,26,67}
  wire _overflow = _commonCase & _T_58[12:10] == 3'h3;	// MulAddRecFN.scala:492:{27,56}, :520:32
  wire _T_62 = _commonCase & _totalUnderflowY & _roundMagUp;	// MulAddRecFN.scala:526:60
  wire _pegMaxFiniteMagOut = _overflow & ~_overflowY_roundMagUp;	// MulAddRecFN.scala:527:{39,42}
  wire _T_63 = _T_60 | _isInfC | _overflow & _overflowY_roundMagUp;	// MulAddRecFN.scala:529:{36,49}
  wire _T_64 = _isNaNA | _isNaNB | _isNaNC | _T_61;	// MulAddRecFN.scala:530:47
  assign io_out = {~_T_64 & (~_doSubMags & io_fromPreMul_opSignC | _mulSpecial &
                ~(&_io_fromPreMul_highExpC_2to1) & io_fromPreMul_signProd | ~_mulSpecial &
                &_io_fromPreMul_highExpC_2to1 & io_fromPreMul_opSignC | ~_mulSpecial & _notSpecial_addZeros
                & _doSubMags & _roundingMode_min) | _commonCase & _T_43, _T_59 & ~(_notSpecial_addZeros |
                ~(|_T_42) | _totalUnderflowY ? 12'hE00 : 12'h0) & ~(_T_62 ? 12'hC31 : 12'h0) & {1'h1,
                ~_pegMaxFiniteMagOut, 10'h3FF} & {2'h3, ~_T_63, 9'h1FF} | (_T_62 ? 12'h3CE : 12'h0) |
                (_pegMaxFiniteMagOut ? 12'hBFF : 12'h0) | (_T_63 ? 12'hC00 : 12'h0) | (_T_64 ? 12'hE00 :
                12'h0), (_totalUnderflowY & _roundMagUp | _T_64 ? {_T_64, 51'h0} : _T_40 ? _T_57[51:0] :
                _T_57[52:1]) | {52{_pegMaxFiniteMagOut}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5924:10, Bitwise.scala:71:12, Cat.scala:30:58, MulAddRecFN.scala:208:52, :220:52, :231:35, :271:13, :439:54, :490:{12,31,55}, :525:40, :533:51, :534:{24,51}, :535:{10,51,78}, :536:59, :538:{20,31,55,70}, :541:{14,18}, :545:{14,18}, :546:19, :549:{14,18}, :552:17, :553:{14,18}, :557:16, :558:16, :562:16, :565:15, :566:16, :568:{12,30,45}, :569:16, :571:11
  assign io_exceptionFlags = {_isNaNA & ~io_fromPreMul_isNaN_isQuietNaNA | _isNaNB & ~io_fromPreMul_isNaN_isQuietNaNB |
                _isNaNC & ~io_fromPreMul_isNaN_isQuietNaNC | _T_61, 1'h0, _overflow, _commonCase &
                _inexactY & (_T_45 | _T_44 <= {2'h0, _T_40 ? 11'h402 : 11'h401}), _overflow | _commonCase &
                _inexactY};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/FPU.fir:5924:10, Cat.scala:30:58, MulAddRecFN.scala:207:46, :211:{28,31}, :217:{28,31}, :223:{28,31}, :226:37, :499:35, :500:29, :501:26, :519:55, :521:32, :522:{28,43}
endmodule

