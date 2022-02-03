// Standard header to adapt well known macros to our needs.
`ifdef RANDOMIZE_REG_INIT
  `define RANDOMIZE
`endif

// RANDOM may be set to an expression that produces a 32-bit random unsigned value.
`ifndef RANDOM
  `define RANDOM {$random}
`endif

// Users can define 'PRINTF_COND' to add an extra gate to prints.
`ifdef PRINTF_COND
  `define PRINTF_COND_ (`PRINTF_COND)
`else
  `define PRINTF_COND_ 1
`endif

// Users can define 'STOP_COND' to add an extra gate to stop conditions.
`ifdef STOP_COND
  `define STOP_COND_ (`STOP_COND)
`else
  `define STOP_COND_ 1
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

module _ext(	// Rocket.scala:682:23
  input  [4:0]  R0_addr,
  input         R0_en, R0_clk,
  input  [4:0]  R1_addr,
  input         R1_en, R1_clk,
  input  [4:0]  W0_addr,
  input         W0_en, W0_clk,
  input  [63:0] W0_data,
  output [63:0] R0_data, R1_data);

  reg [63:0] Memory[0:30];

  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end // always @(posedge)
  assign R0_data = R0_en ? Memory[R0_addr] : 64'bx;	// Rocket.scala:682:23
  assign R1_data = R1_en ? Memory[R1_addr] : 64'bx;	// Rocket.scala:682:23
endmodule

module RocketCore(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  input         clock, reset, io_interrupts_debug, io_interrupts_mtip,
  input         io_interrupts_msip, io_interrupts_meip, io_interrupts_seip,
  input  [63:0] io_hartid,
  input         io_imem_resp_valid, io_imem_resp_bits_btb_valid,
  input         io_imem_resp_bits_btb_bits_taken,
  input  [1:0]  io_imem_resp_bits_btb_bits_mask,
  input         io_imem_resp_bits_btb_bits_bridx,
  input  [38:0] io_imem_resp_bits_btb_bits_target,
  input  [5:0]  io_imem_resp_bits_btb_bits_entry,
  input  [6:0]  io_imem_resp_bits_btb_bits_bht_history,
  input  [1:0]  io_imem_resp_bits_btb_bits_bht_value,
  input  [39:0] io_imem_resp_bits_pc,
  input  [31:0] io_imem_resp_bits_data,
  input  [1:0]  io_imem_resp_bits_mask,
  input         io_imem_resp_bits_xcpt_if, io_imem_resp_bits_replay,
  input  [39:0] io_imem_npc,
  input         io_imem_acquire, io_dmem_req_ready, io_dmem_s2_nack, io_dmem_acquire,
  input         io_dmem_release, io_dmem_resp_valid,
  input  [39:0] io_dmem_resp_bits_addr,
  input  [6:0]  io_dmem_resp_bits_tag,
  input  [4:0]  io_dmem_resp_bits_cmd,
  input  [2:0]  io_dmem_resp_bits_typ,
  input  [63:0] io_dmem_resp_bits_data,
  input         io_dmem_resp_bits_replay, io_dmem_resp_bits_has_data,
  input  [63:0] io_dmem_resp_bits_data_word_bypass, io_dmem_resp_bits_store_data,
  input         io_dmem_replay_next, io_dmem_xcpt_ma_ld, io_dmem_xcpt_ma_st,
  input         io_dmem_xcpt_pf_ld, io_dmem_xcpt_pf_st, io_dmem_ordered,
  input         io_fpu_fcsr_flags_valid,
  input  [4:0]  io_fpu_fcsr_flags_bits,
  input  [63:0] io_fpu_store_data, io_fpu_toint_data,
  input         io_fpu_fcsr_rdy, io_fpu_nack_mem, io_fpu_illegal_rm,
  input  [4:0]  io_fpu_dec_cmd,
  input         io_fpu_dec_ldst, io_fpu_dec_wen, io_fpu_dec_ren1, io_fpu_dec_ren2,
  input         io_fpu_dec_ren3, io_fpu_dec_swap12, io_fpu_dec_swap23, io_fpu_dec_single,
  input         io_fpu_dec_fromint, io_fpu_dec_toint, io_fpu_dec_fastpipe,
  input         io_fpu_dec_fma, io_fpu_dec_div, io_fpu_dec_sqrt, io_fpu_dec_wflags,
  input         io_fpu_sboard_set, io_fpu_sboard_clr,
  input  [4:0]  io_fpu_sboard_clra,
  input         io_rocc_cmd_ready, io_rocc_resp_valid,
  input  [4:0]  io_rocc_resp_bits_rd,
  input  [63:0] io_rocc_resp_bits_data,
  input         io_rocc_mem_req_valid,
  input  [39:0] io_rocc_mem_req_bits_addr,
  input  [6:0]  io_rocc_mem_req_bits_tag,
  input  [4:0]  io_rocc_mem_req_bits_cmd,
  input  [2:0]  io_rocc_mem_req_bits_typ,
  input         io_rocc_mem_req_bits_phys,
  input  [63:0] io_rocc_mem_req_bits_data,
  input         io_rocc_mem_s1_kill,
  input  [63:0] io_rocc_mem_s1_data,
  input         io_rocc_mem_invalidate_lr, io_rocc_busy, io_rocc_interrupt,
  output        io_imem_req_valid,
  output [39:0] io_imem_req_bits_pc,
  output        io_imem_req_bits_speculative, io_imem_resp_ready,
  output        io_imem_btb_update_valid, io_imem_btb_update_bits_prediction_valid,
  output        io_imem_btb_update_bits_prediction_bits_taken,
  output [1:0]  io_imem_btb_update_bits_prediction_bits_mask,
  output        io_imem_btb_update_bits_prediction_bits_bridx,
  output [38:0] io_imem_btb_update_bits_prediction_bits_target,
  output [5:0]  io_imem_btb_update_bits_prediction_bits_entry,
  output [6:0]  io_imem_btb_update_bits_prediction_bits_bht_history,
  output [1:0]  io_imem_btb_update_bits_prediction_bits_bht_value,
  output [38:0] io_imem_btb_update_bits_pc, io_imem_btb_update_bits_target,
  output        io_imem_btb_update_bits_taken, io_imem_btb_update_bits_isValid,
  output        io_imem_btb_update_bits_isJump, io_imem_btb_update_bits_isReturn,
  output [38:0] io_imem_btb_update_bits_br_pc,
  output        io_imem_bht_update_valid, io_imem_bht_update_bits_prediction_valid,
  output        io_imem_bht_update_bits_prediction_bits_taken,
  output [1:0]  io_imem_bht_update_bits_prediction_bits_mask,
  output        io_imem_bht_update_bits_prediction_bits_bridx,
  output [38:0] io_imem_bht_update_bits_prediction_bits_target,
  output [5:0]  io_imem_bht_update_bits_prediction_bits_entry,
  output [6:0]  io_imem_bht_update_bits_prediction_bits_bht_history,
  output [1:0]  io_imem_bht_update_bits_prediction_bits_bht_value,
  output [38:0] io_imem_bht_update_bits_pc,
  output        io_imem_bht_update_bits_taken, io_imem_bht_update_bits_mispredict,
  output        io_imem_ras_update_valid, io_imem_ras_update_bits_isCall,
  output        io_imem_ras_update_bits_isReturn,
  output [38:0] io_imem_ras_update_bits_returnAddr,
  output        io_imem_ras_update_bits_prediction_valid,
  output        io_imem_ras_update_bits_prediction_bits_taken,
  output [1:0]  io_imem_ras_update_bits_prediction_bits_mask,
  output        io_imem_ras_update_bits_prediction_bits_bridx,
  output [38:0] io_imem_ras_update_bits_prediction_bits_target,
  output [5:0]  io_imem_ras_update_bits_prediction_bits_entry,
  output [6:0]  io_imem_ras_update_bits_prediction_bits_bht_history,
  output [1:0]  io_imem_ras_update_bits_prediction_bits_bht_value,
  output        io_imem_flush_icache, io_imem_flush_tlb, io_dmem_req_valid,
  output [39:0] io_dmem_req_bits_addr,
  output [6:0]  io_dmem_req_bits_tag,
  output [4:0]  io_dmem_req_bits_cmd,
  output [2:0]  io_dmem_req_bits_typ,
  output        io_dmem_req_bits_phys,
  output [63:0] io_dmem_req_bits_data,
  output        io_dmem_s1_kill,
  output [63:0] io_dmem_s1_data,
  output        io_dmem_invalidate_lr,
  output [3:0]  io_ptw_ptbr_mode,
  output [15:0] io_ptw_ptbr_asid,
  output [43:0] io_ptw_ptbr_ppn,
  output        io_ptw_invalidate, io_ptw_status_debug,
  output [31:0] io_ptw_status_isa,
  output [1:0]  io_ptw_status_prv,
  output        io_ptw_status_sd,
  output [26:0] io_ptw_status_zero2,
  output [1:0]  io_ptw_status_sxl, io_ptw_status_uxl,
  output        io_ptw_status_sd_rv32,
  output [7:0]  io_ptw_status_zero1,
  output        io_ptw_status_tsr, io_ptw_status_tw, io_ptw_status_tvm,
  output        io_ptw_status_mxr, io_ptw_status_pum, io_ptw_status_mprv,
  output [1:0]  io_ptw_status_xs, io_ptw_status_fs, io_ptw_status_mpp, io_ptw_status_hpp,
  output        io_ptw_status_spp, io_ptw_status_mpie, io_ptw_status_hpie,
  output        io_ptw_status_spie, io_ptw_status_upie, io_ptw_status_mie,
  output        io_ptw_status_hie, io_ptw_status_sie, io_ptw_status_uie,
  output [31:0] io_fpu_inst,
  output [63:0] io_fpu_fromint_data,
  output [2:0]  io_fpu_fcsr_rm,
  output        io_fpu_dmem_resp_val,
  output [2:0]  io_fpu_dmem_resp_type,
  output [4:0]  io_fpu_dmem_resp_tag,
  output [63:0] io_fpu_dmem_resp_data,
  output        io_fpu_valid, io_fpu_killx, io_fpu_killm, io_rocc_cmd_valid,
  output [6:0]  io_rocc_cmd_bits_inst_funct,
  output [4:0]  io_rocc_cmd_bits_inst_rs2, io_rocc_cmd_bits_inst_rs1,
  output        io_rocc_cmd_bits_inst_xd, io_rocc_cmd_bits_inst_xs1,
  output        io_rocc_cmd_bits_inst_xs2,
  output [4:0]  io_rocc_cmd_bits_inst_rd,
  output [6:0]  io_rocc_cmd_bits_inst_opcode,
  output [63:0] io_rocc_cmd_bits_rs1, io_rocc_cmd_bits_rs2,
  output        io_rocc_cmd_bits_status_debug,
  output [31:0] io_rocc_cmd_bits_status_isa,
  output [1:0]  io_rocc_cmd_bits_status_prv,
  output        io_rocc_cmd_bits_status_sd,
  output [26:0] io_rocc_cmd_bits_status_zero2,
  output [1:0]  io_rocc_cmd_bits_status_sxl, io_rocc_cmd_bits_status_uxl,
  output        io_rocc_cmd_bits_status_sd_rv32,
  output [7:0]  io_rocc_cmd_bits_status_zero1,
  output        io_rocc_cmd_bits_status_tsr, io_rocc_cmd_bits_status_tw,
  output        io_rocc_cmd_bits_status_tvm, io_rocc_cmd_bits_status_mxr,
  output        io_rocc_cmd_bits_status_pum, io_rocc_cmd_bits_status_mprv,
  output [1:0]  io_rocc_cmd_bits_status_xs, io_rocc_cmd_bits_status_fs,
  output [1:0]  io_rocc_cmd_bits_status_mpp, io_rocc_cmd_bits_status_hpp,
  output        io_rocc_cmd_bits_status_spp, io_rocc_cmd_bits_status_mpie,
  output        io_rocc_cmd_bits_status_hpie, io_rocc_cmd_bits_status_spie,
  output        io_rocc_cmd_bits_status_upie, io_rocc_cmd_bits_status_mie,
  output        io_rocc_cmd_bits_status_hie, io_rocc_cmd_bits_status_sie,
  output        io_rocc_cmd_bits_status_uie, io_rocc_resp_ready, io_rocc_mem_req_ready,
  output        io_rocc_mem_s2_nack, io_rocc_mem_acquire, io_rocc_mem_release,
  output        io_rocc_mem_resp_valid,
  output [39:0] io_rocc_mem_resp_bits_addr,
  output [6:0]  io_rocc_mem_resp_bits_tag,
  output [4:0]  io_rocc_mem_resp_bits_cmd,
  output [2:0]  io_rocc_mem_resp_bits_typ,
  output [63:0] io_rocc_mem_resp_bits_data,
  output        io_rocc_mem_resp_bits_replay, io_rocc_mem_resp_bits_has_data,
  output [63:0] io_rocc_mem_resp_bits_data_word_bypass, io_rocc_mem_resp_bits_store_data,
  output        io_rocc_mem_replay_next, io_rocc_mem_xcpt_ma_ld, io_rocc_mem_xcpt_ma_st,
  output        io_rocc_mem_xcpt_pf_ld, io_rocc_mem_xcpt_pf_st, io_rocc_mem_ordered,
  output        io_rocc_exception);

  wire        _T;	// Rocket.scala:581:41
  wire        _T_0;	// Rocket.scala:443:23
  wire        _T_1;	// Rocket.scala:420:38
  wire        _T_2;	// Rocket.scala:351:32
  wire        div_io_req_ready;	// Rocket.scala:268:19
  wire        div_io_resp_valid;	// Rocket.scala:268:19
  wire [63:0] div_io_resp_bits_data;	// Rocket.scala:268:19
  wire [4:0]  div_io_resp_bits_tag;	// Rocket.scala:268:19
  wire [63:0] alu_io_out;	// Rocket.scala:261:19
  wire [63:0] alu_io_adder_out;	// Rocket.scala:261:19
  wire        alu_io_cmp_out;	// Rocket.scala:261:19
  wire        bpu_io_xcpt_if;	// Rocket.scala:215:19
  wire        bpu_io_xcpt_ld;	// Rocket.scala:215:19
  wire        bpu_io_xcpt_st;	// Rocket.scala:215:19
  wire        bpu_io_debug_if;	// Rocket.scala:215:19
  wire        bpu_io_debug_ld;	// Rocket.scala:215:19
  wire        bpu_io_debug_st;	// Rocket.scala:215:19
  wire [63:0] csr_io_rw_rdata;	// Rocket.scala:187:19
  wire        csr_io_decode_fp_illegal;	// Rocket.scala:187:19
  wire        csr_io_decode_read_illegal;	// Rocket.scala:187:19
  wire        csr_io_decode_write_illegal;	// Rocket.scala:187:19
  wire        csr_io_decode_write_flush;	// Rocket.scala:187:19
  wire        csr_io_decode_system_illegal;	// Rocket.scala:187:19
  wire        csr_io_csr_stall;	// Rocket.scala:187:19
  wire        csr_io_eret;	// Rocket.scala:187:19
  wire        csr_io_singleStep;	// Rocket.scala:187:19
  wire        csr_io_status_debug;	// Rocket.scala:187:19
  wire [31:0] csr_io_status_isa;	// Rocket.scala:187:19
  wire [1:0]  csr_io_status_prv;	// Rocket.scala:187:19
  wire        csr_io_status_sd;	// Rocket.scala:187:19
  wire        csr_io_status_tsr;	// Rocket.scala:187:19
  wire        csr_io_status_tw;	// Rocket.scala:187:19
  wire        csr_io_status_tvm;	// Rocket.scala:187:19
  wire        csr_io_status_mxr;	// Rocket.scala:187:19
  wire        csr_io_status_pum;	// Rocket.scala:187:19
  wire        csr_io_status_mprv;	// Rocket.scala:187:19
  wire [1:0]  csr_io_status_fs;	// Rocket.scala:187:19
  wire [1:0]  csr_io_status_mpp;	// Rocket.scala:187:19
  wire        csr_io_status_spp;	// Rocket.scala:187:19
  wire        csr_io_status_mpie;	// Rocket.scala:187:19
  wire        csr_io_status_spie;	// Rocket.scala:187:19
  wire        csr_io_status_mie;	// Rocket.scala:187:19
  wire        csr_io_status_sie;	// Rocket.scala:187:19
  wire [39:0] csr_io_evec;	// Rocket.scala:187:19
  wire        csr_io_fatc;	// Rocket.scala:187:19
  wire [63:0] csr_io_time;	// Rocket.scala:187:19
  wire        csr_io_interrupt;	// Rocket.scala:187:19
  wire [63:0] csr_io_interrupt_cause;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_dmode;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_action;	// Rocket.scala:187:19
  wire [1:0]  csr_io_bp_0_control_tmatch;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_m;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_s;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_u;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_x;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_w;	// Rocket.scala:187:19
  wire        csr_io_bp_0_control_r;	// Rocket.scala:187:19
  wire [38:0] csr_io_bp_0_address;	// Rocket.scala:187:19
  wire [63:0] mem_R0_data;	// Rocket.scala:682:23
  wire [63:0] mem_R1_data;	// Rocket.scala:682:23
  wire [39:0] ibuf_io_pc;	// Rocket.scala:165:20
  wire        ibuf_io_btb_resp_taken;	// Rocket.scala:165:20
  wire [1:0]  ibuf_io_btb_resp_mask;	// Rocket.scala:165:20
  wire        ibuf_io_btb_resp_bridx;	// Rocket.scala:165:20
  wire [38:0] ibuf_io_btb_resp_target;	// Rocket.scala:165:20
  wire [5:0]  ibuf_io_btb_resp_entry;	// Rocket.scala:165:20
  wire [6:0]  ibuf_io_btb_resp_bht_history;	// Rocket.scala:165:20
  wire [1:0]  ibuf_io_btb_resp_bht_value;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_valid;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_bits_pf0;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_bits_pf1;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_bits_replay;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_bits_btb_hit;	// Rocket.scala:165:20
  wire        ibuf_io_inst_0_bits_rvc;	// Rocket.scala:165:20
  wire [31:0] ibuf_io_inst_0_bits_inst_bits;	// Rocket.scala:165:20
  wire [4:0]  ibuf_io_inst_0_bits_inst_rd;	// Rocket.scala:165:20
  wire [4:0]  ibuf_io_inst_0_bits_inst_rs1;	// Rocket.scala:165:20
  wire [4:0]  ibuf_io_inst_0_bits_inst_rs2;	// Rocket.scala:165:20
  wire [4:0]  ibuf_io_inst_0_bits_inst_rs3;	// Rocket.scala:165:20
  wire [31:0] ibuf_io_inst_0_bits_raw;	// Rocket.scala:165:20
  reg  [63:0] casez_tmp;	// Rocket.scala:251:14
  reg  [63:0] casez_tmp_3;	// Rocket.scala:251:14
  reg         ex_ctrl_legal;	// Rocket.scala:115:20
  reg         ex_ctrl_fp;	// Rocket.scala:115:20
  reg         ex_ctrl_branch;	// Rocket.scala:115:20
  reg         ex_ctrl_jal;	// Rocket.scala:115:20
  reg         ex_ctrl_jalr;	// Rocket.scala:115:20
  reg         ex_ctrl_rxs2;	// Rocket.scala:115:20
  reg         ex_ctrl_rxs1;	// Rocket.scala:115:20
  reg  [1:0]  ex_ctrl_sel_alu2;	// Rocket.scala:115:20
  reg  [1:0]  ex_ctrl_sel_alu1;	// Rocket.scala:115:20
  reg  [2:0]  ex_ctrl_sel_imm;	// Rocket.scala:115:20
  reg         ex_ctrl_alu_dw;	// Rocket.scala:115:20
  reg  [3:0]  ex_ctrl_alu_fn;	// Rocket.scala:115:20
  reg         ex_ctrl_mem;	// Rocket.scala:115:20
  reg  [4:0]  ex_ctrl_mem_cmd;	// Rocket.scala:115:20
  reg  [2:0]  ex_ctrl_mem_type;	// Rocket.scala:115:20
  reg         ex_ctrl_rfs1;	// Rocket.scala:115:20
  reg         ex_ctrl_rfs2;	// Rocket.scala:115:20
  reg         ex_ctrl_rfs3;	// Rocket.scala:115:20
  reg         ex_ctrl_wfd;	// Rocket.scala:115:20
  reg         ex_ctrl_div;	// Rocket.scala:115:20
  reg         ex_ctrl_wxd;	// Rocket.scala:115:20
  reg  [2:0]  ex_ctrl_csr;	// Rocket.scala:115:20
  reg         ex_ctrl_fence_i;	// Rocket.scala:115:20
  reg         ex_ctrl_fence;	// Rocket.scala:115:20
  reg         ex_ctrl_amo;	// Rocket.scala:115:20
  reg         ex_ctrl_dp;	// Rocket.scala:115:20
  reg         mem_ctrl_legal;	// Rocket.scala:116:21
  reg         mem_ctrl_fp;	// Rocket.scala:116:21
  reg         mem_ctrl_branch;	// Rocket.scala:116:21
  reg         mem_ctrl_jal;	// Rocket.scala:116:21
  reg         mem_ctrl_jalr;	// Rocket.scala:116:21
  reg         mem_ctrl_rxs2;	// Rocket.scala:116:21
  reg         mem_ctrl_rxs1;	// Rocket.scala:116:21
  reg  [1:0]  mem_ctrl_sel_alu2;	// Rocket.scala:116:21
  reg  [1:0]  mem_ctrl_sel_alu1;	// Rocket.scala:116:21
  reg  [2:0]  mem_ctrl_sel_imm;	// Rocket.scala:116:21
  reg         mem_ctrl_alu_dw;	// Rocket.scala:116:21
  reg  [3:0]  mem_ctrl_alu_fn;	// Rocket.scala:116:21
  reg         mem_ctrl_mem;	// Rocket.scala:116:21
  reg  [4:0]  mem_ctrl_mem_cmd;	// Rocket.scala:116:21
  reg  [2:0]  mem_ctrl_mem_type;	// Rocket.scala:116:21
  reg         mem_ctrl_rfs1;	// Rocket.scala:116:21
  reg         mem_ctrl_rfs2;	// Rocket.scala:116:21
  reg         mem_ctrl_rfs3;	// Rocket.scala:116:21
  reg         mem_ctrl_wfd;	// Rocket.scala:116:21
  reg         mem_ctrl_div;	// Rocket.scala:116:21
  reg         mem_ctrl_wxd;	// Rocket.scala:116:21
  reg  [2:0]  mem_ctrl_csr;	// Rocket.scala:116:21
  reg         mem_ctrl_fence_i;	// Rocket.scala:116:21
  reg         mem_ctrl_fence;	// Rocket.scala:116:21
  reg         mem_ctrl_amo;	// Rocket.scala:116:21
  reg         mem_ctrl_dp;	// Rocket.scala:116:21
  reg         wb_ctrl_legal;	// Rocket.scala:117:20
  reg         wb_ctrl_fp;	// Rocket.scala:117:20
  reg         wb_ctrl_branch;	// Rocket.scala:117:20
  reg         wb_ctrl_jal;	// Rocket.scala:117:20
  reg         wb_ctrl_jalr;	// Rocket.scala:117:20
  reg         wb_ctrl_rxs2;	// Rocket.scala:117:20
  reg         wb_ctrl_rxs1;	// Rocket.scala:117:20
  reg  [1:0]  wb_ctrl_sel_alu2;	// Rocket.scala:117:20
  reg  [1:0]  wb_ctrl_sel_alu1;	// Rocket.scala:117:20
  reg  [2:0]  wb_ctrl_sel_imm;	// Rocket.scala:117:20
  reg         wb_ctrl_alu_dw;	// Rocket.scala:117:20
  reg  [3:0]  wb_ctrl_alu_fn;	// Rocket.scala:117:20
  reg         wb_ctrl_mem;	// Rocket.scala:117:20
  reg  [4:0]  wb_ctrl_mem_cmd;	// Rocket.scala:117:20
  reg  [2:0]  wb_ctrl_mem_type;	// Rocket.scala:117:20
  reg         wb_ctrl_rfs1;	// Rocket.scala:117:20
  reg         wb_ctrl_rfs2;	// Rocket.scala:117:20
  reg         wb_ctrl_rfs3;	// Rocket.scala:117:20
  reg         wb_ctrl_wfd;	// Rocket.scala:117:20
  reg         wb_ctrl_div;	// Rocket.scala:117:20
  reg         wb_ctrl_wxd;	// Rocket.scala:117:20
  reg  [2:0]  wb_ctrl_csr;	// Rocket.scala:117:20
  reg         wb_ctrl_fence_i;	// Rocket.scala:117:20
  reg         wb_ctrl_fence;	// Rocket.scala:117:20
  reg         wb_ctrl_amo;	// Rocket.scala:117:20
  reg         wb_ctrl_dp;	// Rocket.scala:117:20
  reg         ex_reg_xcpt_interrupt;	// Rocket.scala:119:35
  reg         ex_reg_valid;	// Rocket.scala:120:35
  reg         ex_reg_rvc;	// Rocket.scala:121:35
  reg         ex_reg_btb_hit;	// Rocket.scala:122:35
  reg         ex_reg_btb_resp_taken;	// Rocket.scala:123:35
  reg  [1:0]  ex_reg_btb_resp_mask;	// Rocket.scala:123:35
  reg         ex_reg_btb_resp_bridx;	// Rocket.scala:123:35
  reg  [38:0] ex_reg_btb_resp_target;	// Rocket.scala:123:35
  reg  [5:0]  ex_reg_btb_resp_entry;	// Rocket.scala:123:35
  reg  [6:0]  ex_reg_btb_resp_bht_history;	// Rocket.scala:123:35
  reg  [1:0]  ex_reg_btb_resp_bht_value;	// Rocket.scala:123:35
  reg         ex_reg_xcpt;	// Rocket.scala:124:35
  reg         ex_reg_flush_pipe;	// Rocket.scala:125:35
  reg         ex_reg_load_use;	// Rocket.scala:126:35
  reg  [63:0] ex_cause;	// Rocket.scala:127:35
  reg         ex_reg_replay;	// Rocket.scala:128:26
  reg  [39:0] ex_reg_pc;	// Rocket.scala:129:22
  reg  [31:0] ex_reg_inst;	// Rocket.scala:130:24
  reg         mem_reg_xcpt_interrupt;	// Rocket.scala:132:36
  reg         mem_reg_valid;	// Rocket.scala:133:36
  reg         mem_reg_rvc;	// Rocket.scala:134:36
  reg         mem_reg_btb_hit;	// Rocket.scala:135:36
  reg         mem_reg_btb_resp_taken;	// Rocket.scala:136:36
  reg  [1:0]  mem_reg_btb_resp_mask;	// Rocket.scala:136:36
  reg         mem_reg_btb_resp_bridx;	// Rocket.scala:136:36
  reg  [38:0] mem_reg_btb_resp_target;	// Rocket.scala:136:36
  reg  [5:0]  mem_reg_btb_resp_entry;	// Rocket.scala:136:36
  reg  [6:0]  mem_reg_btb_resp_bht_history;	// Rocket.scala:136:36
  reg  [1:0]  mem_reg_btb_resp_bht_value;	// Rocket.scala:136:36
  reg         mem_reg_xcpt;	// Rocket.scala:137:36
  reg         mem_reg_replay;	// Rocket.scala:138:36
  reg         mem_reg_flush_pipe;	// Rocket.scala:139:36
  reg  [63:0] mem_reg_cause;	// Rocket.scala:140:36
  reg         mem_reg_slow_bypass;	// Rocket.scala:141:36
  reg         mem_reg_load;	// Rocket.scala:142:36
  reg         mem_reg_store;	// Rocket.scala:143:36
  reg  [39:0] mem_reg_pc;	// Rocket.scala:144:23
  reg  [31:0] mem_reg_inst;	// Rocket.scala:145:25
  reg  [63:0] mem_reg_wdata;	// Rocket.scala:146:26
  reg  [63:0] mem_reg_rs2;	// Rocket.scala:147:24
  reg         wb_reg_valid;	// Rocket.scala:150:35
  reg         wb_reg_xcpt;	// Rocket.scala:151:35
  reg         wb_reg_replay;	// Rocket.scala:152:35
  reg  [63:0] wb_reg_cause;	// Rocket.scala:153:35
  reg  [39:0] wb_reg_pc;	// Rocket.scala:154:22
  reg  [31:0] wb_reg_inst;	// Rocket.scala:155:24
  reg  [63:0] wb_reg_wdata;	// Rocket.scala:156:25
  reg         id_reg_fence;	// Rocket.scala:178:25
  reg         ex_reg_rs_bypass_0;	// Rocket.scala:247:29
  reg         ex_reg_rs_bypass_1;	// Rocket.scala:247:29
  reg  [1:0]  ex_reg_rs_lsb_0;	// Rocket.scala:248:26
  reg  [1:0]  ex_reg_rs_lsb_1;	// Rocket.scala:248:26
  reg  [61:0] ex_reg_rs_msb_0;	// Rocket.scala:249:26
  reg  [61:0] ex_reg_rs_msb_1;	// Rocket.scala:249:26
  reg         _T_4_40;	// Rocket.scala:396:37
  reg  [31:0] _T_5_51;	// Rocket.scala:668:25
  reg  [31:0] _T_6_68;	// Rocket.scala:668:25
  reg         dcache_blocked;	// Rocket.scala:522:27
  reg         rocc_blocked;	// Rocket.scala:524:25
  reg  [63:0] _T_7_82;	// Rocket.scala:639:42
  reg  [63:0] _T_8_83;	// Rocket.scala:639:33
  reg  [63:0] _T_9_84;	// Rocket.scala:640:42
  reg  [63:0] _T_10_85;	// Rocket.scala:640:33

  wire _take_pc_mem_wb = _T_1 | _T_2;	// Rocket.scala:161:35, :351:32, :420:38
  wire _T_3 = {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[6:0]} == 8'h3;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_4 = {ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:3], ibuf_io_inst_0_bits_inst_bits[1:0]} == 7'h23;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_5 = {ibuf_io_inst_0_bits_inst_bits[28:27], ibuf_io_inst_0_bits_inst_bits[14:13],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 11'hAF;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_6 = {ibuf_io_inst_0_bits_inst_bits[31:29], ibuf_io_inst_0_bits_inst_bits[27],
                ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6:0]} == 13'h2AF;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_7 = {ibuf_io_inst_0_bits_inst_bits[31:27], ibuf_io_inst_0_bits_inst_bits[24:20],
                ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6:0]} == 19'h80AF;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_8 = {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:2]} == 4'h1 |
                ibuf_io_inst_0_bits_inst_bits[6:5] == 2'h2;	// Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20
  wire _T_9 = {ibuf_io_inst_0_bits_inst_bits[6:5], ibuf_io_inst_0_bits_inst_bits[2]} == 3'h2 |
                {ibuf_io_inst_0_bits_inst_bits[5:4], ibuf_io_inst_0_bits_inst_bits[2]} == 3'h4 |
                {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[3]} == 3'h5 | {ibuf_io_inst_0_bits_inst_bits[30],
                ibuf_io_inst_0_bits_inst_bits[25], ibuf_io_inst_0_bits_inst_bits[13:12],
                ibuf_io_inst_0_bits_inst_bits[5], ibuf_io_inst_0_bits_inst_bits[2]} == 6'h12;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20, :704:24, :705:26, :715:22
  wire _T_10 = {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[2]} == 2'h0;	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :292:24
  wire _T_11 = _T_10 | {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[5],
                ibuf_io_inst_0_bits_inst_bits[2]} == 3'h2 | ibuf_io_inst_0_bits_inst_bits[5:3] == 3'h4 |
                {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4]} == 3'h4 | {ibuf_io_inst_0_bits_inst_bits[31],
                ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[5:4],
                ibuf_io_inst_0_bits_inst_bits[2]} == 5'h1A;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20, :705:26, :715:22
  wire _T_12 = {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:0]} == 7'h3 | _T_3 | {ibuf_io_inst_0_bits_inst_bits[12],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 8'h3 | {ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 10'h8F | _T_4 | _T_5 | _T_6 | _T_7;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20
  wire _T_13 = {ibuf_io_inst_0_bits_inst_bits[25], ibuf_io_inst_0_bits_inst_bits[6:4],
                ibuf_io_inst_0_bits_inst_bits[2]} == 5'h16;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_14 = {ibuf_io_inst_0_bits_inst_bits[6:5], ibuf_io_inst_0_bits_inst_bits[2]} == 3'h0 |
                {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4]} == 2'h1 |
                {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[5],
                ibuf_io_inst_0_bits_inst_bits[2]} == 3'h3 | &{ibuf_io_inst_0_bits_inst_bits[5],
                ibuf_io_inst_0_bits_inst_bits[3]} | &{ibuf_io_inst_0_bits_inst_bits[12],
                ibuf_io_inst_0_bits_inst_bits[5:4]} | &{ibuf_io_inst_0_bits_inst_bits[13],
                ibuf_io_inst_0_bits_inst_bits[5:4]} | {ibuf_io_inst_0_bits_inst_bits[31],
                ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[4]} == 3'h5;	// Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20, :704:24, :712:24
  wire _T_15 = {ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6:4]} == 5'h7;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire [2:0] _T_16 = {_T_15, &{ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[6:4]},
                &{ibuf_io_inst_0_bits_inst_bits[12], ibuf_io_inst_0_bits_inst_bits[6:4]}};	// Cat.scala:30:58, Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_17 = {ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:3]} == 5'h9;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_18 = {ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[3]} == 4'h5;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
  wire _T_19 = _T_16 == 3'h2 | _T_16 == 3'h3;	// Package.scala:7:{47,62}, Rocket.scala:704:24, :705:26
  wire _id_csr_en = _T_19 | _T_16 == 3'h1;	// Package.scala:7:{47,62}, Rocket.scala:709:23
  wire _csr_io_status_isa_2 = csr_io_status_isa[2];	// Rocket.scala:187:19, :199:51
  wire _id_mem_busy = ~io_dmem_ordered | _T;	// Rocket.scala:207:{21,38}, :581:41
  wire [38:0] _mem_reg_wdata_38to0 = mem_reg_wdata[38:0];	// Rocket.scala:219:13
  wire [4:0] _ex_waddr = ex_reg_inst[11:7];	// Rocket.scala:235:29
  wire [4:0] _mem_waddr = mem_reg_inst[11:7];	// Rocket.scala:236:31
  wire [4:0] _wb_waddr = wb_reg_inst[11:7];	// Rocket.scala:237:29
  wire _T_20 = mem_reg_valid & mem_ctrl_wxd;	// Rocket.scala:241:20
  always @(*) begin	// Rocket.scala:251:14
    casez (ex_reg_rs_lsb_0)	// Cat.scala:30:58, Rocket.scala:251:14
      2'b00:
        casez_tmp = 64'h0;	// Rocket.scala:157:23, :251:14
      2'b01:
        casez_tmp = mem_reg_wdata;	// Rocket.scala:219:13, :251:14
      2'b10:
        casez_tmp = wb_reg_wdata;	// Rocket.scala:246:23, :251:14
      default:
        casez_tmp = io_dmem_resp_bits_data_word_bypass;	// Rocket.scala:251:14
    endcase	// Cat.scala:30:58, Rocket.scala:251:14
  end // always @(*)
  wire [63:0] _ex_rs_0 = ex_reg_rs_bypass_0 ? casez_tmp : {ex_reg_rs_msb_0, ex_reg_rs_lsb_0};	// Cat.scala:30:58, Rocket.scala:251:14
  always @(*) begin	// Rocket.scala:251:14
    casez (ex_reg_rs_lsb_1)	// Cat.scala:30:58, Rocket.scala:251:14
      2'b00:
        casez_tmp_3 = 64'h0;	// Rocket.scala:157:23, :251:14
      2'b01:
        casez_tmp_3 = mem_reg_wdata;	// Rocket.scala:219:13, :251:14
      2'b10:
        casez_tmp_3 = wb_reg_wdata;	// Rocket.scala:246:23, :251:14
      default:
        casez_tmp_3 = io_dmem_resp_bits_data_word_bypass;	// Rocket.scala:251:14
    endcase	// Cat.scala:30:58, Rocket.scala:251:14
  end // always @(*)
  wire [63:0] _ex_rs_1 = ex_reg_rs_bypass_1 ? casez_tmp_3 : {ex_reg_rs_msb_1, ex_reg_rs_lsb_1};	// Cat.scala:30:58, Rocket.scala:251:14
  wire _T_21 = ex_ctrl_sel_imm == 3'h5;	// Rocket.scala:704:24
  wire _T_22 = ~_T_21 & ex_reg_inst[31];	// Rocket.scala:235:29, :704:{19,48}
  wire _T_23 = ex_ctrl_sel_imm == 3'h2;	// Rocket.scala:704:24, :705:26
  wire _T_24 = _T_23 | _T_21;	// Rocket.scala:707:33
  wire _ex_reg_inst_20 = ex_reg_inst[20];	// Rocket.scala:235:29, :708:39
  wire _T_25 = ex_ctrl_sel_imm == 3'h1;	// Rocket.scala:704:24, :709:23
  wire _ex_reg_inst_7 = ex_reg_inst[7];	// Rocket.scala:235:29, :709:39
  wire _T_26 = ex_ctrl_sel_imm == 3'h0;	// Rocket.scala:704:24, :712:24
  wire [39:0] _T_27 = ex_ctrl_sel_alu1 == 2'h2 ? ex_reg_pc : 40'h0;	// Mux.scala:31:69, :46:{16,19}, Rocket.scala:187:19
  wire [31:0] _T_28 = &ex_ctrl_sel_alu2 ? {_T_22, _T_23 ? ex_reg_inst[30:20] : {11{_T_22}}, ex_ctrl_sel_imm !=
                3'h2 & ex_ctrl_sel_imm != 3'h3 ? {8{_T_22}} : ex_reg_inst[19:12], ~_T_24 & (ex_ctrl_sel_imm
                == 3'h3 ? _ex_reg_inst_20 : _T_25 ? _ex_reg_inst_7 : _T_22), _T_24 ? 6'h0 :
                ex_reg_inst[30:25], _T_23 ? 4'h0 : _T_26 | _T_25 ? ex_reg_inst[11:8] : _T_21 ?
                ex_reg_inst[19:16] : ex_reg_inst[24:21], _T_26 ? _ex_reg_inst_7 : ex_ctrl_sel_imm == 3'h4 ?
                _ex_reg_inst_20 : _T_21 & ex_reg_inst[15]} : {28'h0, ex_ctrl_sel_alu2 == 2'h1 ? (ex_reg_rvc
                ? 4'h2 : 4'h4) : 4'h0};	// Cat.scala:30:58, Mux.scala:31:69, :46:{16,19}, Rocket.scala:187:19, :235:29, :259:19, :704:24, :705:{21,26,41}, :706:{21,26,36,43,65}, :707:18, :708:{18,23}, :709:18, :710:{20,66}, :711:19, :712:{19,34,57}, :713:{19,39,52}, :714:17, :715:{17,22}, :716:{17,37}
  wire _T_29 = ex_reg_valid & ex_ctrl_div;	// Rocket.scala:240:19, :269:36
  wire _T_30 = ex_reg_valid | ex_reg_replay | ex_reg_xcpt_interrupt;	// Rocket.scala:240:19, :323:{34,51}
  wire _wb_dcache_miss = wb_ctrl_mem & ~io_dmem_resp_valid;	// Rocket.scala:324:{36,39}
  wire _replay_ex = ex_reg_replay | ex_reg_valid & (ex_ctrl_mem & ~io_dmem_req_ready | ex_ctrl_div &
                ~div_io_req_ready | _wb_dcache_miss & ex_reg_load_use);	// Rocket.scala:240:19, :268:19, :269:36, :285:13, :299:21, :323:34, :325:{42,45}, :326:{42,45}, :327:43, :328:{33,50,75}
  wire _T_31 = _take_pc_mem_wb | _replay_ex | ~ex_reg_valid;	// Rocket.scala:240:19, :329:{48,51}
  wire _mem_br_taken = mem_reg_wdata[0];	// Rocket.scala:219:13, :338:35
  wire _T_32 = mem_ctrl_branch & _mem_br_taken;	// Rocket.scala:340:25
  wire _mem_reg_inst_31 = mem_reg_inst[31];	// Rocket.scala:236:31, :704:48
  wire _mem_reg_inst_7 = mem_reg_inst[7];	// Rocket.scala:236:31, :709:39
  wire [31:0] _T_33 = _T_32 ? {{20{_mem_reg_inst_31}}, _mem_reg_inst_7, mem_reg_inst[30:25], mem_reg_inst[11:8],
                1'h0} : mem_ctrl_jal ? {{12{_mem_reg_inst_31}}, mem_reg_inst[19:12], mem_reg_inst[20],
                mem_reg_inst[30:21], 1'h0} : {28'h0, mem_reg_rvc ? 4'h2 : 4'h4};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Cat.scala:30:58, Mux.scala:46:16, Rocket.scala:187:19, :236:31, :259:19, :340:8, :341:8, :342:8, :706:65, :708:39, :710:66, :712:57
  wire [39:0] _T_34 = mem_reg_pc + {{8{_T_33[31]}}, _T_33};	// Rocket.scala:339:41
  wire [25:0] _mem_reg_wdata_63to38 = mem_reg_wdata[63:38];	// Rocket.scala:219:13, :653:16
  wire [1:0] _mem_reg_wdata_39to38 = mem_reg_wdata[39:38];	// Rocket.scala:219:13, :654:15
  wire [39:0] _T_35 = mem_ctrl_jalr ? {_mem_reg_wdata_63to38 == 26'h0 | _mem_reg_wdata_63to38 == 26'h1 ?
                |_mem_reg_wdata_39to38 : &_mem_reg_wdata_63to38 | _mem_reg_wdata_63to38 == 26'h3FFFFFE ?
                &_mem_reg_wdata_39to38 : mem_reg_wdata[38], _mem_reg_wdata_38to0} : _T_34;	// Cat.scala:30:58, Rocket.scala:219:13, :343:21, :656:{10,13,25,30,45}, :657:{10,20,33,45,61,76}
  wire [39:0] _T_36 = _T_35 & 40'hFFFFFFFFFE;	// Rocket.scala:343:111
  wire _mem_misprediction = _T_30 ? _T_36 != ex_reg_pc : ~ibuf_io_inst_0_valid | _T_36 != ibuf_io_pc;	// Mux.scala:46:16, Rocket.scala:165:20, :344:{26,48,66,98}
  wire _mem_npc_misaligned = ~_csr_io_status_isa_2 & _T_35[1];	// Rocket.scala:199:33, :345:{56,66}
  wire [63:0] _T_37 = ~mem_reg_xcpt & (mem_ctrl_jalr ^ _mem_npc_misaligned) ? {{24{_T_34[39]}}, _T_34} :
                mem_reg_wdata;	// Rocket.scala:219:13, :343:21, :346:{26,27,41,59}
  wire _T_38 = mem_ctrl_branch | mem_ctrl_jalr | mem_ctrl_jal;	// Rocket.scala:340:25, :341:8, :343:21, :347:50
  assign _T_2 = mem_reg_valid & (_mem_misprediction | mem_reg_flush_pipe);	// Rocket.scala:241:20, :351:{32,54}
  wire _mem_breakpoint = mem_reg_load & bpu_io_xcpt_ld | mem_reg_store & bpu_io_xcpt_st;	// Rocket.scala:215:19, :362:18, :363:19, :377:{38,57,75}
  wire _dcache_kill_mem = _T_20 & io_dmem_replay_next;	// Rocket.scala:392:55
  wire _T_39 = _dcache_kill_mem | _T_1 | mem_reg_xcpt | ~mem_reg_valid;	// Rocket.scala:241:20, :346:27, :395:{68,71}, :420:38
  wire _wb_wxd = wb_reg_valid & wb_ctrl_wxd;	// Rocket.scala:405:13, :414:29
  wire _T_41 = wb_ctrl_div | _wb_dcache_miss;	// Rocket.scala:405:13, :415:35
  wire _replay_wb_common = io_dmem_s2_nack | wb_reg_replay;	// Rocket.scala:416:42
  assign _T_1 = _replay_wb_common | wb_reg_xcpt | csr_io_eret;	// Rocket.scala:187:19, :420:{27,38}
  wire _dmem_resp_fpu = io_dmem_resp_bits_tag[0];	// Rocket.scala:423:45
  wire [4:0] _dmem_resp_waddr = io_dmem_resp_bits_tag[5:1];	// Rocket.scala:425:46
  wire _dmem_resp_valid = io_dmem_resp_valid & io_dmem_resp_bits_has_data;	// Rocket.scala:426:44
  wire _dmem_resp_replay = _dmem_resp_valid & io_dmem_resp_bits_replay;	// Rocket.scala:427:42
  wire _T_42 = _dmem_resp_replay & ~_dmem_resp_fpu;	// Rocket.scala:423:23, :442:26
  assign _T_0 = ~_T_42 & ~_wb_wxd;	// Rocket.scala:429:24, :443:23
  wire [4:0] _T_43 = _T_42 ? _dmem_resp_waddr : div_io_resp_bits_tag;	// Rocket.scala:268:19, :446:14
  wire _T_44 = _T_42 | _T_0 & div_io_resp_valid;	// Decoupled.scala:30:37, Rocket.scala:268:19, :443:23, :447:12
  wire _T_45 = wb_reg_valid & ~_replay_wb_common & ~wb_reg_xcpt;	// Rocket.scala:414:29, :420:27, :450:{34,45,48}
  wire _wb_wen = _T_45 & wb_ctrl_wxd;	// Rocket.scala:405:13, :451:25
  wire _rf_wen = _wb_wen | _T_44;	// Rocket.scala:452:23
  wire [4:0] _rf_waddr = _T_44 ? _T_43 : _wb_waddr;	// Rocket.scala:453:21
  wire [63:0] _rf_wdata = _dmem_resp_valid & ~_dmem_resp_fpu ? io_dmem_resp_bits_data : _T_44 ? div_io_resp_bits_data
                : |wb_ctrl_csr ? csr_io_rw_rdata : wb_reg_wdata;	// Rocket.scala:187:19, :246:23, :268:19, :405:13, :423:23, :454:{21,38}, :455:21, :456:{21,34}
  wire [63:0] _T_46 = _rf_wen & |_rf_waddr & _rf_waddr == ibuf_io_inst_0_bits_inst_rs1 ? _rf_wdata : mem_R0_data;	// Rocket.scala:165:20, :682:23, :694:16, :697:{20,39}
  wire [63:0] _T_47 = _rf_wen & |_rf_waddr & _rf_waddr == ibuf_io_inst_0_bits_inst_rs2 ? _rf_wdata : mem_R1_data;	// Rocket.scala:165:20, :682:23, :694:16, :697:{20,39}
  wire [25:0] _wb_reg_wdata_63to38 = wb_reg_wdata[63:38];	// Rocket.scala:246:23, :653:16
  wire [1:0] _wb_reg_wdata_39to38 = wb_reg_wdata[39:38];	// Rocket.scala:246:23, :654:15
  wire _T_48 = _T_11 & |ibuf_io_inst_0_bits_inst_rs1;	// Rocket.scala:165:20, :479:42, :689:45
  wire _T_49 = _T_9 & |ibuf_io_inst_0_bits_inst_rs2;	// Rocket.scala:165:20, :243:82, :480:42
  wire _T_50 = _T_14 & |ibuf_io_inst_0_bits_inst_rd;	// Rocket.scala:165:20, :481:{42,55}
  wire [31:0] _T_52 = {_T_5_51[31:1], 1'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:669:{35,40}
  wire [31:0] _T_53 = {27'h0, ibuf_io_inst_0_bits_inst_rs1};	// Rocket.scala:165:20, :187:19, :665:35
  wire [31:0] _T_54 = _T_52 >> _T_53;	// Rocket.scala:665:35
  wire [31:0] _T_55 = {27'h0, ibuf_io_inst_0_bits_inst_rs2};	// Rocket.scala:165:20, :187:19, :665:35
  wire [31:0] _T_56 = _T_52 >> _T_55;	// Rocket.scala:665:35
  wire [31:0] _T_57 = {27'h0, ibuf_io_inst_0_bits_inst_rd};	// Rocket.scala:165:20, :187:19, :665:35
  wire [31:0] _T_58 = _T_52 >> _T_57;	// Rocket.scala:665:35
  wire _T_59 = ibuf_io_inst_0_bits_inst_rs1 == _ex_waddr;	// Rocket.scala:165:20, :494:70
  wire _T_60 = ibuf_io_inst_0_bits_inst_rs2 == _ex_waddr;	// Rocket.scala:165:20, :494:70
  wire _T_61 = ibuf_io_inst_0_bits_inst_rd == _ex_waddr;	// Rocket.scala:165:20, :494:70
  wire _T_62 = ibuf_io_inst_0_bits_inst_rs1 == _mem_waddr;	// Rocket.scala:165:20, :503:72
  wire _T_63 = ibuf_io_inst_0_bits_inst_rs2 == _mem_waddr;	// Rocket.scala:165:20, :503:72
  wire _T_64 = ibuf_io_inst_0_bits_inst_rd == _mem_waddr;	// Rocket.scala:165:20, :503:72
  wire _data_hazard_mem = mem_ctrl_wxd & (_T_48 & _T_62 | _T_49 & _T_63 | _T_50 & _T_64);	// Rocket.scala:241:20, :503:38, :648:{27,50}
  wire _T_65 = ibuf_io_inst_0_bits_inst_rs1 == _wb_waddr;	// Rocket.scala:165:20, :509:70
  wire _T_66 = ibuf_io_inst_0_bits_inst_rs2 == _wb_waddr;	// Rocket.scala:165:20, :509:70
  wire _T_67 = ibuf_io_inst_0_bits_inst_rd == _wb_waddr;	// Rocket.scala:165:20, :509:70
  wire [31:0] _T_69 = _T_6_68 >> _T_53;	// Rocket.scala:663:60, :665:35
  wire [31:0] _T_70 = _T_6_68 >> _T_55;	// Rocket.scala:663:60, :665:35
  wire [31:0] _T_71 = _T_6_68 >> ibuf_io_inst_0_bits_inst_rs3;	// Rocket.scala:165:20, :663:60, :665:35
  wire [31:0] _T_72 = _T_6_68 >> _T_57;	// Rocket.scala:663:60, :665:35
  wire _T_73 = ex_reg_valid & (ex_ctrl_wxd & (_T_48 & _T_59 | _T_49 & _T_60 | _T_50 & _T_61) &
                (|ex_ctrl_csr | ex_ctrl_jalr | ex_ctrl_mem | ex_ctrl_div | ex_ctrl_fp) | ex_ctrl_wfd &
                (io_fpu_dec_ren1 & _T_59 | io_fpu_dec_ren2 & _T_60 | io_fpu_dec_ren3 &
                ibuf_io_inst_0_bits_inst_rs3 == _ex_waddr | io_fpu_dec_wen & _T_61)) | mem_reg_valid &
                (_data_hazard_mem & (|mem_ctrl_csr | mem_ctrl_mem & mem_reg_slow_bypass | mem_ctrl_div |
                mem_ctrl_fp) | mem_ctrl_wfd & (io_fpu_dec_ren1 & _T_62 | io_fpu_dec_ren2 & _T_63 |
                io_fpu_dec_ren3 & ibuf_io_inst_0_bits_inst_rs3 == _mem_waddr | io_fpu_dec_wen & _T_64)) |
                wb_reg_valid & (wb_ctrl_wxd & (_T_48 & _T_65 | _T_49 & _T_66 | _T_50 & _T_67) & _T_41 |
                wb_ctrl_wfd & (io_fpu_dec_ren1 & _T_65 | io_fpu_dec_ren2 & _T_66 | io_fpu_dec_ren3 &
                ibuf_io_inst_0_bits_inst_rs3 == _wb_waddr | io_fpu_dec_wen & _T_67)) | _T_48 & _T_54[0] |
                _T_49 & _T_56[0] | _T_50 & _T_58[0] | _T_8 & (_id_csr_en & ~io_fpu_fcsr_rdy |
                io_fpu_dec_ren1 & _T_69[0] | io_fpu_dec_ren2 & _T_70[0] | io_fpu_dec_ren3 & _T_71[0] |
                io_fpu_dec_wen & _T_72[0]) | _T_12 & dcache_blocked | _T_13 & (~(div_io_req_ready |
                div_io_resp_valid & ~_wb_wxd) | _T_29) | _id_mem_busy & (_T_18 &
                ibuf_io_inst_0_bits_inst_bits[26] | _T_17 | id_reg_fence & _T_12) | csr_io_csr_stall;	// Rocket.scala:165:20, :187:19, :204:29, :211:49, :213:{17,33,65,81}, :240:19, :241:{20,39}, :268:19, :269:36, :285:13, :287:17, :360:14, :367:25, :405:13, :414:29, :429:24, :493:{38,94}, :495:{39,76}, :496:{35,54,74}, :502:{40,66,100}, :504:{41,78}, :505:{37,57,78}, :510:{39,76}, :511:{35,54,71}, :519:{15,18,35}, :523:62, :529:16, :530:17, :532:{17,21,40,62,75}, :533:17, :648:{27,50}, :665:35
  wire _T_74 = ~ibuf_io_inst_0_valid | ibuf_io_inst_0_bits_replay | _take_pc_mem_wb | _T_73 |
                csr_io_interrupt;	// Rocket.scala:165:20, :187:19, :535:{17,104}
  wire [39:0] _T_75 = wb_reg_xcpt | csr_io_eret ? csr_io_evec : _replay_wb_common ? wb_reg_pc : _T_36;	// Rocket.scala:187:19, :411:15, :420:27, :540:{8,17}, :541:8
  wire _T_76 = mem_reg_valid & ~_T_1;	// Rocket.scala:241:20, :401:34, :420:38, :549:85
  wire _T_77 = mem_ctrl_jal | mem_ctrl_jalr;	// Rocket.scala:341:8, :343:21, :551:50
  wire _T_78 = mem_ctrl_jalr & {mem_reg_inst[19:18], mem_reg_inst[16:15]} == 4'h1;	// Decode.scala:13:121, Rocket.scala:236:31, :343:21, :552:{53,76}
  wire [38:0] _T_79 = mem_reg_pc[38:0] + {37'h0, ~mem_reg_rvc, 1'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:339:41, :342:8, :554:{69,74}
  wire [38:0] _T_80 = ~_T_79 | 39'h3;	// Rocket.scala:555:{35,66}
  assign _T = ex_reg_valid & ex_ctrl_mem;	// Rocket.scala:240:19, :285:13, :581:41
  wire [25:0] _ex_rs_0_63to38 = _ex_rs_0[63:38];	// Rocket.scala:653:16
  wire [1:0] _alu_io_adder_out_39to38 = alu_io_adder_out[39:38];	// Rocket.scala:261:19, :654:15
  wire _T_81 = _T_39 | _mem_breakpoint;	// Rocket.scala:591:35
  `ifndef SYNTHESIS	// Rocket.scala:115:20
    `ifdef RANDOMIZE_REG_INIT	// Rocket.scala:115:20
      reg [31:0] _RANDOM;	// Rocket.scala:115:20
      reg [31:0] _RANDOM_11;	// Rocket.scala:115:20
      reg [31:0] _RANDOM_12;	// Rocket.scala:116:21
      reg [31:0] _RANDOM_13;	// Rocket.scala:117:20
      reg [31:0] _RANDOM_14;	// Rocket.scala:123:35
      reg [31:0] _RANDOM_15;	// Rocket.scala:123:35
      reg [31:0] _RANDOM_16;	// Rocket.scala:127:35
      reg [31:0] _RANDOM_17;	// Rocket.scala:127:35
      reg [31:0] _RANDOM_18;	// Rocket.scala:129:22
      reg [31:0] _RANDOM_19;	// Rocket.scala:129:22
      reg [31:0] _RANDOM_20;	// Rocket.scala:130:24
      reg [31:0] _RANDOM_21;	// Rocket.scala:136:36
      reg [31:0] _RANDOM_22;	// Rocket.scala:136:36
      reg [31:0] _RANDOM_23;	// Rocket.scala:140:36
      reg [31:0] _RANDOM_24;	// Rocket.scala:140:36
      reg [31:0] _RANDOM_25;	// Rocket.scala:144:23
      reg [31:0] _RANDOM_26;	// Rocket.scala:145:25
      reg [31:0] _RANDOM_27;	// Rocket.scala:146:26
      reg [31:0] _RANDOM_28;	// Rocket.scala:146:26
      reg [31:0] _RANDOM_29;	// Rocket.scala:147:24
      reg [31:0] _RANDOM_30;	// Rocket.scala:147:24
      reg [31:0] _RANDOM_31;	// Rocket.scala:153:35
      reg [31:0] _RANDOM_32;	// Rocket.scala:153:35
      reg [31:0] _RANDOM_33;	// Rocket.scala:154:22
      reg [31:0] _RANDOM_34;	// Rocket.scala:155:24
      reg [31:0] _RANDOM_35;	// Rocket.scala:156:25
      reg [31:0] _RANDOM_36;	// Rocket.scala:156:25
      reg [31:0] _RANDOM_37;	// Rocket.scala:248:26
      reg [31:0] _RANDOM_38;	// Rocket.scala:249:26
      reg [31:0] _RANDOM_39;	// Rocket.scala:249:26
      reg [31:0] _RANDOM_40;	// Rocket.scala:249:26
      reg [31:0] _RANDOM_41;	// Rocket.scala:668:25
      reg [31:0] _RANDOM_42;	// Rocket.scala:668:25
      reg [31:0] _RANDOM_43;	// Rocket.scala:522:27
      reg [31:0] _RANDOM_44;	// Rocket.scala:639:42
      reg [31:0] _RANDOM_45;	// Rocket.scala:639:42
      reg [31:0] _RANDOM_46;	// Rocket.scala:639:33
      reg [31:0] _RANDOM_47;	// Rocket.scala:639:33
      reg [31:0] _RANDOM_48;	// Rocket.scala:640:42
      reg [31:0] _RANDOM_49;	// Rocket.scala:640:42
      reg [31:0] _RANDOM_50;	// Rocket.scala:640:33
      reg [31:0] _RANDOM_51;	// Rocket.scala:640:33

    `endif
    initial begin	// Rocket.scala:115:20
      `INIT_RANDOM_PROLOG_	// Rocket.scala:115:20
      `ifdef RANDOMIZE_REG_INIT	// Rocket.scala:115:20
        _RANDOM = `RANDOM;	// Rocket.scala:115:20
        ex_ctrl_legal = _RANDOM[0];	// Rocket.scala:115:20
        ex_ctrl_fp = _RANDOM[1];	// Rocket.scala:115:20
        ex_ctrl_branch = _RANDOM[2];	// Rocket.scala:115:20
        ex_ctrl_jal = _RANDOM[3];	// Rocket.scala:115:20
        ex_ctrl_jalr = _RANDOM[4];	// Rocket.scala:115:20
        ex_ctrl_rxs2 = _RANDOM[5];	// Rocket.scala:115:20
        ex_ctrl_rxs1 = _RANDOM[6];	// Rocket.scala:115:20
        ex_ctrl_sel_alu2 = _RANDOM[8:7];	// Rocket.scala:115:20
        ex_ctrl_sel_alu1 = _RANDOM[10:9];	// Rocket.scala:115:20
        ex_ctrl_sel_imm = _RANDOM[13:11];	// Rocket.scala:115:20
        ex_ctrl_alu_dw = _RANDOM[14];	// Rocket.scala:115:20
        ex_ctrl_alu_fn = _RANDOM[18:15];	// Rocket.scala:115:20
        ex_ctrl_mem = _RANDOM[19];	// Rocket.scala:115:20
        ex_ctrl_mem_cmd = _RANDOM[24:20];	// Rocket.scala:115:20
        ex_ctrl_mem_type = _RANDOM[27:25];	// Rocket.scala:115:20
        ex_ctrl_rfs1 = _RANDOM[28];	// Rocket.scala:115:20
        ex_ctrl_rfs2 = _RANDOM[29];	// Rocket.scala:115:20
        ex_ctrl_rfs3 = _RANDOM[30];	// Rocket.scala:115:20
        ex_ctrl_wfd = _RANDOM[31];	// Rocket.scala:115:20
        _RANDOM_11 = `RANDOM;	// Rocket.scala:115:20
        ex_ctrl_div = _RANDOM_11[0];	// Rocket.scala:115:20
        ex_ctrl_wxd = _RANDOM_11[1];	// Rocket.scala:115:20
        ex_ctrl_csr = _RANDOM_11[4:2];	// Rocket.scala:115:20
        ex_ctrl_fence_i = _RANDOM_11[5];	// Rocket.scala:115:20
        ex_ctrl_fence = _RANDOM_11[6];	// Rocket.scala:115:20
        ex_ctrl_amo = _RANDOM_11[7];	// Rocket.scala:115:20
        ex_ctrl_dp = _RANDOM_11[8];	// Rocket.scala:115:20
        mem_ctrl_legal = _RANDOM_11[9];	// Rocket.scala:116:21
        mem_ctrl_fp = _RANDOM_11[10];	// Rocket.scala:116:21
        mem_ctrl_branch = _RANDOM_11[11];	// Rocket.scala:116:21
        mem_ctrl_jal = _RANDOM_11[12];	// Rocket.scala:116:21
        mem_ctrl_jalr = _RANDOM_11[13];	// Rocket.scala:116:21
        mem_ctrl_rxs2 = _RANDOM_11[14];	// Rocket.scala:116:21
        mem_ctrl_rxs1 = _RANDOM_11[15];	// Rocket.scala:116:21
        mem_ctrl_sel_alu2 = _RANDOM_11[17:16];	// Rocket.scala:116:21
        mem_ctrl_sel_alu1 = _RANDOM_11[19:18];	// Rocket.scala:116:21
        mem_ctrl_sel_imm = _RANDOM_11[22:20];	// Rocket.scala:116:21
        mem_ctrl_alu_dw = _RANDOM_11[23];	// Rocket.scala:116:21
        mem_ctrl_alu_fn = _RANDOM_11[27:24];	// Rocket.scala:116:21
        mem_ctrl_mem = _RANDOM_11[28];	// Rocket.scala:116:21
        _RANDOM_12 = `RANDOM;	// Rocket.scala:116:21
        mem_ctrl_mem_cmd = {_RANDOM_12[1:0], _RANDOM_11[31:29]};	// Rocket.scala:116:21
        mem_ctrl_mem_type = _RANDOM_12[4:2];	// Rocket.scala:116:21
        mem_ctrl_rfs1 = _RANDOM_12[5];	// Rocket.scala:116:21
        mem_ctrl_rfs2 = _RANDOM_12[6];	// Rocket.scala:116:21
        mem_ctrl_rfs3 = _RANDOM_12[7];	// Rocket.scala:116:21
        mem_ctrl_wfd = _RANDOM_12[8];	// Rocket.scala:116:21
        mem_ctrl_div = _RANDOM_12[9];	// Rocket.scala:116:21
        mem_ctrl_wxd = _RANDOM_12[10];	// Rocket.scala:116:21
        mem_ctrl_csr = _RANDOM_12[13:11];	// Rocket.scala:116:21
        mem_ctrl_fence_i = _RANDOM_12[14];	// Rocket.scala:116:21
        mem_ctrl_fence = _RANDOM_12[15];	// Rocket.scala:116:21
        mem_ctrl_amo = _RANDOM_12[16];	// Rocket.scala:116:21
        mem_ctrl_dp = _RANDOM_12[17];	// Rocket.scala:116:21
        wb_ctrl_legal = _RANDOM_12[18];	// Rocket.scala:117:20
        wb_ctrl_fp = _RANDOM_12[19];	// Rocket.scala:117:20
        wb_ctrl_branch = _RANDOM_12[20];	// Rocket.scala:117:20
        wb_ctrl_jal = _RANDOM_12[21];	// Rocket.scala:117:20
        wb_ctrl_jalr = _RANDOM_12[22];	// Rocket.scala:117:20
        wb_ctrl_rxs2 = _RANDOM_12[23];	// Rocket.scala:117:20
        wb_ctrl_rxs1 = _RANDOM_12[24];	// Rocket.scala:117:20
        wb_ctrl_sel_alu2 = _RANDOM_12[26:25];	// Rocket.scala:117:20
        wb_ctrl_sel_alu1 = _RANDOM_12[28:27];	// Rocket.scala:117:20
        wb_ctrl_sel_imm = _RANDOM_12[31:29];	// Rocket.scala:117:20
        _RANDOM_13 = `RANDOM;	// Rocket.scala:117:20
        wb_ctrl_alu_dw = _RANDOM_13[0];	// Rocket.scala:117:20
        wb_ctrl_alu_fn = _RANDOM_13[4:1];	// Rocket.scala:117:20
        wb_ctrl_mem = _RANDOM_13[5];	// Rocket.scala:117:20
        wb_ctrl_mem_cmd = _RANDOM_13[10:6];	// Rocket.scala:117:20
        wb_ctrl_mem_type = _RANDOM_13[13:11];	// Rocket.scala:117:20
        wb_ctrl_rfs1 = _RANDOM_13[14];	// Rocket.scala:117:20
        wb_ctrl_rfs2 = _RANDOM_13[15];	// Rocket.scala:117:20
        wb_ctrl_rfs3 = _RANDOM_13[16];	// Rocket.scala:117:20
        wb_ctrl_wfd = _RANDOM_13[17];	// Rocket.scala:117:20
        wb_ctrl_div = _RANDOM_13[18];	// Rocket.scala:117:20
        wb_ctrl_wxd = _RANDOM_13[19];	// Rocket.scala:117:20
        wb_ctrl_csr = _RANDOM_13[22:20];	// Rocket.scala:117:20
        wb_ctrl_fence_i = _RANDOM_13[23];	// Rocket.scala:117:20
        wb_ctrl_fence = _RANDOM_13[24];	// Rocket.scala:117:20
        wb_ctrl_amo = _RANDOM_13[25];	// Rocket.scala:117:20
        wb_ctrl_dp = _RANDOM_13[26];	// Rocket.scala:117:20
        ex_reg_xcpt_interrupt = _RANDOM_13[27];	// Rocket.scala:119:35
        ex_reg_valid = _RANDOM_13[28];	// Rocket.scala:120:35
        ex_reg_rvc = _RANDOM_13[29];	// Rocket.scala:121:35
        ex_reg_btb_hit = _RANDOM_13[30];	// Rocket.scala:122:35
        ex_reg_btb_resp_taken = _RANDOM_13[31];	// Rocket.scala:123:35
        _RANDOM_14 = `RANDOM;	// Rocket.scala:123:35
        ex_reg_btb_resp_mask = _RANDOM_14[1:0];	// Rocket.scala:123:35
        ex_reg_btb_resp_bridx = _RANDOM_14[2];	// Rocket.scala:123:35
        _RANDOM_15 = `RANDOM;	// Rocket.scala:123:35
        ex_reg_btb_resp_target = {_RANDOM_15[9:0], _RANDOM_14[31:3]};	// Rocket.scala:123:35
        ex_reg_btb_resp_entry = _RANDOM_15[15:10];	// Rocket.scala:123:35
        ex_reg_btb_resp_bht_history = _RANDOM_15[22:16];	// Rocket.scala:123:35
        ex_reg_btb_resp_bht_value = _RANDOM_15[24:23];	// Rocket.scala:123:35
        ex_reg_xcpt = _RANDOM_15[25];	// Rocket.scala:124:35
        ex_reg_flush_pipe = _RANDOM_15[26];	// Rocket.scala:125:35
        ex_reg_load_use = _RANDOM_15[27];	// Rocket.scala:126:35
        _RANDOM_16 = `RANDOM;	// Rocket.scala:127:35
        _RANDOM_17 = `RANDOM;	// Rocket.scala:127:35
        ex_cause = {_RANDOM_17[27:0], _RANDOM_16, _RANDOM_15[31:28]};	// Rocket.scala:127:35
        ex_reg_replay = _RANDOM_17[28];	// Rocket.scala:128:26
        _RANDOM_18 = `RANDOM;	// Rocket.scala:129:22
        _RANDOM_19 = `RANDOM;	// Rocket.scala:129:22
        ex_reg_pc = {_RANDOM_19[4:0], _RANDOM_18, _RANDOM_17[31:29]};	// Rocket.scala:129:22
        _RANDOM_20 = `RANDOM;	// Rocket.scala:130:24
        ex_reg_inst = {_RANDOM_20[4:0], _RANDOM_19[31:5]};	// Rocket.scala:130:24
        mem_reg_xcpt_interrupt = _RANDOM_20[5];	// Rocket.scala:132:36
        mem_reg_valid = _RANDOM_20[6];	// Rocket.scala:133:36
        mem_reg_rvc = _RANDOM_20[7];	// Rocket.scala:134:36
        mem_reg_btb_hit = _RANDOM_20[8];	// Rocket.scala:135:36
        mem_reg_btb_resp_taken = _RANDOM_20[9];	// Rocket.scala:136:36
        mem_reg_btb_resp_mask = _RANDOM_20[11:10];	// Rocket.scala:136:36
        mem_reg_btb_resp_bridx = _RANDOM_20[12];	// Rocket.scala:136:36
        _RANDOM_21 = `RANDOM;	// Rocket.scala:136:36
        mem_reg_btb_resp_target = {_RANDOM_21[19:0], _RANDOM_20[31:13]};	// Rocket.scala:136:36
        mem_reg_btb_resp_entry = _RANDOM_21[25:20];	// Rocket.scala:136:36
        _RANDOM_22 = `RANDOM;	// Rocket.scala:136:36
        mem_reg_btb_resp_bht_history = {_RANDOM_22[0], _RANDOM_21[31:26]};	// Rocket.scala:136:36
        mem_reg_btb_resp_bht_value = _RANDOM_22[2:1];	// Rocket.scala:136:36
        mem_reg_xcpt = _RANDOM_22[3];	// Rocket.scala:137:36
        mem_reg_replay = _RANDOM_22[4];	// Rocket.scala:138:36
        mem_reg_flush_pipe = _RANDOM_22[5];	// Rocket.scala:139:36
        _RANDOM_23 = `RANDOM;	// Rocket.scala:140:36
        _RANDOM_24 = `RANDOM;	// Rocket.scala:140:36
        mem_reg_cause = {_RANDOM_24[5:0], _RANDOM_23, _RANDOM_22[31:6]};	// Rocket.scala:140:36
        mem_reg_slow_bypass = _RANDOM_24[6];	// Rocket.scala:141:36
        mem_reg_load = _RANDOM_24[7];	// Rocket.scala:142:36
        mem_reg_store = _RANDOM_24[8];	// Rocket.scala:143:36
        _RANDOM_25 = `RANDOM;	// Rocket.scala:144:23
        mem_reg_pc = {_RANDOM_25[16:0], _RANDOM_24[31:9]};	// Rocket.scala:144:23
        _RANDOM_26 = `RANDOM;	// Rocket.scala:145:25
        mem_reg_inst = {_RANDOM_26[16:0], _RANDOM_25[31:17]};	// Rocket.scala:145:25
        _RANDOM_27 = `RANDOM;	// Rocket.scala:146:26
        _RANDOM_28 = `RANDOM;	// Rocket.scala:146:26
        mem_reg_wdata = {_RANDOM_28[16:0], _RANDOM_27, _RANDOM_26[31:17]};	// Rocket.scala:146:26
        _RANDOM_29 = `RANDOM;	// Rocket.scala:147:24
        _RANDOM_30 = `RANDOM;	// Rocket.scala:147:24
        mem_reg_rs2 = {_RANDOM_30[16:0], _RANDOM_29, _RANDOM_28[31:17]};	// Rocket.scala:147:24
        wb_reg_valid = _RANDOM_30[17];	// Rocket.scala:150:35
        wb_reg_xcpt = _RANDOM_30[18];	// Rocket.scala:151:35
        wb_reg_replay = _RANDOM_30[19];	// Rocket.scala:152:35
        _RANDOM_31 = `RANDOM;	// Rocket.scala:153:35
        _RANDOM_32 = `RANDOM;	// Rocket.scala:153:35
        wb_reg_cause = {_RANDOM_32[19:0], _RANDOM_31, _RANDOM_30[31:20]};	// Rocket.scala:153:35
        _RANDOM_33 = `RANDOM;	// Rocket.scala:154:22
        wb_reg_pc = {_RANDOM_33[27:0], _RANDOM_32[31:20]};	// Rocket.scala:154:22
        _RANDOM_34 = `RANDOM;	// Rocket.scala:155:24
        wb_reg_inst = {_RANDOM_34[27:0], _RANDOM_33[31:28]};	// Rocket.scala:155:24
        _RANDOM_35 = `RANDOM;	// Rocket.scala:156:25
        _RANDOM_36 = `RANDOM;	// Rocket.scala:156:25
        wb_reg_wdata = {_RANDOM_36[27:0], _RANDOM_35, _RANDOM_34[31:28]};	// Rocket.scala:156:25
        id_reg_fence = _RANDOM_36[28];	// Rocket.scala:178:25
        ex_reg_rs_bypass_0 = _RANDOM_36[29];	// Rocket.scala:247:29
        ex_reg_rs_bypass_1 = _RANDOM_36[30];	// Rocket.scala:247:29
        _RANDOM_37 = `RANDOM;	// Rocket.scala:248:26
        ex_reg_rs_lsb_0 = {_RANDOM_37[0], _RANDOM_36[31]};	// Rocket.scala:248:26
        ex_reg_rs_lsb_1 = _RANDOM_37[2:1];	// Rocket.scala:248:26
        _RANDOM_38 = `RANDOM;	// Rocket.scala:249:26
        _RANDOM_39 = `RANDOM;	// Rocket.scala:249:26
        ex_reg_rs_msb_0 = {_RANDOM_39[0], _RANDOM_38, _RANDOM_37[31:3]};	// Rocket.scala:249:26
        _RANDOM_40 = `RANDOM;	// Rocket.scala:249:26
        ex_reg_rs_msb_1 = {_RANDOM_40[30:0], _RANDOM_39[31:1]};	// Rocket.scala:249:26
        _T_4 = _RANDOM_40[31];	// Rocket.scala:396:37
        _RANDOM_41 = `RANDOM;	// Rocket.scala:668:25
        _T_5 = _RANDOM_41;	// Rocket.scala:668:25
        _RANDOM_42 = `RANDOM;	// Rocket.scala:668:25
        _T_6 = _RANDOM_42;	// Rocket.scala:668:25
        _RANDOM_43 = `RANDOM;	// Rocket.scala:522:27
        dcache_blocked = _RANDOM_43[0];	// Rocket.scala:522:27
        rocc_blocked = _RANDOM_43[1];	// Rocket.scala:524:25
        _RANDOM_44 = `RANDOM;	// Rocket.scala:639:42
        _RANDOM_45 = `RANDOM;	// Rocket.scala:639:42
        _T_7 = {_RANDOM_45[1:0], _RANDOM_44, _RANDOM_43[31:2]};	// Rocket.scala:639:42
        _RANDOM_46 = `RANDOM;	// Rocket.scala:639:33
        _RANDOM_47 = `RANDOM;	// Rocket.scala:639:33
        _T_8 = {_RANDOM_47[1:0], _RANDOM_46, _RANDOM_45[31:2]};	// Rocket.scala:639:33
        _RANDOM_48 = `RANDOM;	// Rocket.scala:640:42
        _RANDOM_49 = `RANDOM;	// Rocket.scala:640:42
        _T_9 = {_RANDOM_49[1:0], _RANDOM_48, _RANDOM_47[31:2]};	// Rocket.scala:640:42
        _RANDOM_50 = `RANDOM;	// Rocket.scala:640:33
        _RANDOM_51 = `RANDOM;	// Rocket.scala:640:33
        _T_10 = {_RANDOM_51[1:0], _RANDOM_50, _RANDOM_49[31:2]};	// Rocket.scala:640:33
      `endif
    end // initial
  `endif
      wire [31:0] _T_86 = _T_52 & ~(_T_44 ? 32'h1 << _T_43 : 32'h0);	// Decode.scala:13:121, Rocket.scala:664:{62,64}, :672:{49,62}
      wire _T_87 = _T_41 & _wb_wen;	// Rocket.scala:490:28
      wire [31:0] _T_88 = 32'h1 << _wb_waddr;	// Rocket.scala:672:62
      wire _T_89 = (_wb_dcache_miss & wb_ctrl_wfd | io_fpu_sboard_set) & _T_45;	// Rocket.scala:405:13, :515:{35,50,72}
      wire [31:0] _T_90 = _T_89 ? _T_88 : 32'h0;	// Decode.scala:13:121, Rocket.scala:672:49
      wire _T_91 = _dmem_resp_replay & _dmem_resp_fpu;	// Rocket.scala:516:38
      wire [31:0] _T_92 = (_T_6_68 | _T_90) & ~(_T_91 ? 32'h1 << _dmem_resp_waddr : 32'h0);	// Decode.scala:13:121, Rocket.scala:663:60, :664:{62,64}, :672:{49,62}
      wire _T_93 = _T_89 | _T_91;	// Rocket.scala:675:17
      wire _T_94 = ibuf_io_inst_0_bits_inst_bits[4:3] == 2'h1;	// Decode.scala:13:{65,121}, Mux.scala:31:69, Rocket.scala:165:20
      wire _T_95 = {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[3]} == 2'h1;	// Decode.scala:13:{65,121}, Mux.scala:31:69, Rocket.scala:165:20
      wire _T_96 = {ibuf_io_inst_0_bits_inst_bits[30], ibuf_io_inst_0_bits_inst_bits[13:12],
                ibuf_io_inst_0_bits_inst_bits[5:4], ibuf_io_inst_0_bits_inst_bits[2]} == 6'h26;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
      wire _T_97 = {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:3]} == 4'hC;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
      wire _T_98 = ex_ctrl_mem_cmd[3] | ex_ctrl_mem_cmd == 5'h4;	// Consts.scala:33:{29,33,40}, Rocket.scala:285:13
      wire _T_99 = _T_3 | {ibuf_io_inst_0_bits_inst_bits[12], ibuf_io_inst_0_bits_inst_bits[6:5],
                ibuf_io_inst_0_bits_inst_bits[3:0]} == 7'h3 | {ibuf_io_inst_0_bits_inst_bits[14:13],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 9'hF | {ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:4], ibuf_io_inst_0_bits_inst_bits[2:0]} == 9'hB |
                {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:0]} == 6'h17 |
                {ibuf_io_inst_0_bits_inst_bits[31:26], ibuf_io_inst_0_bits_inst_bits[6:0]} == 13'h33 |
                {ibuf_io_inst_0_bits_inst_bits[31], ibuf_io_inst_0_bits_inst_bits[29:25],
                ibuf_io_inst_0_bits_inst_bits[14:12], ibuf_io_inst_0_bits_inst_bits[6:4],
                ibuf_io_inst_0_bits_inst_bits[2:0]} == 15'h1B | {ibuf_io_inst_0_bits_inst_bits[26],
                ibuf_io_inst_0_bits_inst_bits[6:4], ibuf_io_inst_0_bits_inst_bits[1:0]} == 6'h13 |
                {ibuf_io_inst_0_bits_inst_bits[31:29], ibuf_io_inst_0_bits_inst_bits[26],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 11'h53 | {ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:3], ibuf_io_inst_0_bits_inst_bits[1:0]} == 9'h33 |
                ibuf_io_inst_0_bits_inst_bits[6:0] == 7'h6F | {ibuf_io_inst_0_bits_inst_bits[31:21],
                ibuf_io_inst_0_bits_inst_bits[19:0]} == 31'h73 | {ibuf_io_inst_0_bits_inst_bits[31:26],
                ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:0]} == 14'h53 | {ibuf_io_inst_0_bits_inst_bits[31:25],
                ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:0]} == 15'h5B | _T_4 | {ibuf_io_inst_0_bits_inst_bits[13],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 8'h93 | _T_5 | {ibuf_io_inst_0_bits_inst_bits[13],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 8'hF3 | {ibuf_io_inst_0_bits_inst_bits[31],
                ibuf_io_inst_0_bits_inst_bits[29:26], ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 15'h293 | {ibuf_io_inst_0_bits_inst_bits[31],
                ibuf_io_inst_0_bits_inst_bits[29:25], ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:0]} == 15'h15B |
                {ibuf_io_inst_0_bits_inst_bits[31], ibuf_io_inst_0_bits_inst_bits[29:25],
                ibuf_io_inst_0_bits_inst_bits[14:12], ibuf_io_inst_0_bits_inst_bits[6:4],
                ibuf_io_inst_0_bits_inst_bits[2:0]} == 15'h15B | {ibuf_io_inst_0_bits_inst_bits[31:25],
                ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6:4],
                ibuf_io_inst_0_bits_inst_bits[2:0]} == 14'hDB | _T_6 | _T_7 |
                {ibuf_io_inst_0_bits_inst_bits[31:30], ibuf_io_inst_0_bits_inst_bits[28:0]} == 31'h10200073
                | ibuf_io_inst_0_bits_inst_bits == 32'h10500073 | {ibuf_io_inst_0_bits_inst_bits[31:25],
                ibuf_io_inst_0_bits_inst_bits[14:0]} == 22'h48073 | {ibuf_io_inst_0_bits_inst_bits[31:28],
                ibuf_io_inst_0_bits_inst_bits[26], ibuf_io_inst_0_bits_inst_bits[14:13],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 14'h853 | {ibuf_io_inst_0_bits_inst_bits[30:26],
                ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6:0]} == 14'h1053 |
                {ibuf_io_inst_0_bits_inst_bits[30:26], ibuf_io_inst_0_bits_inst_bits[14],
                ibuf_io_inst_0_bits_inst_bits[12], ibuf_io_inst_0_bits_inst_bits[6:0]} == 14'h1053 |
                {ibuf_io_inst_0_bits_inst_bits[30:20], ibuf_io_inst_0_bits_inst_bits[6:0]} == 18'h200D3 |
                {ibuf_io_inst_0_bits_inst_bits[30:20], ibuf_io_inst_0_bits_inst_bits[6:0]} == 18'h21053 |
                {ibuf_io_inst_0_bits_inst_bits[31:26], ibuf_io_inst_0_bits_inst_bits[24:20],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 18'h16053 | ibuf_io_inst_0_bits_inst_bits ==
                32'h7B200073 | {ibuf_io_inst_0_bits_inst_bits[31:29], ibuf_io_inst_0_bits_inst_bits[27:26],
                ibuf_io_inst_0_bits_inst_bits[24:22], ibuf_io_inst_0_bits_inst_bits[6:0]} == 15'h6053 |
                {ibuf_io_inst_0_bits_inst_bits[31:26], ibuf_io_inst_0_bits_inst_bits[24:20],
                ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6:0]} == 20'hE0053 |
                {ibuf_io_inst_0_bits_inst_bits[31:29], ibuf_io_inst_0_bits_inst_bits[27:26],
                ibuf_io_inst_0_bits_inst_bits[24:20], ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:0]} == 20'hE0053 | {ibuf_io_inst_0_bits_inst_bits[14:13],
                ibuf_io_inst_0_bits_inst_bits[5:0]} == 8'h23 | {ibuf_io_inst_0_bits_inst_bits[13:12],
                ibuf_io_inst_0_bits_inst_bits[6:5], ibuf_io_inst_0_bits_inst_bits[3:0]} == 8'h73 |
                {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6:0]} == 8'hE3 |
                {ibuf_io_inst_0_bits_inst_bits[31:26], ibuf_io_inst_0_bits_inst_bits[14:12],
                ibuf_io_inst_0_bits_inst_bits[6:4], ibuf_io_inst_0_bits_inst_bits[2:0]} == 15'h1B;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20
      wire _T_100 = {ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[5:2]} == 5'h9;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
      wire [1:0] _T_101 = {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[3]};	// Decode.scala:13:65, Rocket.scala:165:20
      wire _T_102 = ibuf_io_inst_0_bits_inst_bits[4:3] == 2'h0;	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :292:24
      wire _T_103 = {ibuf_io_inst_0_bits_inst_bits[31], ibuf_io_inst_0_bits_inst_bits[6:5]} == 3'h2;	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :705:26
      wire _T_104 = ibuf_io_inst_0_bits_inst_bits[6:4] == 3'h4;	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :715:22
      wire _T_105 = {ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:3]} == 5'h1;	// Decode.scala:13:{65,121}, Rocket.scala:165:20
      wire _T_106 = {ibuf_io_inst_0_bits_inst_bits[12], ibuf_io_inst_0_bits_inst_bits[6],
                ibuf_io_inst_0_bits_inst_bits[4:2]} == 5'h11 | {ibuf_io_inst_0_bits_inst_bits[25],
                ibuf_io_inst_0_bits_inst_bits[6:5]} == 3'h6 | {ibuf_io_inst_0_bits_inst_bits[31:30],
                ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[6:4]} == 6'h15;	// Consts.scala:35:48, Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20
      wire _id_csr_ren = _T_19 & ~(|ibuf_io_inst_0_bits_inst_rs1);	// Rocket.scala:165:20, :190:54, :689:45
      wire _id_xcpt_if = ibuf_io_inst_0_bits_pf0 | ibuf_io_inst_0_bits_pf1;	// Rocket.scala:165:20, :221:45
      wire _T_107 = csr_io_interrupt | bpu_io_debug_if | bpu_io_xcpt_if | _id_xcpt_if | ~_T_99 | _T_13 &
                ~(csr_io_status_isa[12]) | _T_18 & ~(csr_io_status_isa[0]) | _T_8 &
                (csr_io_decode_fp_illegal | io_fpu_illegal_rm) | _T_106 & ~(csr_io_status_isa[3]) |
                ibuf_io_inst_0_bits_rvc & ~_csr_io_status_isa_2 | _id_csr_en & (csr_io_decode_read_illegal
                | ~_id_csr_ren & csr_io_decode_write_illegal) | _T_15 & csr_io_decode_system_illegal;	// Rocket.scala:165:20, :187:19, :192:54, :194:25, :195:{17,20,38}, :196:{17,20,38}, :197:{16,45}, :198:{16,19,37}, :199:{30,33}, :201:{15,46,61}, :202:20, :215:19, :645:26
      wire _T_108 = ex_reg_valid & ex_ctrl_wxd;	// Rocket.scala:240:19
      wire _T_109 = _T_20 & ~mem_ctrl_mem;	// Rocket.scala:241:{36,39}
      wire _id_bypass_src_0_1 = _T_108 & _ex_waddr == ibuf_io_inst_0_bits_inst_rs1;	// Rocket.scala:165:20, :243:{74,82}
      wire _T_110 = _mem_waddr == ibuf_io_inst_0_bits_inst_rs1;	// Rocket.scala:165:20, :243:82
      wire _id_bypass_src_0_2 = _T_109 & _T_110;	// Rocket.scala:243:74
      wire _id_bypass_src_1_1 = _T_108 & _ex_waddr == ibuf_io_inst_0_bits_inst_rs2;	// Rocket.scala:165:20, :243:{74,82}
      wire _T_111 = _mem_waddr == ibuf_io_inst_0_bits_inst_rs2;	// Rocket.scala:165:20, :243:82
      wire _id_bypass_src_1_2 = _T_109 & _T_111;	// Rocket.scala:243:74
      wire _T_112 = ~_take_pc_mem_wb & ibuf_io_inst_0_valid;	// Rocket.scala:165:20, :277:{20,29}
      wire _T_113 = ~bpu_io_xcpt_if & ~ibuf_io_inst_0_bits_pf0 & ibuf_io_inst_0_bits_pf1;	// Rocket.scala:165:20, :215:19, :293:{13,32,58}
      wire _T_114 = _T_100 & csr_io_status_debug;	// Rocket.scala:187:19, :301:24
      wire _T_115 = ~(|ibuf_io_inst_0_bits_inst_rs1) | _id_bypass_src_0_1 | _id_bypass_src_0_2 | _T_20 & _T_110;	// Rocket.scala:165:20, :243:{74,82}, :307:48, :689:45
      wire _T_116 = _T_11 & ~_T_115;	// Rocket.scala:311:{23,26}
      wire _T_117 = ~(|ibuf_io_inst_0_bits_inst_rs2) | _id_bypass_src_1_1 | _id_bypass_src_1_2 | _T_20 & _T_111;	// Rocket.scala:165:20, :243:{74,82}, :307:48
      wire _T_118 = _T_9 & ~_T_117;	// Rocket.scala:311:{23,26}
      wire _T_119 = ex_ctrl_mem_cmd == 5'h7;	// Decode.scala:13:65, Rocket.scala:285:13, :331:40
      wire _ex_xcpt = ex_reg_xcpt_interrupt | ex_reg_xcpt;	// Rocket.scala:323:51, :334:28
      wire _mem_debug_breakpoint = mem_reg_load & bpu_io_debug_ld | mem_reg_store & bpu_io_debug_st;	// Rocket.scala:215:19, :362:18, :363:19, :378:{44,64,82}
      wire _T_120 = mem_ctrl_mem & io_dmem_xcpt_ma_st;	// Rocket.scala:241:39, :383:19
      wire _T_121 = mem_ctrl_mem & io_dmem_xcpt_ma_ld;	// Rocket.scala:241:39, :384:19
      wire _T_122 = mem_ctrl_mem & io_dmem_xcpt_pf_st;	// Rocket.scala:241:39, :385:19
      wire _T_123 = mem_reg_xcpt_interrupt | mem_reg_xcpt;	// Rocket.scala:337:54, :346:27, :389:29
      wire _mem_xcpt = _T_123 | mem_reg_valid & (_mem_debug_breakpoint | _mem_breakpoint | _mem_npc_misaligned |
                _T_120 | _T_121 | _T_122 | mem_ctrl_mem & io_dmem_xcpt_pf_ld);	// Rocket.scala:241:{20,39}, :386:19, :390:20, :645:26
      wire _T_124 = mem_reg_valid & mem_ctrl_fp & io_fpu_nack_mem;	// Rocket.scala:241:20, :360:14, :393:51
      wire _T_125 = mem_ctrl_mem & _mem_xcpt & ~_T_81 & ~(|{io_dmem_xcpt_ma_ld, io_dmem_xcpt_ma_st,
                io_dmem_xcpt_pf_ld, io_dmem_xcpt_pf_st} | reset);	// Rocket.scala:241:39, :592:37, :593:{11,25,32}
  always @(posedge clock) begin	// Rocket.scala:276:16
    if (reset) begin	// Rocket.scala:178:25
      id_reg_fence <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:178:25
      _T_5_51 <= 32'h0;	// Decode.scala:13:121, Rocket.scala:668:25
      _T_6_68 <= 32'h0;	// Decode.scala:13:121, Rocket.scala:668:25
    end
    else begin	// Rocket.scala:178:25
      id_reg_fence <= _T_105 | _T_18 & ibuf_io_inst_0_bits_inst_bits[25] | id_reg_fence & _id_mem_busy;	// Rocket.scala:165:20, :205:29, :206:52, :211:{16,33,49}
      _T_5_51 <= _T_44 | _T_87 ? _T_86 | (_T_87 ? _T_88 : 32'h0) : _T_44 ? _T_86 : _T_5_51;	// Decode.scala:13:121, Rocket.scala:663:60, :669:35, :672:49, :675:17, :676:23
      _T_6_68 <= _T_93 | io_fpu_sboard_clr ? _T_92 & ~(io_fpu_sboard_clr ? 32'h1 << io_fpu_sboard_clra :
                                                32'h0) : _T_93 ? _T_92 : {32{_T_89}} & _T_90 | _T_6_68;	// Decode.scala:13:121, Rocket.scala:663:60, :664:{62,64}, :672:{49,62}, :675:17, :676:23
    end
    ex_reg_valid <= ~_T_74;	// Rocket.scala:185:34, :276:16
    ex_reg_replay <= _T_112 & ibuf_io_inst_0_bits_replay;	// Rocket.scala:165:20, :277:{17,54}
    ex_reg_xcpt <= ~_T_74 & _T_107;	// Rocket.scala:185:34, :278:{15,30}
    ex_reg_xcpt_interrupt <= _T_112 & csr_io_interrupt;	// Rocket.scala:187:19, :279:{25,62}
    if (_T_107)	// Rocket.scala:280:33
      ex_cause <= csr_io_interrupt ? csr_io_interrupt_cause : {60'h0, bpu_io_debug_if ? 4'hD : {2'h0,
                                                bpu_io_xcpt_if ? 2'h3 : _id_xcpt_if ? 2'h1 : 2'h2}};	// Mux.scala:31:69, Rocket.scala:187:19, :215:19, :280:33, :292:24
    ex_reg_btb_hit <= ibuf_io_inst_0_bits_btb_hit;	// Rocket.scala:165:20, :281:18
    if (ibuf_io_inst_0_bits_btb_hit) begin	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_taken <= ibuf_io_btb_resp_taken;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_mask <= ibuf_io_btb_resp_mask;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_bridx <= ibuf_io_btb_resp_bridx;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_target <= ibuf_io_btb_resp_target;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_entry <= ibuf_io_btb_resp_entry;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_bht_history <= ibuf_io_btb_resp_bht_history;	// Rocket.scala:165:20, :282:57
      ex_reg_btb_resp_bht_value <= ibuf_io_btb_resp_bht_value;	// Rocket.scala:165:20, :282:57
    end
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_legal <= _T_99;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_fp <= _T_8;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_branch <= {ibuf_io_inst_0_bits_inst_bits[6:4], ibuf_io_inst_0_bits_inst_bits[2]} == 4'hC;	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_jal <= &{ibuf_io_inst_0_bits_inst_bits[6:5], ibuf_io_inst_0_bits_inst_bits[3]};	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_jalr <= _T_100;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_rxs2 <= _T_9;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_rxs1 <= _T_11;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_sel_imm <= {ibuf_io_inst_0_bits_inst_bits[5:4] == 2'h0 | {ibuf_io_inst_0_bits_inst_bits[13],
                                                ibuf_io_inst_0_bits_inst_bits[4:2]} == 4'h1 | {ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 2'h2, _T_94 | &{ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]}, _T_94 | {ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 2'h2};	// Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20, :285:13, :292:24
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_mem <= _T_12;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_mem_cmd <= {1'h0, {ibuf_io_inst_0_bits_inst_bits[28:27], ibuf_io_inst_0_bits_inst_bits[13],
                                                ibuf_io_inst_0_bits_inst_bits[3]} == 4'h3, _T_95 | &{ibuf_io_inst_0_bits_inst_bits[27],
                                                ibuf_io_inst_0_bits_inst_bits[3]} | &{ibuf_io_inst_0_bits_inst_bits[28],
                                                ibuf_io_inst_0_bits_inst_bits[3]} | &{ibuf_io_inst_0_bits_inst_bits[31],
                                                ibuf_io_inst_0_bits_inst_bits[3]}, &{ibuf_io_inst_0_bits_inst_bits[28],
                                                ibuf_io_inst_0_bits_inst_bits[13], ibuf_io_inst_0_bits_inst_bits[3]} |
                                                &{ibuf_io_inst_0_bits_inst_bits[30], ibuf_io_inst_0_bits_inst_bits[13],
                                                ibuf_io_inst_0_bits_inst_bits[3]}, _T_95 | {ibuf_io_inst_0_bits_inst_bits[5],
                                                ibuf_io_inst_0_bits_inst_bits[3]} == 2'h2 | &{ibuf_io_inst_0_bits_inst_bits[28:27],
                                                ibuf_io_inst_0_bits_inst_bits[5]} | &{ibuf_io_inst_0_bits_inst_bits[29],
                                                ibuf_io_inst_0_bits_inst_bits[5]}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20, :285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_mem_type <= ibuf_io_inst_0_bits_inst_bits[14:12];	// Cat.scala:30:58, Rocket.scala:165:20, :285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_rfs1 <= _T_103 | {ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[6:5]} == 3'h2 |
                                                _T_104;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20, :285:13, :705:26
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_rfs2 <= ibuf_io_inst_0_bits_inst_bits[6:2] == 5'h9 | {ibuf_io_inst_0_bits_inst_bits[30],
                                                ibuf_io_inst_0_bits_inst_bits[6:5]} == 3'h2 | _T_104 | {ibuf_io_inst_0_bits_inst_bits[31],
                                                ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[6:5]} == 4'h6;	// Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20, :285:13, :705:26
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_rfs3 <= _T_104;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_wfd <= ibuf_io_inst_0_bits_inst_bits[5:2] == 4'h1 | _T_103 | _T_104 |
                                                {ibuf_io_inst_0_bits_inst_bits[28], ibuf_io_inst_0_bits_inst_bits[6:5]} == 3'h6;	// Consts.scala:35:48, Decode.scala:13:{65,121}, :14:30, Rocket.scala:165:20, :285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_div <= _T_13;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_wxd <= _T_14;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_fence <= _T_105;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_amo <= _T_18;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:285:13
      ex_ctrl_dp <= _T_106;	// Rocket.scala:285:13
    if (~_T_74)	// Rocket.scala:287:17
      ex_ctrl_csr <= _id_csr_ren ? 3'h5 : _T_16;	// Rocket.scala:191:19, :287:17, :704:24
    if (~_T_74)	// Rocket.scala:289:22
      ex_ctrl_alu_fn <= _T_107 ? 4'h0 : {{ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[4], ibuf_io_inst_0_bits_inst_bits[2]} == 5'hA | _T_97 | _T_96
                                                | {ibuf_io_inst_0_bits_inst_bits[30], ibuf_io_inst_0_bits_inst_bits[13:12],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 6'h2A, {ibuf_io_inst_0_bits_inst_bits[13],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 4'hA | {ibuf_io_inst_0_bits_inst_bits[30],
                                                ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[4], ibuf_io_inst_0_bits_inst_bits[2]} == 5'hA |
                                                {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[12],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 5'h12 | _T_97, {ibuf_io_inst_0_bits_inst_bits[14],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 4'h4 | {ibuf_io_inst_0_bits_inst_bits[13],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:3]} == 4'hC |
                                                {ibuf_io_inst_0_bits_inst_bits[13:12], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[4], ibuf_io_inst_0_bits_inst_bits[2]} == 5'h1A |
                                                {ibuf_io_inst_0_bits_inst_bits[14:13], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[4], ibuf_io_inst_0_bits_inst_bits[2]} == 5'h1A | _T_96 |
                                                {ibuf_io_inst_0_bits_inst_bits[30], ibuf_io_inst_0_bits_inst_bits[12],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 5'h1A, {ibuf_io_inst_0_bits_inst_bits[13:12],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 5'hA | {ibuf_io_inst_0_bits_inst_bits[12],
                                                ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:3]} == 4'hC |
                                                {ibuf_io_inst_0_bits_inst_bits[14:12], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[2]} == 5'h1C};	// Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30, Mux.scala:46:16, Rocket.scala:165:20, :259:19, :289:22
    if (~_T_74)	// Rocket.scala:290:22
      ex_ctrl_alu_dw <= _T_107 | ~(ibuf_io_inst_0_bits_inst_bits[4]) | ~(ibuf_io_inst_0_bits_inst_bits[3]);	// Decode.scala:13:{65,121}, Rocket.scala:165:20, :290:22
    if (~_T_74)	// Rocket.scala:291:24
      ex_ctrl_sel_alu1 <= _T_107 ? 2'h2 : {{ibuf_io_inst_0_bits_inst_bits[5:4], ibuf_io_inst_0_bits_inst_bits[2]} ==
                                                3'h3 | &_T_101, {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[2]} ==
                                                2'h0 | {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4]} == 2'h0 | _T_10
                                                | {ibuf_io_inst_0_bits_inst_bits[5], ibuf_io_inst_0_bits_inst_bits[2]} == 2'h0 | _T_102};	// Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20, :291:24, :292:24, :704:24
    if (~_T_74)	// Rocket.scala:294:26
      ex_ctrl_sel_alu2 <= _T_107 ? {1'h0, _T_113} : {{ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[3]} == 2'h0 | _T_10 | _T_102 |
                                                {ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[3]} == 2'h2,
                                                {ibuf_io_inst_0_bits_inst_bits[6], ibuf_io_inst_0_bits_inst_bits[4:3]} == 3'h0 |
                                                ~(ibuf_io_inst_0_bits_inst_bits[5]) | ibuf_io_inst_0_bits_inst_bits[3:2] == 2'h1 | &_T_101
                                                | &{ibuf_io_inst_0_bits_inst_bits[14], ibuf_io_inst_0_bits_inst_bits[6],
                                                ibuf_io_inst_0_bits_inst_bits[4]}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Cat.scala:30:58, Decode.scala:13:{65,121}, :14:30, Mux.scala:31:69, Rocket.scala:165:20, :292:24, :294:26, :712:24
    if (~_T_74)	// Rocket.scala:295:20
      ex_reg_rvc <= _T_107 & _T_113 | ibuf_io_inst_0_bits_rvc;	// Rocket.scala:165:20, :295:20
    if (~_T_74)	// Rocket.scala:299:21
      ex_reg_load_use <= mem_reg_valid & _data_hazard_mem & mem_ctrl_mem;	// Rocket.scala:241:{20,39}, :299:21, :506:51
    if (~_T_74)	// Rocket.scala:302:25
      ex_reg_flush_pipe <= _T_114 | _T_17 | _T_15 | _id_csr_en & ~_id_csr_ren & csr_io_decode_write_flush |
                                                csr_io_singleStep;	// Rocket.scala:187:19, :192:{54,66}, :302:25
    if (~_T_74)	// Rocket.scala:303:23
      ex_ctrl_fence_i <= _T_114 | _T_17;	// Rocket.scala:303:23
    if (~_T_74)	// Rocket.scala:309:27
      ex_reg_rs_bypass_0 <= _T_115;	// Rocket.scala:309:27
    if (~_T_74)	// Rocket.scala:312:26
      ex_reg_rs_lsb_0 <= _T_116 ? _T_46[1:0] : |ibuf_io_inst_0_bits_inst_rs1 ? (_id_bypass_src_0_1 ? 2'h1 : {1'h1,
                                                ~_id_bypass_src_0_2}) : 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69, Rocket.scala:165:20, :292:24, :312:{26,37}, :689:45
    if (~(_T_74 | ~_T_116))	// Rocket.scala:313:26
      ex_reg_rs_msb_0 <= _T_46[63:2];	// Rocket.scala:313:{26,38}
    if (~_T_74)	// Rocket.scala:309:27
      ex_reg_rs_bypass_1 <= _T_117;	// Rocket.scala:309:27
    if (~_T_74)	// Rocket.scala:312:26
      ex_reg_rs_lsb_1 <= _T_118 ? _T_47[1:0] : |ibuf_io_inst_0_bits_inst_rs2 ? (_id_bypass_src_1_1 ? 2'h1 : {1'h1,
                                                ~_id_bypass_src_1_2}) : 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69, Rocket.scala:165:20, :243:82, :292:24, :312:{26,37}
    if (~(_T_74 | ~_T_118))	// Rocket.scala:313:26
      ex_reg_rs_msb_1 <= _T_47[63:2];	// Rocket.scala:313:{26,38}
    if (~_T_74 | csr_io_interrupt | ibuf_io_inst_0_bits_replay) begin	// Rocket.scala:165:20, :185:34, :187:19, :317:41, :319:15
      ex_reg_inst <= ibuf_io_inst_0_bits_inst_bits;	// Rocket.scala:165:20, :318:17
      ex_reg_pc <= ibuf_io_pc;	// Rocket.scala:165:20, :319:15
    end
    mem_reg_valid <= ~_T_31;	// Rocket.scala:353:{17,20}
    mem_reg_replay <= ~_take_pc_mem_wb & _replay_ex;	// Rocket.scala:277:20, :354:{18,37}
    mem_reg_xcpt <= ~_T_31 & _ex_xcpt;	// Rocket.scala:353:20, :355:{16,31}
    mem_reg_xcpt_interrupt <= ~_take_pc_mem_wb & ex_reg_xcpt_interrupt;	// Rocket.scala:277:20, :323:51, :356:{26,45}
    if (_ex_xcpt)	// Rocket.scala:357:34
      mem_reg_cause <= ex_cause;	// Rocket.scala:280:33, :357:34
    if (_T_30) begin	// Rocket.scala:364:21
      mem_ctrl_legal <= ex_ctrl_legal;	// Rocket.scala:285:13, :360:14
      mem_ctrl_fp <= ex_ctrl_fp;	// Rocket.scala:285:13, :360:14
      mem_ctrl_branch <= ex_ctrl_branch;	// Rocket.scala:285:13, :360:14
      mem_ctrl_jal <= ex_ctrl_jal;	// Rocket.scala:285:13, :360:14
      mem_ctrl_jalr <= ex_ctrl_jalr;	// Rocket.scala:285:13, :360:14
      mem_ctrl_rxs2 <= ex_ctrl_rxs2;	// Rocket.scala:285:13, :360:14
      mem_ctrl_rxs1 <= ex_ctrl_rxs1;	// Rocket.scala:285:13, :360:14
      mem_ctrl_sel_alu2 <= ex_ctrl_sel_alu2;	// Mux.scala:46:19, Rocket.scala:360:14
      mem_ctrl_sel_alu1 <= ex_ctrl_sel_alu1;	// Mux.scala:46:19, Rocket.scala:360:14
      mem_ctrl_sel_imm <= ex_ctrl_sel_imm;	// Rocket.scala:360:14, :704:24
      mem_ctrl_alu_dw <= ex_ctrl_alu_dw;	// Rocket.scala:262:13, :360:14
      mem_ctrl_alu_fn <= ex_ctrl_alu_fn;	// Rocket.scala:263:13, :360:14
      mem_ctrl_mem <= ex_ctrl_mem;	// Rocket.scala:285:13, :360:14
      mem_ctrl_mem_cmd <= ex_ctrl_mem_cmd;	// Rocket.scala:285:13, :360:14
      mem_ctrl_mem_type <= ex_ctrl_mem_type;	// Rocket.scala:285:13, :360:14
      mem_ctrl_rfs1 <= ex_ctrl_rfs1;	// Rocket.scala:285:13, :360:14
      mem_ctrl_rfs2 <= ex_ctrl_rfs2;	// Rocket.scala:285:13, :360:14
      mem_ctrl_rfs3 <= ex_ctrl_rfs3;	// Rocket.scala:285:13, :360:14
      mem_ctrl_wfd <= ex_ctrl_wfd;	// Rocket.scala:285:13, :360:14
      mem_ctrl_div <= ex_ctrl_div;	// Rocket.scala:269:36, :360:14
      mem_ctrl_wxd <= ex_ctrl_wxd;	// Rocket.scala:240:19, :360:14
      mem_ctrl_csr <= ex_ctrl_csr;	// Rocket.scala:287:17, :360:14
      mem_ctrl_fence_i <= ex_ctrl_fence_i;	// Rocket.scala:303:23, :360:14
      mem_ctrl_fence <= ex_ctrl_fence;	// Rocket.scala:285:13, :360:14
      mem_ctrl_amo <= ex_ctrl_amo;	// Rocket.scala:285:13, :360:14
      mem_ctrl_dp <= ex_ctrl_dp;	// Rocket.scala:285:13, :360:14
      mem_reg_rvc <= ex_reg_rvc;	// Rocket.scala:259:19, :361:17
      mem_reg_load <= ex_ctrl_mem & (ex_ctrl_mem_cmd == 5'h0 | ex_ctrl_mem_cmd == 5'h6 | _T_119 | _T_98);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:11:8, Consts.scala:35:{31,48,75}, Rocket.scala:285:13, :362:{18,33}
      mem_reg_store <= ex_ctrl_mem & (ex_ctrl_mem_cmd == 5'h1 | _T_119 | _T_98);	// Consts.scala:36:{32,59}, Decode.scala:13:121, Rocket.scala:285:13, :363:{19,34}
      mem_reg_btb_hit <= ex_reg_btb_hit;	// Rocket.scala:364:21
    end
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_taken <= ex_reg_btb_resp_taken;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_mask <= ex_reg_btb_resp_mask;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_bridx <= ex_reg_btb_resp_bridx;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_target <= ex_reg_btb_resp_target;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_entry <= ex_reg_btb_resp_entry;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_bht_history <= ex_reg_btb_resp_bht_history;	// Rocket.scala:282:57, :365:46
    if (_T_30 & ex_reg_btb_hit)	// Rocket.scala:364:21, :365:46
      mem_reg_btb_resp_bht_value <= ex_reg_btb_resp_bht_value;	// Rocket.scala:282:57, :365:46
    if (_T_30) begin	// Rocket.scala:371:19
      mem_reg_flush_pipe <= ex_reg_flush_pipe;	// Rocket.scala:302:25, :366:24
      mem_reg_slow_bypass <= _T_119 | ex_ctrl_mem_type == 3'h0 | ex_ctrl_mem_type == 3'h4 | ex_ctrl_mem_type == 3'h1 |
                                                ex_ctrl_mem_type == 3'h5;	// Rocket.scala:285:13, :331:{50,91}, :367:25, :704:24, :709:23, :712:24, :715:22
      mem_reg_inst <= ex_reg_inst;	// Rocket.scala:235:29, :369:18
      mem_reg_pc <= ex_reg_pc;	// Mux.scala:46:16, Rocket.scala:370:16
      mem_reg_wdata <= alu_io_out;	// Rocket.scala:261:19, :371:19
    end
    if (_T_30 & ex_ctrl_rxs2 & ex_ctrl_mem)	// Rocket.scala:285:13, :373:19
      mem_reg_rs2 <= _ex_rs_1;	// Rocket.scala:373:19
    _T_4_40 <= div_io_req_ready & _T_29;	// Decoupled.scala:30:37, Rocket.scala:268:19, :396:37
    wb_reg_valid <= ~(_T_39 | _mem_xcpt | _T_124);	// Rocket.scala:397:45, :400:{16,19}
    wb_reg_replay <= (_dcache_kill_mem | mem_reg_replay | _T_124) & ~_T_1;	// Rocket.scala:337:36, :394:55, :401:{17,31,34}, :420:38
    wb_reg_xcpt <= _mem_xcpt & ~_T_1;	// Rocket.scala:401:34, :402:{15,27}, :420:38
    if (_mem_xcpt)	// Rocket.scala:403:34
      wb_reg_cause <= _T_123 ? mem_reg_cause : {60'h0, _mem_debug_breakpoint ? 4'hD : {1'h0, _mem_breakpoint ?
                                                3'h3 : _mem_npc_misaligned ? 3'h0 : _T_120 ? 3'h6 : _T_121 ? 3'h4 : {1'h1, _T_122, 1'h1}}};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Consts.scala:35:48, Mux.scala:31:69, Rocket.scala:357:34, :403:34, :704:24, :712:24, :715:22
    if (mem_reg_valid | mem_reg_replay | mem_reg_xcpt_interrupt) begin	// Rocket.scala:241:20, :337:{36,54}, :411:15
      wb_ctrl_legal <= mem_ctrl_legal;	// Rocket.scala:360:14, :405:13
      wb_ctrl_fp <= mem_ctrl_fp;	// Rocket.scala:360:14, :405:13
      wb_ctrl_branch <= mem_ctrl_branch;	// Rocket.scala:340:25, :405:13
      wb_ctrl_jal <= mem_ctrl_jal;	// Rocket.scala:341:8, :405:13
      wb_ctrl_jalr <= mem_ctrl_jalr;	// Rocket.scala:343:21, :405:13
      wb_ctrl_rxs2 <= mem_ctrl_rxs2;	// Rocket.scala:360:14, :405:13
      wb_ctrl_rxs1 <= mem_ctrl_rxs1;	// Rocket.scala:360:14, :405:13
      wb_ctrl_sel_alu2 <= mem_ctrl_sel_alu2;	// Rocket.scala:360:14, :405:13
      wb_ctrl_sel_alu1 <= mem_ctrl_sel_alu1;	// Rocket.scala:360:14, :405:13
      wb_ctrl_sel_imm <= mem_ctrl_sel_imm;	// Rocket.scala:360:14, :405:13
      wb_ctrl_alu_dw <= mem_ctrl_alu_dw;	// Rocket.scala:360:14, :405:13
      wb_ctrl_alu_fn <= mem_ctrl_alu_fn;	// Rocket.scala:360:14, :405:13
      wb_ctrl_mem <= mem_ctrl_mem;	// Rocket.scala:241:39, :405:13
      wb_ctrl_mem_cmd <= mem_ctrl_mem_cmd;	// Rocket.scala:360:14, :405:13
      wb_ctrl_mem_type <= mem_ctrl_mem_type;	// Rocket.scala:360:14, :405:13
      wb_ctrl_rfs1 <= mem_ctrl_rfs1;	// Rocket.scala:360:14, :405:13
      wb_ctrl_rfs2 <= mem_ctrl_rfs2;	// Rocket.scala:360:14, :405:13
      wb_ctrl_rfs3 <= mem_ctrl_rfs3;	// Rocket.scala:360:14, :405:13
      wb_ctrl_wfd <= mem_ctrl_wfd;	// Rocket.scala:360:14, :405:13
      wb_ctrl_div <= mem_ctrl_div;	// Rocket.scala:360:14, :405:13
      wb_ctrl_wxd <= mem_ctrl_wxd;	// Rocket.scala:241:20, :405:13
      wb_ctrl_csr <= mem_ctrl_csr;	// Rocket.scala:360:14, :405:13
      wb_ctrl_fence_i <= mem_ctrl_fence_i;	// Rocket.scala:360:14, :405:13
      wb_ctrl_fence <= mem_ctrl_fence;	// Rocket.scala:360:14, :405:13
      wb_ctrl_amo <= mem_ctrl_amo;	// Rocket.scala:360:14, :405:13
      wb_ctrl_dp <= mem_ctrl_dp;	// Rocket.scala:360:14, :405:13
      wb_reg_wdata <= ~mem_reg_xcpt & mem_ctrl_fp & mem_ctrl_wxd ? io_fpu_toint_data : _T_37;	// Rocket.scala:241:20, :346:27, :360:14, :406:{18,24,54}
      wb_reg_inst <= mem_reg_inst;	// Rocket.scala:236:31, :410:17
      wb_reg_pc <= mem_reg_pc;	// Rocket.scala:339:41, :411:15
    end
    dcache_blocked <= ~io_dmem_req_ready & (_T | dcache_blocked);	// Rocket.scala:325:45, :523:{18,40,62}, :581:41
    rocc_blocked <= ~wb_reg_xcpt & ~io_rocc_cmd_ready & rocc_blocked;	// Rocket.scala:417:56, :420:27, :450:48, :525:{16,54,76}
    `ifndef SYNTHESIS	// Rocket.scala:593:11
      if (`PRINTF_COND_ & _T_125)	// Rocket.scala:593:11
        $fwrite(32'h80000002, "Assertion failed\n    at Rocket.scala:593 assert(io.dmem.xcpt.asUInt.orR) // make sure s1_kill is exhaustive\n");	// Rocket.scala:593:11
      if (`STOP_COND_ & _T_125)	// Rocket.scala:593:11
        $fatal;	// Rocket.scala:593:11
    `endif
    _T_7_82 <= _ex_rs_0;	// Rocket.scala:639:42
    _T_8_83 <= _T_7_82;	// Rocket.scala:639:33
    _T_9_84 <= _ex_rs_1;	// Rocket.scala:640:42
    _T_10_85 <= _T_9_84;	// Rocket.scala:640:33
    `ifndef SYNTHESIS	// Rocket.scala:636:11
      if (`PRINTF_COND_ & ~reset)	// Rocket.scala:636:11
        $fwrite(32'h80000002, "C%d: %d [%d] pc=[%x] W[r%d=%x][%d] R[r%d=%x] R[r%d=%x] inst=[%x] DASM(%x)\n", io_hartid, csr_io_time[31:0], _T_45, wb_reg_pc, _rf_wen ? _rf_waddr : 5'h0, _rf_wdata, _rf_wen, wb_reg_inst[19:15], _T_8_83, wb_reg_inst[24:20], _T_10_85, wb_reg_inst, wb_reg_inst);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:11:8, Rocket.scala:187:19, :237:29, :411:15, :636:11, :637:32, :638:13, :639:21, :640:21
    `endif
  end // always @(posedge)
  IBuf ibuf (	// Rocket.scala:165:20
    .clock                             (clock),
    .reset                             (reset),
    .io_imem_valid                     (io_imem_resp_valid),
    .io_imem_bits_btb_valid            (io_imem_resp_bits_btb_valid),
    .io_imem_bits_btb_bits_taken       (io_imem_resp_bits_btb_bits_taken),
    .io_imem_bits_btb_bits_mask        (io_imem_resp_bits_btb_bits_mask),
    .io_imem_bits_btb_bits_bridx       (io_imem_resp_bits_btb_bits_bridx),
    .io_imem_bits_btb_bits_target      (io_imem_resp_bits_btb_bits_target),
    .io_imem_bits_btb_bits_entry       (io_imem_resp_bits_btb_bits_entry),
    .io_imem_bits_btb_bits_bht_history (io_imem_resp_bits_btb_bits_bht_history),
    .io_imem_bits_btb_bits_bht_value   (io_imem_resp_bits_btb_bits_bht_value),
    .io_imem_bits_pc                   (io_imem_resp_bits_pc),
    .io_imem_bits_data                 (io_imem_resp_bits_data),
    .io_imem_bits_mask                 (io_imem_resp_bits_mask),
    .io_imem_bits_xcpt_if              (io_imem_resp_bits_xcpt_if),
    .io_imem_bits_replay               (io_imem_resp_bits_replay),
    .io_kill                           (_take_pc_mem_wb),
    .io_inst_0_ready                   (~_T_73 | csr_io_interrupt),	// Rocket.scala:187:19, :547:{28,41}
    .io_imem_ready                     (io_imem_resp_ready),
    .io_pc                             (ibuf_io_pc),
    .io_btb_resp_taken                 (ibuf_io_btb_resp_taken),
    .io_btb_resp_mask                  (ibuf_io_btb_resp_mask),
    .io_btb_resp_bridx                 (ibuf_io_btb_resp_bridx),
    .io_btb_resp_target                (ibuf_io_btb_resp_target),
    .io_btb_resp_entry                 (ibuf_io_btb_resp_entry),
    .io_btb_resp_bht_history           (ibuf_io_btb_resp_bht_history),
    .io_btb_resp_bht_value             (ibuf_io_btb_resp_bht_value),
    .io_inst_0_valid                   (ibuf_io_inst_0_valid),
    .io_inst_0_bits_pf0                (ibuf_io_inst_0_bits_pf0),
    .io_inst_0_bits_pf1                (ibuf_io_inst_0_bits_pf1),
    .io_inst_0_bits_replay             (ibuf_io_inst_0_bits_replay),
    .io_inst_0_bits_btb_hit            (ibuf_io_inst_0_bits_btb_hit),
    .io_inst_0_bits_rvc                (ibuf_io_inst_0_bits_rvc),
    .io_inst_0_bits_inst_bits          (ibuf_io_inst_0_bits_inst_bits),
    .io_inst_0_bits_inst_rd            (ibuf_io_inst_0_bits_inst_rd),
    .io_inst_0_bits_inst_rs1           (ibuf_io_inst_0_bits_inst_rs1),
    .io_inst_0_bits_inst_rs2           (ibuf_io_inst_0_bits_inst_rs2),
    .io_inst_0_bits_inst_rs3           (ibuf_io_inst_0_bits_inst_rs3),
    .io_inst_0_bits_raw                (ibuf_io_inst_0_bits_raw)
  );
  _ext mem (	// Rocket.scala:682:23
    .R0_addr (~ibuf_io_inst_0_bits_inst_rs1),	// Rocket.scala:165:20, :683:39
    .R0_en   (1'h1),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
    .R0_clk  (clock),
    .R1_addr (~ibuf_io_inst_0_bits_inst_rs2),	// Rocket.scala:165:20, :683:39
    .R1_en   (1'h1),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
    .R1_clk  (clock),
    .W0_addr (~_rf_waddr),	// Rocket.scala:683:39
    .W0_en   (_rf_wen & |_rf_waddr),	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1316:9, Rocket.scala:694:16
    .W0_clk  (clock),
    .W0_data (_rf_wdata),
    .R0_data (mem_R0_data),
    .R1_data (mem_R1_data)
  );
  CSRFile csr (	// Rocket.scala:187:19
    .clock                    (clock),
    .reset                    (reset),
    .io_interrupts_debug      (io_interrupts_debug),
    .io_interrupts_mtip       (io_interrupts_mtip),
    .io_interrupts_msip       (io_interrupts_msip),
    .io_interrupts_meip       (io_interrupts_meip),
    .io_interrupts_seip       (io_interrupts_seip),
    .io_hartid                (io_hartid),
    .io_rw_addr               (wb_reg_inst[31:20]),	// Rocket.scala:237:29, :475:32
    .io_rw_cmd                (wb_reg_valid ? wb_ctrl_csr : 3'h0),	// Rocket.scala:405:13, :414:29, :476:23, :712:24
    .io_rw_wdata              (wb_reg_wdata),	// Rocket.scala:246:23
    .io_decode_csr            (ibuf_io_inst_0_bits_raw[31:20]),	// Rocket.scala:165:20, :461:48
    .io_exception             (wb_reg_xcpt),	// Rocket.scala:420:27
    .io_retire                (_T_45),
    .io_cause                 (wb_reg_cause),	// Rocket.scala:403:34
    .io_pc                    (wb_reg_pc),	// Rocket.scala:411:15
    .io_badaddr               ({_wb_reg_wdata_63to38 == 26'h0 | _wb_reg_wdata_63to38 == 26'h1 ? |_wb_reg_wdata_39to38 :
                &_wb_reg_wdata_63to38 | _wb_reg_wdata_63to38 == 26'h3FFFFFE ? &_wb_reg_wdata_39to38 :
                wb_reg_wdata[38], wb_reg_wdata[38:0]}),	// Cat.scala:30:58, Rocket.scala:246:23, :656:{10,13,25,30,45}, :657:{10,20,33,45,61,76}, :658:16
    .io_fcsr_flags_valid      (io_fpu_fcsr_flags_valid),
    .io_fcsr_flags_bits       (io_fpu_fcsr_flags_bits),
    .io_rocc_interrupt        (io_rocc_interrupt),
    .io_rw_rdata              (csr_io_rw_rdata),
    .io_decode_fp_illegal     (csr_io_decode_fp_illegal),
    .io_decode_read_illegal   (csr_io_decode_read_illegal),
    .io_decode_write_illegal  (csr_io_decode_write_illegal),
    .io_decode_write_flush    (csr_io_decode_write_flush),
    .io_decode_system_illegal (csr_io_decode_system_illegal),
    .io_csr_stall             (csr_io_csr_stall),
    .io_eret                  (csr_io_eret),
    .io_singleStep            (csr_io_singleStep),
    .io_status_debug          (csr_io_status_debug),
    .io_status_isa            (csr_io_status_isa),
    .io_status_prv            (csr_io_status_prv),
    .io_status_sd             (csr_io_status_sd),
    .io_status_tsr            (csr_io_status_tsr),
    .io_status_tw             (csr_io_status_tw),
    .io_status_tvm            (csr_io_status_tvm),
    .io_status_mxr            (csr_io_status_mxr),
    .io_status_pum            (csr_io_status_pum),
    .io_status_mprv           (csr_io_status_mprv),
    .io_status_fs             (csr_io_status_fs),
    .io_status_mpp            (csr_io_status_mpp),
    .io_status_spp            (csr_io_status_spp),
    .io_status_mpie           (csr_io_status_mpie),
    .io_status_spie           (csr_io_status_spie),
    .io_status_mie            (csr_io_status_mie),
    .io_status_sie            (csr_io_status_sie),
    .io_ptbr_mode             (io_ptw_ptbr_mode),
    .io_ptbr_ppn              (io_ptw_ptbr_ppn),
    .io_evec                  (csr_io_evec),
    .io_fatc                  (csr_io_fatc),
    .io_time                  (csr_io_time),
    .io_fcsr_rm               (io_fpu_fcsr_rm),
    .io_interrupt             (csr_io_interrupt),
    .io_interrupt_cause       (csr_io_interrupt_cause),
    .io_bp_0_control_dmode    (csr_io_bp_0_control_dmode),
    .io_bp_0_control_action   (csr_io_bp_0_control_action),
    .io_bp_0_control_tmatch   (csr_io_bp_0_control_tmatch),
    .io_bp_0_control_m        (csr_io_bp_0_control_m),
    .io_bp_0_control_s        (csr_io_bp_0_control_s),
    .io_bp_0_control_u        (csr_io_bp_0_control_u),
    .io_bp_0_control_x        (csr_io_bp_0_control_x),
    .io_bp_0_control_w        (csr_io_bp_0_control_w),
    .io_bp_0_control_r        (csr_io_bp_0_control_r),
    .io_bp_0_address          (csr_io_bp_0_address)
  );
  BreakpointUnit bpu (	// Rocket.scala:215:19
    .io_status_debug        (csr_io_status_debug),	// Rocket.scala:187:19
    .io_status_prv          (csr_io_status_prv),	// Rocket.scala:187:19
    .io_bp_0_control_action (csr_io_bp_0_control_action),	// Rocket.scala:187:19
    .io_bp_0_control_tmatch (csr_io_bp_0_control_tmatch),	// Rocket.scala:187:19
    .io_bp_0_control_m      (csr_io_bp_0_control_m),	// Rocket.scala:187:19
    .io_bp_0_control_s      (csr_io_bp_0_control_s),	// Rocket.scala:187:19
    .io_bp_0_control_u      (csr_io_bp_0_control_u),	// Rocket.scala:187:19
    .io_bp_0_control_x      (csr_io_bp_0_control_x),	// Rocket.scala:187:19
    .io_bp_0_control_w      (csr_io_bp_0_control_w),	// Rocket.scala:187:19
    .io_bp_0_control_r      (csr_io_bp_0_control_r),	// Rocket.scala:187:19
    .io_bp_0_address        (csr_io_bp_0_address),	// Rocket.scala:187:19
    .io_pc                  (ibuf_io_pc[38:0]),	// Rocket.scala:165:20, :218:13
    .io_ea                  (_mem_reg_wdata_38to0),
    .io_xcpt_if             (bpu_io_xcpt_if),
    .io_xcpt_ld             (bpu_io_xcpt_ld),
    .io_xcpt_st             (bpu_io_xcpt_st),
    .io_debug_if            (bpu_io_debug_if),
    .io_debug_ld            (bpu_io_debug_ld),
    .io_debug_st            (bpu_io_debug_st)
  );
  ALU alu (	// Rocket.scala:261:19
    .io_dw        (ex_ctrl_alu_dw),	// Rocket.scala:262:13
    .io_fn        (ex_ctrl_alu_fn),	// Rocket.scala:263:13
    .io_in2       (ex_ctrl_sel_alu2 == 2'h2 ? _ex_rs_1 : {{32{_T_28[31]}}, _T_28}),	// Mux.scala:31:69, :46:{16,19}
    .io_in1       (ex_ctrl_sel_alu1 == 2'h1 ? _ex_rs_0 : {{24{_T_27[39]}}, _T_27}),	// Mux.scala:31:69, :46:{16,19}
    .io_out       (alu_io_out),
    .io_adder_out (alu_io_adder_out),
    .io_cmp_out   (alu_io_cmp_out)
  );
  MulDiv div (	// Rocket.scala:268:19
    .clock             (clock),
    .reset             (reset),
    .io_req_valid      (_T_29),
    .io_req_bits_fn    (ex_ctrl_alu_fn),	// Rocket.scala:263:13
    .io_req_bits_dw    (ex_ctrl_alu_dw),	// Rocket.scala:262:13
    .io_req_bits_in1   (_ex_rs_0),
    .io_req_bits_in2   (_ex_rs_1),
    .io_req_bits_tag   (_ex_waddr),
    .io_kill           (_T_39 & _T_4_40),	// Rocket.scala:396:31
    .io_resp_ready     (_T_0),	// Rocket.scala:443:23
    .io_req_ready      (div_io_req_ready),
    .io_resp_valid     (div_io_resp_valid),
    .io_resp_bits_data (div_io_resp_bits_data),
    .io_resp_bits_tag  (div_io_resp_bits_tag)
  );
  assign io_imem_req_valid = _take_pc_mem_wb;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_req_bits_pc = _T_75;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_req_bits_speculative = ~_T_1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:401:34, :420:38
  assign io_imem_btb_update_valid = mem_reg_replay & mem_reg_btb_hit | _T_76 & (_T_32 | mem_ctrl_jalr | mem_ctrl_jal | ~_T_38)
                & _mem_misprediction;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:337:36, :341:8, :343:21, :364:21, :549:{47,67,100,120,123}
  assign io_imem_btb_update_bits_prediction_valid = mem_reg_btb_hit;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:364:21
  assign io_imem_btb_update_bits_prediction_bits_taken = mem_reg_btb_resp_taken;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_mask = mem_reg_btb_resp_mask;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_bridx = mem_reg_btb_resp_bridx;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_target = mem_reg_btb_resp_target;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_entry = mem_reg_btb_resp_entry;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_bht_history = mem_reg_btb_resp_bht_history;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_prediction_bits_bht_value = mem_reg_btb_resp_bht_value;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_btb_update_bits_pc = ~_T_80;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:555:33
  assign io_imem_btb_update_bits_target = _T_75[38:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:553:34
  assign io_imem_btb_update_bits_taken = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_btb_update_bits_isValid = ~mem_reg_replay & _T_38;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:337:36, :550:{38,54}
  assign io_imem_btb_update_bits_isJump = _T_77;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_btb_update_bits_isReturn = _T_78;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_btb_update_bits_br_pc = _T_79;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_bht_update_valid = _T_76 & mem_ctrl_branch;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:340:25, :559:60
  assign io_imem_bht_update_bits_prediction_valid = mem_reg_btb_hit;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:364:21
  assign io_imem_bht_update_bits_prediction_bits_taken = mem_reg_btb_resp_taken;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_mask = mem_reg_btb_resp_mask;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_bridx = mem_reg_btb_resp_bridx;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_target = mem_reg_btb_resp_target;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_entry = mem_reg_btb_resp_entry;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_bht_history = mem_reg_btb_resp_bht_history;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_prediction_bits_bht_value = mem_reg_btb_resp_bht_value;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_bht_update_bits_pc = ~_T_80;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:555:33
  assign io_imem_bht_update_bits_taken = _mem_br_taken;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_bht_update_bits_mispredict = _mem_misprediction;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_ras_update_valid = _T_76;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_ras_update_bits_isCall = _T_77 & _mem_reg_inst_7;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:567:68
  assign io_imem_ras_update_bits_isReturn = _T_78;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_imem_ras_update_bits_returnAddr = _T_37[38:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:566:38
  assign io_imem_ras_update_bits_prediction_valid = mem_reg_btb_hit;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:364:21
  assign io_imem_ras_update_bits_prediction_bits_taken = mem_reg_btb_resp_taken;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_mask = mem_reg_btb_resp_mask;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_bridx = mem_reg_btb_resp_bridx;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_target = mem_reg_btb_resp_target;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_entry = mem_reg_btb_resp_entry;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_bht_history = mem_reg_btb_resp_bht_history;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_ras_update_bits_prediction_bits_bht_value = mem_reg_btb_resp_bht_value;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:365:46
  assign io_imem_flush_icache = wb_reg_valid & wb_ctrl_fence_i & ~io_dmem_s2_nack;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:405:13, :414:29, :544:{59,62}
  assign io_imem_flush_tlb = csr_io_fatc;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_dmem_req_valid = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:581:41
  assign io_dmem_req_bits_addr = {_ex_rs_0_63to38 == 26'h0 | _ex_rs_0_63to38 == 26'h1 ? |_alu_io_adder_out_39to38 :
                &_ex_rs_0_63to38 | _ex_rs_0_63to38 == 26'h3FFFFFE ? &_alu_io_adder_out_39to38 :
                alu_io_adder_out[38], alu_io_adder_out[38:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Cat.scala:30:58, Rocket.scala:261:19, :656:{10,13,25,30,45}, :657:{10,20,33,45,61,76}, :658:16
  assign io_dmem_req_bits_tag = {1'h0, _ex_waddr, ex_ctrl_fp};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:285:13
  assign io_dmem_req_bits_cmd = ex_ctrl_mem_cmd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:285:13
  assign io_dmem_req_bits_typ = ex_ctrl_mem_type;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:285:13
  assign io_dmem_req_bits_phys = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_dmem_req_bits_data = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:157:23
  assign io_dmem_s1_kill = _T_81;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_dmem_s1_data = mem_ctrl_fp ? io_fpu_store_data : mem_reg_rs2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:360:14, :373:19, :590:25
  assign io_dmem_invalidate_lr = wb_reg_xcpt;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:420:27
  assign io_ptw_ptbr_asid = 16'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_invalidate = csr_io_fatc;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_debug = csr_io_status_debug;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_isa = csr_io_status_isa;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_prv = csr_io_status_prv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_sd = csr_io_status_sd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_zero2 = 27'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_sxl = 2'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69
  assign io_ptw_status_uxl = 2'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69
  assign io_ptw_status_sd_rv32 = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_ptw_status_zero1 = 8'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_tsr = csr_io_status_tsr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_tw = csr_io_status_tw;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_tvm = csr_io_status_tvm;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_mxr = csr_io_status_mxr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_pum = csr_io_status_pum;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_mprv = csr_io_status_mprv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_xs = 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:292:24
  assign io_ptw_status_fs = csr_io_status_fs;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_mpp = csr_io_status_mpp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_hpp = 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:292:24
  assign io_ptw_status_spp = csr_io_status_spp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_mpie = csr_io_status_mpie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_hpie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_ptw_status_spie = csr_io_status_spie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_upie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_ptw_status_mie = csr_io_status_mie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_hie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_ptw_status_sie = csr_io_status_sie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_ptw_status_uie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_inst = ibuf_io_inst_0_bits_inst_bits;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:165:20
  assign io_fpu_fromint_data = _ex_rs_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_dmem_resp_val = _dmem_resp_valid & _dmem_resp_fpu;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:576:43
  assign io_fpu_dmem_resp_type = io_dmem_resp_bits_typ;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_dmem_resp_tag = _dmem_resp_waddr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_dmem_resp_data = io_dmem_resp_bits_data_word_bypass;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_valid = ~_T_74 & _T_8;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:185:34, :571:31
  assign io_fpu_killx = _T_31;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_fpu_killm = _T_39;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_valid = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_bits_inst_funct = wb_reg_inst[31:25];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_rs2 = wb_reg_inst[24:20];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_rs1 = wb_reg_inst[19:15];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_xd = wb_reg_inst[14];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_xs1 = wb_reg_inst[13];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_xs2 = wb_reg_inst[12];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_rd = wb_reg_inst[11:7];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_inst_opcode = wb_reg_inst[6:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:237:29, :599:58
  assign io_rocc_cmd_bits_rs1 = wb_reg_wdata;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:246:23
  assign io_rocc_cmd_bits_rs2 = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:157:23
  assign io_rocc_cmd_bits_status_debug = csr_io_status_debug;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_isa = csr_io_status_isa;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_prv = csr_io_status_prv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_sd = csr_io_status_sd;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_zero2 = 27'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_sxl = 2'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69
  assign io_rocc_cmd_bits_status_uxl = 2'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Mux.scala:31:69
  assign io_rocc_cmd_bits_status_sd_rv32 = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_bits_status_zero1 = 8'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_tsr = csr_io_status_tsr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_tw = csr_io_status_tw;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_tvm = csr_io_status_tvm;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_mxr = csr_io_status_mxr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_pum = csr_io_status_pum;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_mprv = csr_io_status_mprv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_xs = 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:292:24
  assign io_rocc_cmd_bits_status_fs = csr_io_status_fs;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_mpp = csr_io_status_mpp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_hpp = 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:292:24
  assign io_rocc_cmd_bits_status_spp = csr_io_status_spp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_mpie = csr_io_status_mpie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_hpie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_bits_status_spie = csr_io_status_spie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_upie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_bits_status_mie = csr_io_status_mie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_hie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_cmd_bits_status_sie = csr_io_status_sie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_cmd_bits_status_uie = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_resp_ready = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_req_ready = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_s2_nack = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_acquire = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_release = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_resp_valid = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_resp_bits_addr = 40'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:187:19
  assign io_rocc_mem_resp_bits_tag = 7'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, :11:8
  assign io_rocc_mem_resp_bits_cmd = 5'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, :11:8
  assign io_rocc_mem_resp_bits_typ = 3'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:712:24
  assign io_rocc_mem_resp_bits_data = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:157:23
  assign io_rocc_mem_resp_bits_replay = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_resp_bits_has_data = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_resp_bits_data_word_bypass = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:157:23
  assign io_rocc_mem_resp_bits_store_data = 64'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10, Rocket.scala:157:23
  assign io_rocc_mem_replay_next = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_xcpt_ma_ld = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_xcpt_ma_st = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_xcpt_pf_ld = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_xcpt_pf_st = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_mem_ordered = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
  assign io_rocc_exception = 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:6:10
endmodule

module IBuf(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10
  input         clock, reset, io_imem_valid, io_imem_bits_btb_valid,
  input         io_imem_bits_btb_bits_taken,
  input  [1:0]  io_imem_bits_btb_bits_mask,
  input         io_imem_bits_btb_bits_bridx,
  input  [38:0] io_imem_bits_btb_bits_target,
  input  [5:0]  io_imem_bits_btb_bits_entry,
  input  [6:0]  io_imem_bits_btb_bits_bht_history,
  input  [1:0]  io_imem_bits_btb_bits_bht_value,
  input  [39:0] io_imem_bits_pc,
  input  [31:0] io_imem_bits_data,
  input  [1:0]  io_imem_bits_mask,
  input         io_imem_bits_xcpt_if, io_imem_bits_replay, io_kill, io_inst_0_ready,
  output        io_imem_ready,
  output [39:0] io_pc,
  output        io_btb_resp_taken,
  output [1:0]  io_btb_resp_mask,
  output        io_btb_resp_bridx,
  output [38:0] io_btb_resp_target,
  output [5:0]  io_btb_resp_entry,
  output [6:0]  io_btb_resp_bht_history,
  output [1:0]  io_btb_resp_bht_value,
  output        io_inst_0_valid, io_inst_0_bits_pf0, io_inst_0_bits_pf1,
  output        io_inst_0_bits_replay, io_inst_0_bits_btb_hit, io_inst_0_bits_rvc,
  output [31:0] io_inst_0_bits_inst_bits,
  output [4:0]  io_inst_0_bits_inst_rd, io_inst_0_bits_inst_rs1, io_inst_0_bits_inst_rs2,
  output [4:0]  io_inst_0_bits_inst_rs3,
  output [31:0] io_inst_0_bits_raw);

  wire [1:0]  _T;	// IBuf.scala:107:41
  wire        RVCExpander_io_rvc;	// IBuf.scala:93:21
  reg         nBufValid;	// IBuf.scala:35:47
  reg         buf_btb_valid;	// IBuf.scala:36:16
  reg         buf_btb_bits_taken;	// IBuf.scala:36:16
  reg  [1:0]  buf_btb_bits_mask;	// IBuf.scala:36:16
  reg         buf_btb_bits_bridx;	// IBuf.scala:36:16
  reg  [38:0] buf_btb_bits_target;	// IBuf.scala:36:16
  reg  [5:0]  buf_btb_bits_entry;	// IBuf.scala:36:16
  reg  [6:0]  buf_btb_bits_bht_history;	// IBuf.scala:36:16
  reg  [1:0]  buf_btb_bits_bht_value;	// IBuf.scala:36:16
  reg  [39:0] buf_pc;	// IBuf.scala:36:16
  reg  [31:0] buf_data;	// IBuf.scala:36:16
  reg  [1:0]  buf_mask;	// IBuf.scala:36:16
  reg         buf_xcpt_if;	// IBuf.scala:36:16
  reg         buf_replay;	// IBuf.scala:36:16
  reg         ibufBTBHit;	// IBuf.scala:37:23
  reg         ibufBTBResp_taken;	// IBuf.scala:38:24
  reg  [1:0]  ibufBTBResp_mask;	// IBuf.scala:38:24
  reg         ibufBTBResp_bridx;	// IBuf.scala:38:24
  reg  [38:0] ibufBTBResp_target;	// IBuf.scala:38:24
  reg  [5:0]  ibufBTBResp_entry;	// IBuf.scala:38:24
  reg  [6:0]  ibufBTBResp_bht_history;	// IBuf.scala:38:24
  reg  [1:0]  ibufBTBResp_bht_value;	// IBuf.scala:38:24

  `ifndef SYNTHESIS	// IBuf.scala:35:47
    `ifdef RANDOMIZE_REG_INIT	// IBuf.scala:35:47
      reg [31:0] _RANDOM;	// IBuf.scala:35:47
      reg [31:0] _RANDOM_0;	// IBuf.scala:36:16
      reg [31:0] _RANDOM_1;	// IBuf.scala:36:16
      reg [31:0] _RANDOM_2;	// IBuf.scala:36:16
      reg [31:0] _RANDOM_3;	// IBuf.scala:36:16
      reg [31:0] _RANDOM_4;	// IBuf.scala:38:24
      reg [31:0] _RANDOM_5;	// IBuf.scala:38:24

    `endif
    initial begin	// IBuf.scala:35:47
      `INIT_RANDOM_PROLOG_	// IBuf.scala:35:47
      `ifdef RANDOMIZE_REG_INIT	// IBuf.scala:35:47
        _RANDOM = `RANDOM;	// IBuf.scala:35:47
        nBufValid = _RANDOM[0];	// IBuf.scala:35:47
        buf_btb_valid = _RANDOM[1];	// IBuf.scala:36:16
        buf_btb_bits_taken = _RANDOM[2];	// IBuf.scala:36:16
        buf_btb_bits_mask = _RANDOM[4:3];	// IBuf.scala:36:16
        buf_btb_bits_bridx = _RANDOM[5];	// IBuf.scala:36:16
        _RANDOM_0 = `RANDOM;	// IBuf.scala:36:16
        buf_btb_bits_target = {_RANDOM_0[12:0], _RANDOM[31:6]};	// IBuf.scala:36:16
        buf_btb_bits_entry = _RANDOM_0[18:13];	// IBuf.scala:36:16
        buf_btb_bits_bht_history = _RANDOM_0[25:19];	// IBuf.scala:36:16
        buf_btb_bits_bht_value = _RANDOM_0[27:26];	// IBuf.scala:36:16
        _RANDOM_1 = `RANDOM;	// IBuf.scala:36:16
        _RANDOM_2 = `RANDOM;	// IBuf.scala:36:16
        buf_pc = {_RANDOM_2[3:0], _RANDOM_1, _RANDOM_0[31:28]};	// IBuf.scala:36:16
        _RANDOM_3 = `RANDOM;	// IBuf.scala:36:16
        buf_data = {_RANDOM_3[3:0], _RANDOM_2[31:4]};	// IBuf.scala:36:16
        buf_mask = _RANDOM_3[5:4];	// IBuf.scala:36:16
        buf_xcpt_if = _RANDOM_3[6];	// IBuf.scala:36:16
        buf_replay = _RANDOM_3[7];	// IBuf.scala:36:16
        ibufBTBHit = _RANDOM_3[8];	// IBuf.scala:37:23
        ibufBTBResp_taken = _RANDOM_3[9];	// IBuf.scala:38:24
        ibufBTBResp_mask = _RANDOM_3[11:10];	// IBuf.scala:38:24
        ibufBTBResp_bridx = _RANDOM_3[12];	// IBuf.scala:38:24
        _RANDOM_4 = `RANDOM;	// IBuf.scala:38:24
        ibufBTBResp_target = {_RANDOM_4[19:0], _RANDOM_3[31:13]};	// IBuf.scala:38:24
        ibufBTBResp_entry = _RANDOM_4[25:20];	// IBuf.scala:38:24
        _RANDOM_5 = `RANDOM;	// IBuf.scala:38:24
        ibufBTBResp_bht_history = {_RANDOM_5[0], _RANDOM_4[31:26]};	// IBuf.scala:38:24
        ibufBTBResp_bht_value = _RANDOM_5[2:1];	// IBuf.scala:38:24
      `endif
    end // initial
  `endif
      wire _pcWordBits = io_imem_bits_pc[1];	// Package.scala:44:13
      wire [1:0] _T_0 = {1'h0, io_imem_bits_btb_bits_bridx};	// IBuf.scala:35:47, :43:100
      wire [1:0] _T_1 = (io_imem_bits_btb_valid & io_imem_bits_btb_bits_taken ? _T_0 + 2'h1 : 2'h2) - {1'h0,
                _pcWordBits};	// IBuf.scala:35:47, :43:{16,40,100,124}, :99:81
      wire [1:0] _T_2 = _T - {1'h0, nBufValid};	// IBuf.scala:35:47, :44:25, :107:41
      wire [1:0] _T_3 = {1'h0, nBufValid};	// IBuf.scala:35:47, :44:25, :46:27
      wire _T_4 = _T >= _T_3;	// IBuf.scala:46:27, :107:41
      wire [1:0] _T_5 = _T_1 - _T_2;	// IBuf.scala:46:72
      wire _T_6 = _T_5[1];	// IBuf.scala:46:65
      wire [190:0] _T_7 = {63'h0, {2{{2{io_imem_bits_data[31:16]}}}}, io_imem_bits_data,
                {2{io_imem_bits_data[15:0]}}} << {185'h0, {1'h0, nBufValid} - 2'h2 - {1'h0, _pcWordBits},
                4'h0};	// Cat.scala:30:58, IBuf.scala:35:47, :44:25, :73:{32,44}, :74:87, :85:25, :126:58, :127:10
      wire [62:0] _T_8 = 63'hFFFFFFFF << {58'h0, nBufValid, 4'h0};	// IBuf.scala:44:25, :76:51, :85:25, :134:10
      wire [31:0] _icMask = _T_8[31:0];	// IBuf.scala:76:92
      wire [31:0] _inst = _T_7[95:64] & _icMask | buf_data & ~_icMask;	// IBuf.scala:60:16, :77:{21,30,41,43}, Package.scala:44:13
      wire [3:0] _T_9 = 4'h1 << (io_imem_valid ? _T_1 : 2'h0) + {1'h0, nBufValid};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, IBuf.scala:35:47, :44:25, :45:{19,49}, OneHot.scala:47:11
      wire [1:0] _T_10 = _T_9[1:0] - 2'h1;	// IBuf.scala:79:33
      wire [1:0] _T_11 = (2'h1 << _T_3) - 2'h1;	// IBuf.scala:80:37, :99:81, OneHot.scala:47:11
      wire [1:0] _xcpt_if = _T_10 & ((buf_xcpt_if ? _T_11 : 2'h0) | (io_imem_bits_xcpt_if ? ~_T_11 : 2'h0));	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, IBuf.scala:59:11, :81:{23,29,61,66,89}
      wire [1:0] _ic_replay = _T_10 & ((buf_replay ? _T_11 : 2'h0) | (io_imem_bits_replay ? ~_T_11 : 2'h0));	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, IBuf.scala:59:11, :81:89, :82:{25,31,62,67}
      wire [63:0] _T_12 = {{2{io_imem_bits_data[31:16]}}, io_imem_bits_data} >> {58'h0, {1'h0, _pcWordBits} + _T_2,
                4'h0};	// Cat.scala:30:58, IBuf.scala:35:47, :57:30, :85:25, :133:58, :134:10
      wire _T_13 = ~io_imem_bits_btb_valid | io_imem_bits_btb_bits_bridx >= _pcWordBits | reset;	// IBuf.scala:84:{9,10,65}
      wire _T_14 = io_imem_valid & _T_4 & _T_2 < _T_1 & ~_T_6;	// IBuf.scala:46:65, :56:{60,66}
  always @(posedge clock) begin	// IBuf.scala:59:11
    if (reset)	// IBuf.scala:35:47
      nBufValid <= 1'h0;	// IBuf.scala:35:47
    else	// IBuf.scala:35:47
      nBufValid <= ~io_kill & (_T_14 ? _T_5[0] : ~_T_4 & nBufValid - _T[0]);	// IBuf.scala:44:25, :49:{21,62}, :58:17, :69:17, :107:41
    if (_T_14) begin	// IBuf.scala:62:18
      buf_btb_valid <= io_imem_bits_btb_valid;	// IBuf.scala:59:11
      buf_btb_bits_taken <= io_imem_bits_btb_bits_taken;	// IBuf.scala:59:11
      buf_btb_bits_mask <= io_imem_bits_btb_bits_mask;	// IBuf.scala:59:11
      buf_btb_bits_bridx <= io_imem_bits_btb_bits_bridx;	// IBuf.scala:59:11
      buf_btb_bits_target <= io_imem_bits_btb_bits_target;	// IBuf.scala:59:11
      buf_btb_bits_entry <= io_imem_bits_btb_bits_entry;	// IBuf.scala:59:11
      buf_btb_bits_bht_history <= io_imem_bits_btb_bits_bht_history;	// IBuf.scala:59:11
      buf_btb_bits_bht_value <= io_imem_bits_btb_bits_bht_value;	// IBuf.scala:59:11
      buf_mask <= io_imem_bits_mask;	// IBuf.scala:59:11
      buf_xcpt_if <= io_imem_bits_xcpt_if;	// IBuf.scala:59:11
      buf_replay <= io_imem_bits_replay;	// IBuf.scala:59:11
      buf_data <= {16'h0, _T_12[15:0]};	// IBuf.scala:60:{16,59}
      buf_pc <= io_imem_bits_pc & 40'hFFFFFFFFFC | io_imem_bits_pc + {37'h0, _T_2, 1'h0} & 40'h3;	// IBuf.scala:35:47, :61:{14,33,35,47,66,107}
      ibufBTBHit <= io_imem_bits_btb_valid;	// IBuf.scala:62:18
    end
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_taken <= io_imem_bits_btb_bits_taken;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_mask <= io_imem_bits_btb_bits_mask;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_target <= io_imem_bits_btb_bits_target;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_entry <= io_imem_bits_btb_bits_entry;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_bht_history <= io_imem_bits_btb_bits_bht_history;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:64:21
      ibufBTBResp_bht_value <= io_imem_bits_btb_bits_bht_value;	// IBuf.scala:64:21
    if (_T_14 & io_imem_bits_btb_valid)	// IBuf.scala:65:27
      ibufBTBResp_bridx <= io_imem_bits_btb_bits_bridx + _T_2[0];	// IBuf.scala:65:{27,58}
    `ifndef SYNTHESIS	// IBuf.scala:84:9
      if (`PRINTF_COND_ & ~_T_13)	// IBuf.scala:84:9
        $fwrite(32'h80000002, "Assertion failed\n    at IBuf.scala:84 assert(!io.imem.bits.btb.valid || io.imem.bits.btb.bits.bridx >= pcWordBits)\n");	// IBuf.scala:84:9
      if (`STOP_COND_ & ~_T_13)	// IBuf.scala:84:9
        $fatal;	// IBuf.scala:84:9
    `endif
  end // always @(posedge)
      wire [3:0] _T_15 = 4'h1 << _T_0 + _T_3 - {1'h0, _pcWordBits};	// IBuf.scala:35:47, :85:{87,100}, OneHot.scala:47:11
      wire [1:0] _T_16 = (ibufBTBHit ? 2'h1 << ibufBTBResp_bridx : 2'h0) & _T_11;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, IBuf.scala:62:18, :65:27, :83:27, :86:35, :99:81, OneHot.scala:47:11
      wire [1:0] _T_17 = _T_16 | (io_imem_bits_btb_valid ? _T_15[1:0] : 2'h0) & ~_T_11;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, IBuf.scala:81:89, :85:25, :86:{45,60}
      wire _T_18 = _T_17[0];	// IBuf.scala:99:63
      wire _T_19 = _ic_replay[0] | ~RVCExpander_io_rvc & (_T_18 | _ic_replay[1]);	// IBuf.scala:93:21, :99:{29,33,37,49,67,79}
      wire _xcpt_if_1 = _xcpt_if[1];	// IBuf.scala:100:75
      wire _T_20 = _T_10[0] & (RVCExpander_io_rvc | _T_10[1] | _xcpt_if_1 | _T_19);	// IBuf.scala:93:21, :100:{32,36,59,81}
  assign _T = io_inst_0_ready & _T_20 ? (RVCExpander_io_rvc ? 2'h1 : 2'h2) : 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1761:15, Decoupled.scala:30:37, IBuf.scala:43:16, :93:21, :99:81, :107:{41,47}
  RVCExpander RVCExpander (	// IBuf.scala:93:21
    .io_in       (_inst),
    .io_out_bits (io_inst_0_bits_inst_bits),
    .io_out_rd   (io_inst_0_bits_inst_rd),
    .io_out_rs1  (io_inst_0_bits_inst_rs1),
    .io_out_rs2  (io_inst_0_bits_inst_rs2),
    .io_out_rs3  (io_inst_0_bits_inst_rs3),
    .io_rvc      (RVCExpander_io_rvc)
  );
  assign io_imem_ready = _T_4 & (_T_2 >= _T_1 | ~_T_6);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:46:{40,53,60,65}
  assign io_pc = nBufValid ? buf_pc : io_imem_bits_pc;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:44:25, :61:14, :89:15
  assign io_btb_resp_taken = |_T_16 ? ibufBTBResp_taken : io_imem_bits_btb_bits_taken;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_btb_resp_mask = |_T_16 ? ibufBTBResp_mask : io_imem_bits_btb_bits_mask;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_btb_resp_bridx = |_T_16 ? ibufBTBResp_bridx : io_imem_bits_btb_bits_bridx;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:65:27, :88:{21,49}
  assign io_btb_resp_target = |_T_16 ? ibufBTBResp_target : io_imem_bits_btb_bits_target;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_btb_resp_entry = |_T_16 ? ibufBTBResp_entry : io_imem_bits_btb_bits_entry;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_btb_resp_bht_history = |_T_16 ? ibufBTBResp_bht_history : io_imem_bits_btb_bits_bht_history;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_btb_resp_bht_value = |_T_16 ? ibufBTBResp_bht_value : io_imem_bits_btb_bits_bht_value;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:64:21, :88:{21,49}
  assign io_inst_0_valid = _T_20;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10
  assign io_inst_0_bits_pf0 = _xcpt_if[0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:101:37
  assign io_inst_0_bits_pf1 = ~RVCExpander_io_rvc & _xcpt_if_1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:93:21, :99:37, :102:42
  assign io_inst_0_bits_replay = _T_19;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10
  assign io_inst_0_bits_btb_hit = _T_18 | ~RVCExpander_io_rvc & _T_17[1];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:93:21, :99:37, :104:{48,64,77}
  assign io_inst_0_bits_rvc = RVCExpander_io_rvc;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10, IBuf.scala:93:21
  assign io_inst_0_bits_raw = _inst;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1747:10
endmodule

module CSRFile(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10
  input         clock, reset, io_interrupts_debug, io_interrupts_mtip,
  input         io_interrupts_msip, io_interrupts_meip, io_interrupts_seip,
  input  [63:0] io_hartid,
  input  [11:0] io_rw_addr,
  input  [2:0]  io_rw_cmd,
  input  [63:0] io_rw_wdata,
  input  [11:0] io_decode_csr,
  input         io_exception, io_retire,
  input  [63:0] io_cause,
  input  [39:0] io_pc, io_badaddr,
  input         io_fcsr_flags_valid,
  input  [4:0]  io_fcsr_flags_bits,
  input         io_rocc_interrupt,
  output [63:0] io_rw_rdata,
  output        io_decode_fp_illegal, io_decode_read_illegal, io_decode_write_illegal,
  output        io_decode_write_flush, io_decode_system_illegal, io_csr_stall, io_eret,
  output        io_singleStep, io_status_debug,
  output [31:0] io_status_isa,
  output [1:0]  io_status_prv,
  output        io_status_sd, io_status_tsr, io_status_tw, io_status_tvm, io_status_mxr,
  output        io_status_pum, io_status_mprv,
  output [1:0]  io_status_fs, io_status_mpp,
  output        io_status_spp, io_status_mpie, io_status_spie, io_status_mie,
  output        io_status_sie,
  output [3:0]  io_ptbr_mode,
  output [43:0] io_ptbr_ppn,
  output [39:0] io_evec,
  output        io_fatc,
  output [63:0] io_time,
  output [2:0]  io_fcsr_rm,
  output        io_interrupt,
  output [63:0] io_interrupt_cause,
  output        io_bp_0_control_dmode, io_bp_0_control_action,
  output [1:0]  io_bp_0_control_tmatch,
  output        io_bp_0_control_m, io_bp_0_control_s, io_bp_0_control_u,
  output        io_bp_0_control_x, io_bp_0_control_w, io_bp_0_control_r,
  output [38:0] io_bp_0_address);

  wire        _T;	// CSR.scala:451:34
  reg  [1:0]  reg_mstatus_prv;	// CSR.scala:197:24
  reg         reg_mstatus_tsr;	// CSR.scala:197:24
  reg         reg_mstatus_tw;	// CSR.scala:197:24
  reg         reg_mstatus_tvm;	// CSR.scala:197:24
  reg         reg_mstatus_mxr;	// CSR.scala:197:24
  reg         reg_mstatus_pum;	// CSR.scala:197:24
  reg         reg_mstatus_mprv;	// CSR.scala:197:24
  reg  [1:0]  reg_mstatus_fs;	// CSR.scala:197:24
  reg  [1:0]  reg_mstatus_mpp;	// CSR.scala:197:24
  reg         reg_mstatus_spp;	// CSR.scala:197:24
  reg         reg_mstatus_mpie;	// CSR.scala:197:24
  reg         reg_mstatus_spie;	// CSR.scala:197:24
  reg         reg_mstatus_mie;	// CSR.scala:197:24
  reg         reg_mstatus_sie;	// CSR.scala:197:24
  reg         reg_dcsr_ebreakm;	// CSR.scala:205:21
  reg         reg_dcsr_ebreaks;	// CSR.scala:205:21
  reg         reg_dcsr_ebreaku;	// CSR.scala:205:21
  reg  [2:0]  reg_dcsr_cause;	// CSR.scala:205:21
  reg         reg_dcsr_debugint;	// CSR.scala:205:21
  reg         reg_dcsr_halt;	// CSR.scala:205:21
  reg         reg_dcsr_step;	// CSR.scala:205:21
  reg  [1:0]  reg_dcsr_prv;	// CSR.scala:205:21
  reg         reg_debug;	// CSR.scala:232:22
  reg  [39:0] reg_dpc;	// CSR.scala:234:20
  reg  [63:0] reg_dscratch;	// CSR.scala:235:25
  reg         reg_singleStepped;	// CSR.scala:236:30
  reg         reg_bp_0_control_dmode;	// CSR.scala:239:19
  reg         reg_bp_0_control_action;	// CSR.scala:239:19
  reg  [1:0]  reg_bp_0_control_tmatch;	// CSR.scala:239:19
  reg         reg_bp_0_control_m;	// CSR.scala:239:19
  reg         reg_bp_0_control_s;	// CSR.scala:239:19
  reg         reg_bp_0_control_u;	// CSR.scala:239:19
  reg         reg_bp_0_control_x;	// CSR.scala:239:19
  reg         reg_bp_0_control_w;	// CSR.scala:239:19
  reg         reg_bp_0_control_r;	// CSR.scala:239:19
  reg  [38:0] reg_bp_0_address;	// CSR.scala:239:19
  reg  [63:0] reg_mie;	// CSR.scala:241:20
  reg  [63:0] reg_mideleg;	// CSR.scala:242:24
  reg  [63:0] reg_medeleg;	// CSR.scala:243:24
  reg         reg_mip_meip;	// CSR.scala:244:20
  reg         reg_mip_seip;	// CSR.scala:244:20
  reg         reg_mip_mtip;	// CSR.scala:244:20
  reg         reg_mip_stip;	// CSR.scala:244:20
  reg         reg_mip_msip;	// CSR.scala:244:20
  reg         reg_mip_ssip;	// CSR.scala:244:20
  reg  [39:0] reg_mepc;	// CSR.scala:245:21
  reg  [63:0] reg_mcause;	// CSR.scala:246:23
  reg  [39:0] reg_mbadaddr;	// CSR.scala:247:25
  reg  [63:0] reg_mscratch;	// CSR.scala:248:25
  reg  [31:0] reg_mtvec;	// CSR.scala:251:27
  reg  [31:0] reg_mcounteren;	// CSR.scala:254:27
  reg  [31:0] reg_scounteren;	// CSR.scala:255:27
  reg  [39:0] reg_sepc;	// CSR.scala:258:21
  reg  [63:0] reg_scause;	// CSR.scala:259:23
  reg  [39:0] reg_sbadaddr;	// CSR.scala:260:25
  reg  [63:0] reg_sscratch;	// CSR.scala:261:25
  reg  [38:0] reg_stvec;	// CSR.scala:262:22
  reg  [3:0]  reg_sptbr_mode;	// CSR.scala:263:22
  reg  [43:0] reg_sptbr_ppn;	// CSR.scala:263:22
  reg         reg_wfi;	// CSR.scala:264:20
  reg  [4:0]  reg_fflags;	// CSR.scala:266:23
  reg  [2:0]  reg_frm;	// CSR.scala:267:20
  reg  [5:0]  _T_0;	// Counters.scala:47:37
  reg  [57:0] _T_1;	// Counters.scala:52:27
  reg  [5:0]  _T_2_3;	// Counters.scala:47:37
  reg  [57:0] _T_3;	// Counters.scala:52:27
  reg  [63:0] reg_misa;	// CSR.scala:307:21

  wire [2:0] _effective_prv = {reg_debug, reg_mstatus_prv};	// Cat.scala:30:58
  wire [63:0] _T_2 = {_T_1, _T_0};	// Cat.scala:30:58, Counters.scala:48:33, :53:43
  wire [63:0] _T_4 = {_T_3, _T_2_3};	// Cat.scala:30:58, Counters.scala:48:33, :53:43
  wire _T_5 = reg_mstatus_prv == 2'h1;	// CSR.scala:222:10, :274:71, Cat.scala:30:58
  wire [63:0] _T_6 = {52'h0, reg_mip_meip, 1'h0, reg_mip_seip, 1'h0, reg_mip_mtip, 1'h0, reg_mip_stip, 1'h0,
                reg_mip_msip, 1'h0, reg_mip_ssip, 1'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, :2228:9, CSR.scala:280:37
  wire [63:0] _pending_interrupts = _T_6 & reg_mie;	// CSR.scala:280:37
  wire _reg_mstatus_prv_1 = reg_mstatus_prv[1];	// CSR.scala:281:42, Cat.scala:30:58
  wire [63:0] _m_interrupts = ~_reg_mstatus_prv_1 | &reg_mstatus_prv & reg_mstatus_mie ? _pending_interrupts &
                ~reg_mideleg : 64'h0;	// CSR.scala:281:{25,42,51,71,81,121,123}, Cat.scala:30:58
  wire [63:0] _all_interrupts = _m_interrupts | ({_m_interrupts[11], _m_interrupts[9], _m_interrupts[7], _m_interrupts[5],
                _m_interrupts[3], _m_interrupts[1]} == 6'h0 & (reg_mstatus_prv == 2'h0 | _T_5 &
                reg_mstatus_sie) ? _pending_interrupts & reg_mideleg : 64'h0);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:281:{25,123}, :282:{25,39,45,65,73,103,144}, :283:37, Cat.scala:30:58, Counters.scala:47:37
  wire _T_7 = reg_dcsr_debugint & ~reg_debug;	// CSR.scala:286:41, :291:47, Cat.scala:30:58
  `ifndef SYNTHESIS	// CSR.scala:197:24
    `ifdef RANDOMIZE_REG_INIT	// CSR.scala:197:24
      reg [31:0] _RANDOM;	// CSR.scala:197:24
      reg [31:0] _RANDOM_4;	// CSR.scala:234:20
      reg [31:0] _RANDOM_5;	// CSR.scala:234:20
      reg [31:0] _RANDOM_6;	// CSR.scala:235:25
      reg [31:0] _RANDOM_7;	// CSR.scala:235:25
      reg [31:0] _RANDOM_8;	// CSR.scala:239:19
      reg [31:0] _RANDOM_9;	// CSR.scala:241:20
      reg [31:0] _RANDOM_10;	// CSR.scala:241:20
      reg [31:0] _RANDOM_11;	// CSR.scala:242:24
      reg [31:0] _RANDOM_12;	// CSR.scala:242:24
      reg [31:0] _RANDOM_13;	// CSR.scala:243:24
      reg [31:0] _RANDOM_14;	// CSR.scala:243:24
      reg [31:0] _RANDOM_15;	// CSR.scala:245:21
      reg [31:0] _RANDOM_16;	// CSR.scala:245:21
      reg [31:0] _RANDOM_17;	// CSR.scala:246:23
      reg [31:0] _RANDOM_18;	// CSR.scala:246:23
      reg [31:0] _RANDOM_19;	// CSR.scala:247:25
      reg [31:0] _RANDOM_20;	// CSR.scala:248:25
      reg [31:0] _RANDOM_21;	// CSR.scala:248:25
      reg [31:0] _RANDOM_22;	// CSR.scala:251:27
      reg [31:0] _RANDOM_23;	// CSR.scala:254:27
      reg [31:0] _RANDOM_24;	// CSR.scala:255:27
      reg [31:0] _RANDOM_25;	// CSR.scala:258:21
      reg [31:0] _RANDOM_26;	// CSR.scala:259:23
      reg [31:0] _RANDOM_27;	// CSR.scala:259:23
      reg [31:0] _RANDOM_28;	// CSR.scala:260:25
      reg [31:0] _RANDOM_29;	// CSR.scala:261:25
      reg [31:0] _RANDOM_30;	// CSR.scala:261:25
      reg [31:0] _RANDOM_31;	// CSR.scala:262:22
      reg [31:0] _RANDOM_32;	// CSR.scala:262:22
      reg [31:0] _RANDOM_33;	// CSR.scala:263:22
      reg [31:0] _RANDOM_34;	// Counters.scala:47:37
      reg [31:0] _RANDOM_35;	// Counters.scala:52:27
      reg [31:0] _RANDOM_36;	// Counters.scala:47:37
      reg [31:0] _RANDOM_37;	// Counters.scala:52:27
      reg [31:0] _RANDOM_38;	// CSR.scala:307:21
      reg [31:0] _RANDOM_39;	// CSR.scala:307:21

    `endif
    initial begin	// CSR.scala:197:24
      `INIT_RANDOM_PROLOG_	// CSR.scala:197:24
      `ifdef RANDOMIZE_REG_INIT	// CSR.scala:197:24
        _RANDOM = `RANDOM;	// CSR.scala:197:24
        reg_mstatus_prv = _RANDOM[1:0];	// CSR.scala:197:24
        reg_mstatus_tsr = _RANDOM[2];	// CSR.scala:197:24
        reg_mstatus_tw = _RANDOM[3];	// CSR.scala:197:24
        reg_mstatus_tvm = _RANDOM[4];	// CSR.scala:197:24
        reg_mstatus_mxr = _RANDOM[5];	// CSR.scala:197:24
        reg_mstatus_pum = _RANDOM[6];	// CSR.scala:197:24
        reg_mstatus_mprv = _RANDOM[7];	// CSR.scala:197:24
        reg_mstatus_fs = _RANDOM[9:8];	// CSR.scala:197:24
        reg_mstatus_mpp = _RANDOM[11:10];	// CSR.scala:197:24
        reg_mstatus_spp = _RANDOM[12];	// CSR.scala:197:24
        reg_mstatus_mpie = _RANDOM[13];	// CSR.scala:197:24
        reg_mstatus_spie = _RANDOM[14];	// CSR.scala:197:24
        reg_mstatus_mie = _RANDOM[15];	// CSR.scala:197:24
        reg_mstatus_sie = _RANDOM[16];	// CSR.scala:197:24
        reg_dcsr_ebreakm = _RANDOM[17];	// CSR.scala:205:21
        reg_dcsr_ebreaks = _RANDOM[18];	// CSR.scala:205:21
        reg_dcsr_ebreaku = _RANDOM[19];	// CSR.scala:205:21
        reg_dcsr_cause = _RANDOM[22:20];	// CSR.scala:205:21
        reg_dcsr_debugint = _RANDOM[23];	// CSR.scala:205:21
        reg_dcsr_halt = _RANDOM[24];	// CSR.scala:205:21
        reg_dcsr_step = _RANDOM[25];	// CSR.scala:205:21
        reg_dcsr_prv = _RANDOM[27:26];	// CSR.scala:205:21
        reg_debug = _RANDOM[28];	// CSR.scala:232:22
        _RANDOM_4 = `RANDOM;	// CSR.scala:234:20
        _RANDOM_5 = `RANDOM;	// CSR.scala:234:20
        reg_dpc = {_RANDOM_5[4:0], _RANDOM_4, _RANDOM[31:29]};	// CSR.scala:234:20
        _RANDOM_6 = `RANDOM;	// CSR.scala:235:25
        _RANDOM_7 = `RANDOM;	// CSR.scala:235:25
        reg_dscratch = {_RANDOM_7[4:0], _RANDOM_6, _RANDOM_5[31:5]};	// CSR.scala:235:25
        reg_singleStepped = _RANDOM_7[5];	// CSR.scala:236:30
        reg_bp_0_control_dmode = _RANDOM_7[6];	// CSR.scala:239:19
        reg_bp_0_control_action = _RANDOM_7[7];	// CSR.scala:239:19
        reg_bp_0_control_tmatch = _RANDOM_7[9:8];	// CSR.scala:239:19
        reg_bp_0_control_m = _RANDOM_7[10];	// CSR.scala:239:19
        reg_bp_0_control_s = _RANDOM_7[11];	// CSR.scala:239:19
        reg_bp_0_control_u = _RANDOM_7[12];	// CSR.scala:239:19
        reg_bp_0_control_x = _RANDOM_7[13];	// CSR.scala:239:19
        reg_bp_0_control_w = _RANDOM_7[14];	// CSR.scala:239:19
        reg_bp_0_control_r = _RANDOM_7[15];	// CSR.scala:239:19
        _RANDOM_8 = `RANDOM;	// CSR.scala:239:19
        reg_bp_0_address = {_RANDOM_8[22:0], _RANDOM_7[31:16]};	// CSR.scala:239:19
        _RANDOM_9 = `RANDOM;	// CSR.scala:241:20
        _RANDOM_10 = `RANDOM;	// CSR.scala:241:20
        reg_mie = {_RANDOM_10[22:0], _RANDOM_9, _RANDOM_8[31:23]};	// CSR.scala:241:20
        _RANDOM_11 = `RANDOM;	// CSR.scala:242:24
        _RANDOM_12 = `RANDOM;	// CSR.scala:242:24
        reg_mideleg = {_RANDOM_12[22:0], _RANDOM_11, _RANDOM_10[31:23]};	// CSR.scala:242:24
        _RANDOM_13 = `RANDOM;	// CSR.scala:243:24
        _RANDOM_14 = `RANDOM;	// CSR.scala:243:24
        reg_medeleg = {_RANDOM_14[22:0], _RANDOM_13, _RANDOM_12[31:23]};	// CSR.scala:243:24
        reg_mip_meip = _RANDOM_14[23];	// CSR.scala:244:20
        reg_mip_seip = _RANDOM_14[24];	// CSR.scala:244:20
        reg_mip_mtip = _RANDOM_14[25];	// CSR.scala:244:20
        reg_mip_stip = _RANDOM_14[26];	// CSR.scala:244:20
        reg_mip_msip = _RANDOM_14[27];	// CSR.scala:244:20
        reg_mip_ssip = _RANDOM_14[28];	// CSR.scala:244:20
        _RANDOM_15 = `RANDOM;	// CSR.scala:245:21
        _RANDOM_16 = `RANDOM;	// CSR.scala:245:21
        reg_mepc = {_RANDOM_16[4:0], _RANDOM_15, _RANDOM_14[31:29]};	// CSR.scala:245:21
        _RANDOM_17 = `RANDOM;	// CSR.scala:246:23
        _RANDOM_18 = `RANDOM;	// CSR.scala:246:23
        reg_mcause = {_RANDOM_18[4:0], _RANDOM_17, _RANDOM_16[31:5]};	// CSR.scala:246:23
        _RANDOM_19 = `RANDOM;	// CSR.scala:247:25
        reg_mbadaddr = {_RANDOM_19[12:0], _RANDOM_18[31:5]};	// CSR.scala:247:25
        _RANDOM_20 = `RANDOM;	// CSR.scala:248:25
        _RANDOM_21 = `RANDOM;	// CSR.scala:248:25
        reg_mscratch = {_RANDOM_21[12:0], _RANDOM_20, _RANDOM_19[31:13]};	// CSR.scala:248:25
        _RANDOM_22 = `RANDOM;	// CSR.scala:251:27
        reg_mtvec = {_RANDOM_22[12:0], _RANDOM_21[31:13]};	// CSR.scala:251:27
        _RANDOM_23 = `RANDOM;	// CSR.scala:254:27
        reg_mcounteren = {_RANDOM_23[12:0], _RANDOM_22[31:13]};	// CSR.scala:254:27
        _RANDOM_24 = `RANDOM;	// CSR.scala:255:27
        reg_scounteren = {_RANDOM_24[12:0], _RANDOM_23[31:13]};	// CSR.scala:255:27
        _RANDOM_25 = `RANDOM;	// CSR.scala:258:21
        reg_sepc = {_RANDOM_25[20:0], _RANDOM_24[31:13]};	// CSR.scala:258:21
        _RANDOM_26 = `RANDOM;	// CSR.scala:259:23
        _RANDOM_27 = `RANDOM;	// CSR.scala:259:23
        reg_scause = {_RANDOM_27[20:0], _RANDOM_26, _RANDOM_25[31:21]};	// CSR.scala:259:23
        _RANDOM_28 = `RANDOM;	// CSR.scala:260:25
        reg_sbadaddr = {_RANDOM_28[28:0], _RANDOM_27[31:21]};	// CSR.scala:260:25
        _RANDOM_29 = `RANDOM;	// CSR.scala:261:25
        _RANDOM_30 = `RANDOM;	// CSR.scala:261:25
        reg_sscratch = {_RANDOM_30[28:0], _RANDOM_29, _RANDOM_28[31:29]};	// CSR.scala:261:25
        _RANDOM_31 = `RANDOM;	// CSR.scala:262:22
        _RANDOM_32 = `RANDOM;	// CSR.scala:262:22
        reg_stvec = {_RANDOM_32[3:0], _RANDOM_31, _RANDOM_30[31:29]};	// CSR.scala:262:22
        reg_sptbr_mode = _RANDOM_32[7:4];	// CSR.scala:263:22
        _RANDOM_33 = `RANDOM;	// CSR.scala:263:22
        reg_sptbr_ppn = {_RANDOM_33[19:0], _RANDOM_32[31:8]};	// CSR.scala:263:22
        reg_wfi = _RANDOM_33[20];	// CSR.scala:264:20
        reg_fflags = _RANDOM_33[25:21];	// CSR.scala:266:23
        reg_frm = _RANDOM_33[28:26];	// CSR.scala:267:20
        _RANDOM_34 = `RANDOM;	// Counters.scala:47:37
        _T_0 = {_RANDOM_34[2:0], _RANDOM_33[31:29]};	// Counters.scala:47:37
        _RANDOM_35 = `RANDOM;	// Counters.scala:52:27
        _T_1 = {_RANDOM_35[28:0], _RANDOM_34[31:3]};	// Counters.scala:52:27
        _RANDOM_36 = `RANDOM;	// Counters.scala:47:37
        _T_2 = {_RANDOM_36[2:0], _RANDOM_35[31:29]};	// Counters.scala:47:37
        _RANDOM_37 = `RANDOM;	// Counters.scala:52:27
        _T_3 = {_RANDOM_37[28:0], _RANDOM_36[31:3]};	// Counters.scala:52:27
        _RANDOM_38 = `RANDOM;	// CSR.scala:307:21
        _RANDOM_39 = `RANDOM;	// CSR.scala:307:21
        reg_misa = {_RANDOM_39[28:0], _RANDOM_38, _RANDOM_37[31:29]};	// CSR.scala:307:21
      `endif
    end // initial
  `endif
      wire [63:0] _T_8 = {&reg_mstatus_fs, 40'h1400, reg_mstatus_tsr, reg_mstatus_tw, reg_mstatus_tvm,
                reg_mstatus_mxr, reg_mstatus_pum, reg_mstatus_mprv, 2'h0, reg_mstatus_fs, reg_mstatus_mpp,
                2'h0, reg_mstatus_spp, reg_mstatus_mpie, 1'h0, reg_mstatus_spie, 1'h0, reg_mstatus_mie,
                1'h0, reg_mstatus_sie, 1'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:281:81, :282:103, :308:{38,40}, :418:62, :419:69, :420:63, :452:13, :453:32
      wire _reg_stvec_38 = reg_stvec[38];	// Package.scala:40:38
      wire _T_9 = io_rw_addr == 12'h7A1;	// CSR.scala:405:73
      wire _T_10 = io_rw_addr == 12'h7A2;	// CSR.scala:405:73
      wire _T_11 = io_rw_addr == 12'hB00;	// CSR.scala:405:73
      wire _T_12 = io_rw_addr == 12'hB02;	// CSR.scala:405:73
      wire _T_13 = io_rw_addr == 12'h301;	// CSR.scala:405:73
      wire _T_14 = io_rw_addr == 12'h300;	// CSR.scala:405:73
      wire _T_15 = io_rw_addr == 12'h305;	// CSR.scala:405:73
      wire _T_16 = io_rw_addr == 12'h344;	// CSR.scala:405:73
      wire _T_17 = io_rw_addr == 12'h304;	// CSR.scala:405:73
      wire _T_18 = io_rw_addr == 12'h303;	// CSR.scala:405:73
      wire _T_19 = io_rw_addr == 12'h302;	// CSR.scala:405:73
      wire _T_20 = io_rw_addr == 12'h340;	// CSR.scala:405:73
      wire _T_21 = io_rw_addr == 12'h341;	// CSR.scala:405:73
      wire _T_22 = io_rw_addr == 12'h343;	// CSR.scala:405:73
      wire _T_23 = io_rw_addr == 12'h342;	// CSR.scala:405:73
      wire _T_24 = io_rw_addr == 12'h7B0;	// CSR.scala:405:73
      wire _T_25 = io_rw_addr == 12'h7B1;	// CSR.scala:405:73
      wire _T_26 = io_rw_addr == 12'h7B2;	// CSR.scala:405:73
      wire _T_27 = io_rw_addr == 12'h1;	// CSR.scala:405:73
      wire _T_28 = io_rw_addr == 12'h2;	// CSR.scala:405:73
      wire _T_29 = io_rw_addr == 12'h3;	// CSR.scala:405:73
      wire _T_30 = io_rw_addr == 12'h100;	// CSR.scala:405:73
      wire _T_31 = io_rw_addr == 12'h144;	// CSR.scala:405:73
      wire _T_32 = io_rw_addr == 12'h104;	// CSR.scala:405:73
      wire _T_33 = io_rw_addr == 12'h140;	// CSR.scala:405:73
      wire _T_34 = io_rw_addr == 12'h142;	// CSR.scala:405:73
      wire _T_35 = io_rw_addr == 12'h143;	// CSR.scala:405:73
      wire _T_36 = io_rw_addr == 12'h180;	// CSR.scala:405:73
      wire _T_37 = io_rw_addr == 12'h141;	// CSR.scala:405:73
      wire _T_38 = io_rw_addr == 12'h105;	// CSR.scala:405:73
      wire _T_39 = io_rw_addr == 12'h106;	// CSR.scala:405:73
      wire _T_40 = io_rw_addr == 12'h306;	// CSR.scala:405:73
      wire _system_insn = io_rw_cmd == 3'h4;	// CSR.scala:409:31, Mux.scala:31:69
      wire [2:0] _io_rw_addr_2to0 = io_rw_addr[2:0];	// CSR.scala:410:37
      wire _insn_rs2 = io_rw_addr[5];	// CSR.scala:411:28
      wire _T_41 = _system_insn & ~_insn_rs2 & _io_rw_addr_2to0 == 3'h0;	// CSR.scala:412:{34,44,53}, Mux.scala:19:72
      wire _insn_break = _system_insn & _io_rw_addr_2to0 == 3'h1;	// CSR.scala:413:{32,41}, Package.scala:7:47
      wire _insn_ret = _system_insn & _io_rw_addr_2to0 == 3'h2;	// CSR.scala:414:{30,39}, Package.scala:7:47
      wire [1:0] _T_42 = {reg_debug, reg_mstatus_prv[1]};	// CSR.scala:418:51, Cat.scala:30:58
      wire _allow_sfence_vma = |_T_42 | ~reg_mstatus_tvm;	// CSR.scala:418:51, :419:{66,69}
      wire _T_43 = reg_mstatus_fs == 2'h0 | ~(reg_misa[5]);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:421:{40,46,49,58}, :452:13
      wire _T_44 = _effective_prv < {1'h0, io_decode_csr[9:8]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:423:{43,58}
      wire _T_45 = io_decode_csr == 12'h7B0;	// CSR.scala:405:73, :424:42
      wire _T_46 = io_decode_csr == 12'h7B1;	// CSR.scala:405:73, :424:42
      wire _T_47 = io_decode_csr == 12'h7B2;	// CSR.scala:405:73, :424:42
      wire _T_48 = io_decode_csr == 12'h1;	// CSR.scala:405:73, :424:42
      wire _T_49 = io_decode_csr == 12'h2;	// CSR.scala:405:73, :424:42
      wire _T_50 = io_decode_csr == 12'h3;	// CSR.scala:405:73, :424:42
      wire _T_51 = io_decode_csr == 12'h180;	// CSR.scala:405:73, :424:42
      wire [31:0] _T_52 = (reg_mcounteren & (_T_5 ? 32'h7 : reg_scounteren)) >> io_decode_csr;	// CSR.scala:274:{33,38}, :426:171
      wire _io_decode_csr_5 = io_decode_csr[5];	// CSR.scala:432:19
      wire [63:0] _cause = _T_41 ? {60'h0, {2'h0, reg_mstatus_prv} - 4'h8} : _insn_break ? 64'h3 : io_cause;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:437:{8,36}, :438:14, Cat.scala:30:58
      wire [5:0] _cause_lsbs = _cause[5:0];	// CSR.scala:439:25
      wire _cause_63 = _cause[63];	// CSR.scala:440:30
      wire _T_53 = _cause_lsbs == 6'hD;	// CSR.scala:440:53, Mux.scala:31:69
      wire _causeIsDebugInt = _cause_63 & _T_53;	// CSR.scala:440:39
      wire _causeIsDebugTrigger = ~_cause_63 & _T_53;	// CSR.scala:441:{29,44}
      wire [3:0] _T_54 = {reg_dcsr_ebreakm, 1'h0, reg_dcsr_ebreaks, reg_dcsr_ebreaku} >> reg_mstatus_prv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:333:27, :442:134, Cat.scala:30:58
      wire _T_55 = reg_singleStepped | _causeIsDebugInt | _causeIsDebugTrigger | ~_cause_63 & _insn_break &
                _T_54[0] | reg_debug;	// CSR.scala:286:70, :441:29, :442:{56,134}, :443:123, Cat.scala:30:58
      wire [63:0] _T_56 = {58'h0, _cause_lsbs};	// CSR.scala:444:93, Counters.scala:52:27
      wire [63:0] _T_57 = reg_mideleg >> _T_56;	// CSR.scala:281:123, :444:93
      wire [63:0] _T_58 = reg_medeleg >> _T_56;	// CSR.scala:444:118
      wire _delegate = ~_reg_mstatus_prv_1 & (_cause_63 ? _T_57[0] : _T_58[0]);	// CSR.scala:281:42, :444:{60,66,93,118}
      wire [39:0] _tvec = _T_55 ? {36'h80, reg_debug, 3'h0} : _delegate ? {_reg_stvec_38, reg_stvec} : {8'h0,
                reg_mtvec};	// CSR.scala:446:{17,45}, Cat.scala:30:58, Mux.scala:19:72, Package.scala:40:38
      wire _T_59 = _T_41 | _insn_break;	// CSR.scala:450:24
  assign _T = reg_dcsr_step & ~reg_debug;	// CSR.scala:286:41, :333:27, :451:34, Cat.scala:30:58
      wire _exception = _T_59 | io_exception;	// CSR.scala:461:43
      wire _T_60 = ~_T_55 & _delegate;	// CSR.scala:481:24, :486:27
      wire _T_61 = ~_T_55 & ~_delegate;	// CSR.scala:481:24, :486:27
      wire [1:0] _T_62 = _exception ? (_T_61 ? 2'h3 : _T_60 ? 2'h1 : reg_mstatus_prv) : reg_mstatus_prv;	// CSR.scala:195:21, :222:10, :493:15, :501:15, Cat.scala:30:58
      wire _io_rw_addr_9 = io_rw_addr[9];	// CSR.scala:506:39
      wire _io_rw_addr_10 = io_rw_addr[10];	// CSR.scala:512:47
      wire _T_63 = _io_rw_addr_9 & _io_rw_addr_10;	// CSR.scala:512:53
      wire _T_64 = _io_rw_addr_9 & ~_io_rw_addr_10;	// CSR.scala:512:53
      wire [1:0] _T_65 = _insn_ret ? (_T_64 ? reg_mstatus_mpp : _T_63 ? reg_dcsr_prv : _io_rw_addr_9 ? _T_62 :
                {1'h0, reg_mstatus_spp}) : _T_62;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:333:27, :452:13, :510:15, :513:15, :521:15
      wire [31:0] _T_66 = _T_15 ? reg_mtvec : 32'h0;	// CSR.scala:251:27, :446:45, Mux.scala:19:72
      wire [31:0] _T_67 = _T_24 ? {16'h4000, reg_dcsr_ebreakm, 1'h0, reg_dcsr_ebreaks, reg_dcsr_ebreaku, 3'h0,
                reg_dcsr_cause, reg_dcsr_debugint, 1'h0, reg_dcsr_halt, reg_dcsr_step, reg_dcsr_prv} :
                32'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:251:27, :291:47, :333:27, Mux.scala:19:72
      wire [39:0] _T_68 = _T_25 ? reg_dpc : 40'h0;	// CSR.scala:483:15, Mux.scala:19:72
      wire [7:0] _T_69 = _T_29 ? {reg_frm, reg_fflags} : 8'h0;	// Cat.scala:30:58, Mux.scala:19:72
      wire [31:0] _T_70 = _T_39 ? reg_scounteren : 32'h0;	// CSR.scala:251:27, :274:38, Mux.scala:19:72
      wire [31:0] _T_71 = _T_40 ? reg_mcounteren : 32'h0;	// CSR.scala:251:27, :274:33, Mux.scala:19:72
      wire [12:0] _T_72 = _T_66[12:0] | (_T_16 ? {1'h0, reg_mip_meip, 1'h0, reg_mip_seip, 1'h0, reg_mip_mtip, 1'h0,
                reg_mip_stip, 1'h0, reg_mip_msip, 1'h0, reg_mip_ssip, 1'h0} : 13'h0) | _T_67[12:0] |
                _T_68[12:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, :2228:9, CSR.scala:278:22, Mux.scala:19:72
      wire [4:0] _T_73 = _T_72[4:0] | (_T_27 ? reg_fflags : 5'h0);	// Cat.scala:30:58, Mux.scala:19:72
      wire [63:0] _T_74 = (_T_9 ? {4'h2, reg_bp_0_control_dmode, 46'h40000000000, reg_bp_0_control_action, 3'h0,
                reg_bp_0_control_tmatch, reg_bp_0_control_m, 1'h0, reg_bp_0_control_s, reg_bp_0_control_u,
                reg_bp_0_control_x, reg_bp_0_control_w, reg_bp_0_control_r} : 64'h0) | (_T_10 ?
                {{25{reg_bp_0_address[38]}}, reg_bp_0_address} : 64'h0) | (_T_11 ? _T_4 : 64'h0) | (_T_12 ?
                _T_2 : 64'h0) | (_T_13 ? reg_misa : 64'h0) | (_T_14 ? _T_8 : 64'h0) | (_T_17 ? reg_mie :
                64'h0) | (_T_18 ? reg_mideleg : 64'h0) | (_T_19 ? reg_medeleg : 64'h0) | (_T_20 ?
                reg_mscratch : 64'h0) | (_T_21 ? {{24{reg_mepc[39]}}, reg_mepc} : 64'h0) | (_T_22 ?
                {{24{reg_mbadaddr[39]}}, reg_mbadaddr} : 64'h0) | (_T_23 ? reg_mcause : 64'h0) |
                (io_rw_addr == 12'hF14 ? io_hartid : 64'h0) | (_T_26 ? reg_dscratch : 64'h0) | (_T_30 ?
                {&reg_mstatus_fs, 40'h1400, reg_mstatus_tsr, reg_mstatus_tw, reg_mstatus_tvm,
                reg_mstatus_mxr, reg_mstatus_pum, 3'h0, reg_mstatus_fs, 4'h0, reg_mstatus_spp, 2'h0,
                reg_mstatus_spie, 3'h0, reg_mstatus_sie, 1'h0} : 64'h0) | (_T_31 ? _T_6 & reg_mideleg :
                64'h0) | (_T_32 ? reg_mie & reg_mideleg : 64'h0) | (_T_33 ? reg_sscratch : 64'h0) | (_T_34
                ? reg_scause : 64'h0) | (_T_35 ? {{24{reg_sbadaddr[39]}}, reg_sbadaddr} : 64'h0) | (_T_36 ?
                {reg_sptbr_mode, 16'h0, reg_sptbr_ppn} : 64'h0) | (_T_37 ? {{24{reg_sepc[39]}}, reg_sepc} :
                64'h0) | (_T_38 ? {{25{_reg_stvec_38}}, reg_stvec} : 64'h0) | {24'h0, _T_68[39:32],
                _T_66[31:13] | _T_67[31:13] | _T_68[31:13] | _T_70[31:13] | _T_71[31:13], _T_72[12:8] |
                _T_70[12:8] | _T_71[12:8], _T_72[7:5] | _T_69[7:5] | _T_70[7:5] | _T_71[7:5], _T_73[4:3] |
                _T_69[4:3] | _T_70[4:3] | _T_71[4:3], _T_73[2:0] | (_T_28 ? reg_frm : 3'h0) | _T_69[2:0] |
                _T_70[2:0] | _T_71[2:0]} | (io_rw_addr == 12'hC00 ? _T_4 : 64'h0) | (io_rw_addr == 12'hC02
                ? _T_2 : 64'h0);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Bitwise.scala:71:12, CSR.scala:239:19, :263:22, :280:37, :281:{25,123}, :282:103, :308:38, :312:48, :360:28, :361:29, :371:60, :377:45, :405:73, :418:62, :419:69, :420:63, :421:58, :444:118, :452:13, :453:32, :488:18, :496:18, :622:54, Cat.scala:30:58, Mux.scala:19:72, Package.scala:40:38
      wire _T_75 = {1'h0, {1'h0, _insn_ret} + {1'h0, _T_41}} + {1'h0, {1'h0, _insn_break} + {1'h0,
                io_exception}} < 3'h2 | reset;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Bitwise.scala:48:55, CSR.scala:462:{9,79}, Package.scala:7:47
      wire _T_76 = ~reg_wfi | ~io_retire | reset;	// CSR.scala:464:29, :466:{9,10,32}
      wire _T_77 = ~reg_singleStepped | ~io_retire | reset;	// CSR.scala:286:70, :466:32, :471:{9,10}
      wire [6:0] _T_78 = {1'h0, _T_0} + {6'h0, io_retire};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Counters.scala:47:37, :48:33
      wire [6:0] _T_79 = {1'h0, _T_2_3} + 7'h1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Counters.scala:48:33
      wire [63:0] _T_80 = _T_8 >> reg_mstatus_prv;	// CSR.scala:475:27, Cat.scala:30:58
      wire _T_81 = _T_80[0];	// CSR.scala:475:27
      wire _T_82 = (~_exception | ~_T_60) & reg_mstatus_sie;	// CSR.scala:282:103, :492:23
      wire _T_83 = _insn_ret & ~_io_rw_addr_9 | (_exception & _T_60 ? _T_81 : reg_mstatus_spie);	// CSR.scala:452:13, :490:24, :506:28, :508:24
      wire _T_84 = (~_insn_ret | _io_rw_addr_9) & (_exception & _T_60 ? reg_mstatus_prv[0] : reg_mstatus_spp);	// CSR.scala:452:13, :491:23, :509:23, Cat.scala:30:58
      wire _reg_mstatus_mpp_1 = reg_mstatus_mpp[1];	// CSR.scala:452:13, :517:28
      wire _T_85 = _insn_ret ? (_T_64 & ~_reg_mstatus_mpp_1 & reg_mstatus_mpp[0] ? reg_mstatus_mpie :
                _io_rw_addr_9 | ~reg_mstatus_spp ? _T_82 : reg_mstatus_spie) : _T_82;	// CSR.scala:452:13, :507:55, :517:33, :518:{50,73}
      wire _T_86 = io_rw_cmd == 3'h3;	// Package.scala:7:47
      wire _T_87 = io_rw_cmd == 3'h2 | _T_86;	// Package.scala:7:{47,62}
      wire [63:0] _wdata = ((_T_87 ? _T_74 : 64'h0) | io_rw_wdata) & ~(_T_86 ? io_rw_wdata : 64'h0);	// CSR.scala:281:25, :406:{19,75,90}, :407:{15,19}
      wire [39:0] _T_88 = ~io_pc | 40'h1;	// CSR.scala:474:{17,24}
      wire _T_89 = _cause == 64'h3 | _cause == 64'h4 | _cause == 64'h6 | _cause == 64'h0 | _cause == 64'h5 |
                _cause == 64'h7 | _cause == 64'h1;	// CSR.scala:281:25, :438:14, Package.scala:7:{47,62}
      wire _reg_misa_2 = reg_misa[2];	// CSR.scala:421:58, :714:46
      wire [39:0] _T_90 = _T_88 | {38'h0, ~_reg_misa_2, 1'h1};	// CSR.scala:203:24, :714:{31,37}
      wire _T_91 = _T_87 | io_rw_cmd == 3'h1;	// Package.scala:7:{47,62}
      wire [63:0] _T_92 = ~_wdata;	// CSR.scala:563:21
      wire [39:0] _T_93 = _T_92[39:0] | {38'h0, ~_reg_misa_2, 1'h1};	// CSR.scala:203:24, :714:{31,37}
      wire [63:0] _T_94 = _wdata & 64'h800000000000001F;	// CSR.scala:577:62
      wire [39:0] _wdata_39to0 = _wdata[39:0];	// CSR.scala:578:63
      wire [3:0] _wdata_63to60 = _wdata[63:60];	// CSR.scala:620:44
      wire _T_95 = _wdata_63to60 == 4'h0;	// CSR.scala:622:{30,54}
      wire _T_96 = _wdata_63to60 == 4'h8;	// CSR.scala:623:30, Mux.scala:31:69
      wire [31:0] _T_97 = {29'h0, _wdata[2:0]};	// CSR.scala:637:70
      wire _T_98 = ~reg_bp_0_control_dmode | reg_debug;	// CSR.scala:312:48, :646:{13,31}, Cat.scala:30:58
      wire [5:0] _wdata_5to0 = _wdata[5:0];	// Counters.scala:67:11
      wire [57:0] _wdata_63to6 = _wdata[63:6];	// Counters.scala:68:28
      wire _T_99 = _wdata[59] & reg_debug;	// CSR.scala:648:48, :649:36, Cat.scala:30:58
  always @(posedge clock) begin	// CSR.scala:197:24
    `ifndef SYNTHESIS	// CSR.scala:462:9
      if (`PRINTF_COND_ & ~_T_75)	// CSR.scala:462:9
        $fwrite(32'h80000002, "Assertion failed: these conditions must be mutually exclusive\n    at CSR.scala:462 assert(PopCount(insn_ret :: insn_call :: insn_break :: io.exception :: Nil) <= 1, \"these conditions must be mutually exclusive\")\n");	// CSR.scala:462:9
      if (`STOP_COND_ & ~_T_75)	// CSR.scala:462:9
        $fatal;	// CSR.scala:462:9
      if (`PRINTF_COND_ & ~_T_76)	// CSR.scala:462:9, :466:9
        $fwrite(32'h80000002, "Assertion failed\n    at CSR.scala:466 assert(!reg_wfi || io.retire === UInt(0))\n");	// CSR.scala:466:9
      if (`STOP_COND_ & ~_T_76)	// CSR.scala:462:9, :466:9
        $fatal;	// CSR.scala:466:9
    `endif
    reg_singleStepped <= _T & (io_retire | reg_singleStepped);	// CSR.scala:286:70, :451:34, :468:43, :469:45
    `ifndef SYNTHESIS	// CSR.scala:470:9
      if (`PRINTF_COND_ & ~_T_77)	// CSR.scala:471:9
        $fwrite(32'h80000002, "Assertion failed\n    at CSR.scala:471 assert(!reg_singleStepped || io.retire === UInt(0))\n");	// CSR.scala:471:9
      if (`STOP_COND_ & ~_T_77)	// CSR.scala:471:9
        $fatal;	// CSR.scala:471:9
    `endif
    if (_T_91 & _T_16)	// CSR.scala:569:22
      reg_mip_stip <= _wdata[5];	// CSR.scala:566:39, :569:22
    reg_mepc <= _T_91 & _T_21 ? ~_T_93 : _exception & _T_61 ? ~_T_90 : reg_mepc;	// CSR.scala:495:16, :573:51, :714:26, Package.scala:40:38
    if (_T_91 & _T_20)	// CSR.scala:574:55
      reg_mscratch <= _wdata;	// CSR.scala:574:55
    reg_mcause <= _T_91 & _T_23 ? _T_94 : _exception & _T_61 ? _cause : reg_mcause;	// CSR.scala:496:18, :577:53
    reg_mbadaddr <= _T_91 & _T_22 ? _wdata_39to0 : _exception & _T_61 & _T_89 ? io_badaddr : reg_mbadaddr;	// CSR.scala:497:43, :578:55, Package.scala:40:38
    reg_fflags <= _T_91 & (_T_29 | _T_27) ? _wdata[4:0] : {5{io_fcsr_flags_valid}} & io_fcsr_flags_bits |
                                reg_fflags;	// CSR.scala:533:16, :588:53, :590:53, Cat.scala:30:58
    if (_T_91)	// CSR.scala:590:71
      reg_frm <= _T_29 ? _wdata[7:5] : _T_28 ? _wdata[2:0] : reg_frm;	// CSR.scala:589:50, :590:71, Cat.scala:30:58
    reg_dpc <= _T_91 & _T_25 ? ~(_T_92[39:0] | 40'h1) : _exception & _T_55 ? ~_T_88 : reg_dpc;	// CSR.scala:474:{15,24}, :483:15, :602:{52,55,64}
    if (_T_91 & _T_26)	// CSR.scala:603:57
      reg_dscratch <= _wdata;	// CSR.scala:603:57
    if (_T_91)	// CSR.scala:617:22
      reg_mip_ssip <= _T_31 ? _wdata[1] : _T_16 ? _wdata[1] : reg_mip_ssip;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:2228:9, CSR.scala:566:39, :568:22, :616:41, :617:22
    if (_T_91 & _T_36)	// CSR.scala:623:63
      reg_sptbr_mode <= _T_96 ? 4'h8 : _T_95 ? 4'h0 : reg_sptbr_mode;	// CSR.scala:377:45, :622:54, :623:63, Mux.scala:31:69
    if (_T_91 & _T_36 & (_T_95 | _T_96))	// CSR.scala:624:36, :625:25
      reg_sptbr_ppn <= {24'h0, _wdata[19:0]};	// Bitwise.scala:71:12, CSR.scala:625:{25,41}
    if (_T_91)	// CSR.scala:629:52
      reg_mie <= _T_32 ? reg_mie & ~reg_mideleg | _wdata & reg_mideleg : _T_17 ? _wdata & 64'hAAA : reg_mie;	// CSR.scala:280:37, :281:123, :572:{50,59}, :629:{52,64,80,89}
    if (_T_91 & _T_33)	// CSR.scala:630:57
      reg_sscratch <= _wdata;	// CSR.scala:630:57
    reg_sepc <= _T_91 & _T_37 ? ~_T_93 : _exception & _T_60 ? ~_T_90 : reg_sepc;	// CSR.scala:487:16, :631:53, :714:26, Package.scala:40:38
    if (_T_91 & _T_38)	// CSR.scala:632:54
      reg_stvec <= {_wdata[38:2], 2'h0};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:632:54
    reg_scause <= _T_91 & _T_34 ? _T_94 : _exception & _T_60 ? _cause : reg_scause;	// CSR.scala:488:18, :633:55
    reg_sbadaddr <= _T_91 & _T_35 ? _wdata_39to0 : _exception & _T_60 & _T_89 ? io_badaddr : reg_sbadaddr;	// CSR.scala:489:43, :634:57, Package.scala:40:38
    if (_T_91 & _T_18)	// CSR.scala:635:56
      reg_mideleg <= _wdata & 64'h222;	// CSR.scala:635:{56,65}
    if (_T_91 & _T_19)	// CSR.scala:636:56
      reg_medeleg <= _wdata & 64'h1AB;	// CSR.scala:636:{56,65}
    if (_T_91 & _T_39)	// CSR.scala:637:61
      reg_scounteren <= _T_97;	// CSR.scala:637:61
    if (_T_91 & _T_40)	// CSR.scala:640:61
      reg_mcounteren <= _T_97;	// CSR.scala:640:61
    if (_T_91 & _T_98 & _T_9)	// CSR.scala:650:22
      reg_bp_0_control_tmatch <= _wdata[8:7];	// CSR.scala:648:48, :650:22
    if (_T_91 & _T_98 & _T_9)	// CSR.scala:650:22
      reg_bp_0_control_m <= _wdata[6];	// CSR.scala:648:48, :650:22
    if (_T_91 & _T_98 & _T_9)	// CSR.scala:650:22
      reg_bp_0_control_s <= _wdata[4];	// CSR.scala:648:48, :650:22
    if (_T_91 & _T_98 & _T_9)	// CSR.scala:650:22
      reg_bp_0_control_u <= _wdata[3];	// CSR.scala:648:48, :650:22
    if (_T_91 & _T_98 & _T_10)	// CSR.scala:654:55
      reg_bp_0_address <= _wdata[38:0];	// CSR.scala:654:55
    reg_mip_mtip <= io_interrupts_mtip;	// CSR.scala:659:11
    reg_mip_msip <= io_interrupts_msip;	// CSR.scala:659:11
    reg_mip_meip <= io_interrupts_meip;	// CSR.scala:659:11
    reg_mip_seip <= io_interrupts_seip;	// CSR.scala:659:11
    if (reset) begin	// CSR.scala:197:24
      reg_mstatus_prv <= 2'h3;	// CSR.scala:195:21, :197:24
      reg_mstatus_tsr <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_tw <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_tvm <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_mxr <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_pum <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_mprv <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_fs <= 2'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_mpp <= 2'h3;	// CSR.scala:195:21, :197:24
      reg_mstatus_spp <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_mpie <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_spie <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_mie <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_mstatus_sie <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:197:24
      reg_dcsr_ebreakm <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_ebreaks <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_ebreaku <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_cause <= 3'h0;	// CSR.scala:205:21, Mux.scala:19:72
      reg_dcsr_debugint <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_halt <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_step <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:205:21
      reg_dcsr_prv <= 2'h3;	// CSR.scala:195:21, :205:21
      reg_debug <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:232:22
      reg_bp_0_control_dmode <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:239:19
      reg_bp_0_control_action <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:239:19
      reg_bp_0_control_x <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:239:19
      reg_bp_0_control_w <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:239:19
      reg_bp_0_control_r <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:239:19
      reg_mtvec <= 32'h0;	// CSR.scala:251:27
      reg_wfi <= 1'h0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:264:20
      _T_0 <= 6'h0;	// Counters.scala:47:37
      _T_1 <= 58'h0;	// Counters.scala:52:27
      _T_2_3 <= 6'h0;	// Counters.scala:47:37
      _T_3 <= 58'h0;	// Counters.scala:52:27
      reg_misa <= 64'h800000000014112D;	// CSR.scala:307:21
    end
    else begin	// CSR.scala:197:24
      reg_mstatus_prv <= _T_65 == 2'h2 ? 2'h0 : _T_65;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:200:19, :697:{21,27}
      reg_wfi <= ~(|{_pending_interrupts[11], _pending_interrupts[9], _pending_interrupts[7],
                                                _pending_interrupts[5], _pending_interrupts[3], _pending_interrupts[1]} | _exception) &
                                                (_system_insn & _io_rw_addr_2to0 == 3'h5 | reg_wfi);	// CSR.scala:280:37, :410:24, :415:{30,39}, :464:29, :465:{28,32,56}
      reg_dcsr_cause <= _exception & _T_55 ? (reg_singleStepped ? 3'h4 : {1'h0, _causeIsDebugInt ? 2'h3 :
                                                _causeIsDebugTrigger ? 2'h2 : 2'h1}) : reg_dcsr_cause;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:195:21, :222:10, :286:70, :333:27, :484:{22,28,54,84}, :697:27, Mux.scala:31:69
      reg_debug <= (~_insn_ret | ~_T_63) & (_exception & _T_55 | reg_debug);	// CSR.scala:482:17, :514:17, Cat.scala:30:58
      reg_mstatus_mie <= _T_91 & _T_14 ? _wdata[3] : _insn_ret & _T_64 & _reg_mstatus_mpp_1 ? reg_mstatus_mpie :
                                                (~_exception | ~_T_61) & reg_mstatus_mie;	// CSR.scala:281:81, :452:13, :500:23, :517:51, :538:47, :539:23
      reg_mstatus_mpie <= _T_91 & _T_14 ? _wdata[7] : _insn_ret & _T_64 | (_exception & _T_61 ? _T_81 :
                                                reg_mstatus_mpie);	// CSR.scala:452:13, :498:24, :519:24, :538:47, :540:24
      reg_mstatus_mprv <= _T_91 & _T_14 ? _wdata[17] : reg_mstatus_mprv;	// CSR.scala:452:13, :538:47, :543:26
      reg_mstatus_mpp <= _T_91 & _T_14 ? _wdata[12:11] : _insn_ret & _T_64 ? 2'h0 : _exception & _T_61 ?
                                                reg_mstatus_prv : reg_mstatus_mpp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13, :499:23, :520:23, :538:47, :544:25, Cat.scala:30:58
      reg_mstatus_mxr <= _T_91 & _T_14 ? _wdata[19] : reg_mstatus_mxr;	// CSR.scala:452:13, :538:47, :545:25
      reg_mstatus_tw <= _T_91 & _T_14 ? _wdata[21] : reg_mstatus_tw;	// CSR.scala:418:62, :538:47, :551:26
      reg_mstatus_tvm <= _T_91 & _T_14 ? _wdata[20] : reg_mstatus_tvm;	// CSR.scala:419:69, :538:47, :552:27
      reg_mstatus_tsr <= _T_91 & _T_14 ? _wdata[22] : reg_mstatus_tsr;	// CSR.scala:420:63, :538:47, :553:27
      reg_misa <= _T_91 & _T_13 ? ~(_T_92 | {60'h0, ~(_wdata[5]), 3'h0}) & 64'h102D | reg_misa & 64'hFD2 :
                                                reg_misa;	// CSR.scala:421:58, :437:8, :562:20, :563:{16,19,28,31,51,58,69}, Mux.scala:19:72
      reg_mtvec <= _T_91 & _T_15 ? {_wdata[31:2], 2'h0} : reg_mtvec;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:446:45, :576:52
      _T_2_3 <= _T_91 & _T_11 ? _wdata_5to0 : _T_79[5:0];	// Counters.scala:49:9, :67:11
      _T_3 <= _T_91 & _T_11 ? _wdata_63to6 : _T_79[6] ? _T_3 + 58'h1 : _T_3;	// Counters.scala:53:{20,38,43}, :68:23
      _T_0 <= _T_91 & _T_12 ? _wdata_5to0 : _T_78[5:0];	// Counters.scala:49:9, :67:11
      _T_1 <= _T_91 & _T_12 ? _wdata_63to6 : _T_78[6] ? _T_1 + 58'h1 : _T_1;	// Counters.scala:53:{20,38,43}, :68:23
      reg_dcsr_halt <= _T_91 & _T_24 ? _wdata[3] : reg_dcsr_halt;	// CSR.scala:333:27, :594:43, :595:23
      reg_dcsr_step <= _T_91 & _T_24 ? _wdata[2] : reg_dcsr_step;	// CSR.scala:333:27, :594:43, :596:23
      reg_dcsr_ebreakm <= _T_91 & _T_24 ? _wdata[15] : reg_dcsr_ebreakm;	// CSR.scala:333:27, :594:43, :597:26
      reg_dcsr_ebreaks <= _T_91 & _T_24 ? _wdata[13] : reg_dcsr_ebreaks;	// CSR.scala:333:27, :594:43, :598:39
      reg_dcsr_ebreaku <= _T_91 & _T_24 ? _wdata[12] : reg_dcsr_ebreaku;	// CSR.scala:333:27, :594:43, :599:41
      reg_dcsr_prv <= _T_91 & _T_24 ? _wdata[1:0] : _exception & _T_55 ? reg_mstatus_prv : reg_dcsr_prv;	// CSR.scala:333:27, :485:20, :594:43, :600:37, Cat.scala:30:58
      reg_mstatus_sie <= _T_91 ? (_T_30 ? _wdata[1] : _T_14 ? _wdata[1] : _T_85) : _T_85;	// CSR.scala:538:47, :550:27, :607:49, :608:25
      reg_mstatus_spie <= _T_91 ? (_T_30 ? _wdata[5] : _T_14 ? _wdata[5] : _T_83) : _T_83;	// CSR.scala:538:47, :549:28, :607:49, :609:26
      reg_mstatus_spp <= _T_91 ? (_T_30 ? _wdata[8] : _T_14 ? _wdata[8] : _T_84) : _T_84;	// CSR.scala:538:47, :548:27, :607:49, :610:25
      reg_mstatus_pum <= _T_91 ? (_T_30 ? _wdata[18] : _T_14 ? _wdata[18] : reg_mstatus_pum) : reg_mstatus_pum;	// CSR.scala:452:13, :538:47, :547:27, :607:49, :611:25
      reg_mstatus_fs <= _T_91 ? (_T_30 ? {2{|(_wdata[14:13])}} : _T_14 ? {2{|(_wdata[14:13])}} : reg_mstatus_fs) :
                                                reg_mstatus_fs;	// Bitwise.scala:71:12, CSR.scala:452:13, :538:47, :557:{47,73}, :607:49, :612:{24,50}
      reg_dcsr_debugint <= io_interrupts_debug;	// CSR.scala:660:21
      reg_bp_0_control_action <= _T_91 & _T_98 & _T_9 ? _T_99 & _wdata[12] : reg_bp_0_control_action;	// CSR.scala:312:48, :648:48, :652:{29,38}, :686:18
      reg_bp_0_control_dmode <= _T_91 & _T_98 & _T_9 ? _T_99 : reg_bp_0_control_dmode;	// CSR.scala:312:48, :651:28, :687:17
      reg_bp_0_control_r <= _T_91 & _T_98 & _T_9 ? _wdata[0] : reg_bp_0_control_r;	// CSR.scala:312:48, :648:48, :650:22, :688:13
      reg_bp_0_control_w <= _T_91 & _T_98 & _T_9 ? _wdata[1] : reg_bp_0_control_w;	// CSR.scala:312:48, :648:48, :650:22, :689:13
      reg_bp_0_control_x <= _T_91 & _T_98 & _T_9 ? _wdata[2] : reg_bp_0_control_x;	// CSR.scala:312:48, :648:48, :650:22, :690:13
    end
  end // always @(posedge)
  assign io_rw_rdata = _T_74;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10
  assign io_decode_fp_illegal = _T_43;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10
  assign io_decode_read_illegal = _T_44 | ~(io_decode_csr == 12'h338 | _T_48 | io_decode_csr == 12'hB09 | io_decode_csr ==
                12'h140 | io_decode_csr == 12'hB1C | io_decode_csr == 12'h337 | io_decode_csr == 12'h324 |
                io_decode_csr == 12'h303 | io_decode_csr == 12'h306 | io_decode_csr == 12'hC11 |
                io_decode_csr == 12'hB0B | io_decode_csr == 12'hC12 | io_decode_csr == 12'hB02 |
                io_decode_csr == 12'h326 | io_decode_csr == 12'hB0D | io_decode_csr == 12'hC0A |
                io_decode_csr == 12'h33E | io_decode_csr == 12'h331 | io_decode_csr == 12'hB13 |
                io_decode_csr == 12'h7A1 | io_decode_csr == 12'h343 | io_decode_csr == 12'h32E |
                io_decode_csr == 12'hB05 | io_decode_csr == 12'hC18 | io_decode_csr == 12'h143 |
                io_decode_csr == 12'h32A | io_decode_csr == 12'h100 | io_decode_csr == 12'hC0B |
                io_decode_csr == 12'hB16 | io_decode_csr == 12'hF14 | io_decode_csr == 12'hC1B |
                io_decode_csr == 12'hC1A | io_decode_csr == 12'hF13 | io_decode_csr == 12'hC04 |
                io_decode_csr == 12'hB1F | _T_45 | io_decode_csr == 12'hB0F | io_decode_csr == 12'hB06 |
                io_decode_csr == 12'hC14 | io_decode_csr == 12'h33A | io_decode_csr == 12'hF12 |
                io_decode_csr == 12'h330 | io_decode_csr == 12'hC1E | io_decode_csr == 12'hC08 |
                io_decode_csr == 12'h340 | io_decode_csr == 12'hB15 | io_decode_csr == 12'hB19 |
                io_decode_csr == 12'hB03 | io_decode_csr == 12'hC10 | io_decode_csr == 12'hC02 |
                io_decode_csr == 12'h104 | io_decode_csr == 12'h328 | io_decode_csr == 12'h336 |
                io_decode_csr == 12'h300 | io_decode_csr == 12'hB1B | _T_50 | io_decode_csr == 12'hC0E |
                io_decode_csr == 12'h334 | io_decode_csr == 12'h344 | io_decode_csr == 12'hB12 |
                io_decode_csr == 12'hB00 | io_decode_csr == 12'hC1C | io_decode_csr == 12'h7A0 |
                io_decode_csr == 12'hC03 | io_decode_csr == 12'hC0D | io_decode_csr == 12'h106 |
                io_decode_csr == 12'h332 | io_decode_csr == 12'hC0C | io_decode_csr == 12'h329 |
                io_decode_csr == 12'hB17 | io_decode_csr == 12'h304 | io_decode_csr == 12'h33C |
                io_decode_csr == 12'h32D | io_decode_csr == 12'hC17 | _T_47 | io_decode_csr == 12'h33D |
                io_decode_csr == 12'h327 | io_decode_csr == 12'h342 | io_decode_csr == 12'hC16 |
                io_decode_csr == 12'h141 | io_decode_csr == 12'h325 | io_decode_csr == 12'hB18 |
                io_decode_csr == 12'hB0A | io_decode_csr == 12'hC1D | io_decode_csr == 12'h142 |
                io_decode_csr == 12'h333 | io_decode_csr == 12'h7A2 | io_decode_csr == 12'h341 |
                io_decode_csr == 12'hC05 | io_decode_csr == 12'h32B | io_decode_csr == 12'h32C | _T_51 |
                io_decode_csr == 12'hB11 | io_decode_csr == 12'h302 | io_decode_csr == 12'h335 |
                io_decode_csr == 12'hC06 | io_decode_csr == 12'hC19 | io_decode_csr == 12'hF11 |
                io_decode_csr == 12'hC0F | io_decode_csr == 12'h33F | io_decode_csr == 12'h105 |
                io_decode_csr == 12'hB14 | _T_46 | io_decode_csr == 12'hB04 | io_decode_csr == 12'hB1A |
                io_decode_csr == 12'h32F | io_decode_csr == 12'hC09 | io_decode_csr == 12'hC1F |
                io_decode_csr == 12'hC13 | io_decode_csr == 12'h305 | io_decode_csr == 12'hB0C |
                io_decode_csr == 12'h33B | _T_49 | io_decode_csr == 12'hB0E | io_decode_csr == 12'h144 |
                io_decode_csr == 12'hC07 | io_decode_csr == 12'hC15 | io_decode_csr == 12'hB1D |
                io_decode_csr == 12'h301 | io_decode_csr == 12'hB07 | io_decode_csr == 12'h339 |
                io_decode_csr == 12'h323 | io_decode_csr == 12'hC00 | io_decode_csr == 12'hB10 |
                io_decode_csr == 12'hB1E | io_decode_csr == 12'hB08) | _T_51 & ~_allow_sfence_vma |
                (io_decode_csr > 12'hBFF & io_decode_csr < 12'hC20 | io_decode_csr > 12'hC7F &
                io_decode_csr < 12'hCA0) & _effective_prv < 3'h2 & _T_52[0] | ~reg_debug & (_T_45 | _T_46 |
                _T_47) | (_T_48 | _T_49 | _T_50) & _T_43;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:286:41, :405:73, :424:{5,42,57}, :425:{34,37}, :426:{67,151,160,171}, :427:{36,88,93}, :428:{69,74}, Cat.scala:30:58, Package.scala:7:47, :47:{47,55,60}
  assign io_decode_write_illegal = &(io_decode_csr[11:10]);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:429:{43,51}
  assign io_decode_write_flush = ~(io_decode_csr > 12'h33F & io_decode_csr < 12'h344 | io_decode_csr > 12'h13F &
                io_decode_csr < 12'h144);	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:405:73, :424:42, :430:{28,44,61,78,95,112,129,146}
  assign io_decode_system_illegal = _T_44 | ~_io_decode_csr_5 & io_decode_csr[2] & ~(|_T_42 | ~reg_mstatus_tw) |
                ~_io_decode_csr_5 & io_decode_csr[1] & ~(|_T_42 | ~reg_mstatus_tsr) | _io_decode_csr_5 &
                ~_allow_sfence_vma;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:418:{51,59,62}, :420:{60,63}, :425:37, :432:{5,39,43,46}, :433:{39,43,46,58}, :434:22
  assign io_csr_stall = reg_wfi;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:464:29
  assign io_eret = _T_59 | _insn_ret;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:450:38
  assign io_singleStep = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:451:34
  assign io_status_debug = reg_debug;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Cat.scala:30:58
  assign io_status_isa = reg_misa[31:0];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:421:58, :455:17
  assign io_status_prv = reg_mstatus_prv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Cat.scala:30:58
  assign io_status_sd = &reg_mstatus_fs;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13, :453:32
  assign io_status_tsr = reg_mstatus_tsr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:420:63
  assign io_status_tw = reg_mstatus_tw;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:418:62
  assign io_status_tvm = reg_mstatus_tvm;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:419:69
  assign io_status_mxr = reg_mstatus_mxr;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_pum = reg_mstatus_pum;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_mprv = reg_mstatus_mprv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_fs = reg_mstatus_fs;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_mpp = reg_mstatus_mpp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_spp = reg_mstatus_spp;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_mpie = reg_mstatus_mpie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_spie = reg_mstatus_spie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:452:13
  assign io_status_mie = reg_mstatus_mie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:281:81
  assign io_status_sie = reg_mstatus_sie;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:282:103
  assign io_ptbr_mode = reg_sptbr_mode;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:377:45
  assign io_ptbr_ppn = reg_sptbr_ppn;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:377:45
  assign io_evec = _insn_ret ? (_T_64 ? reg_mepc : _T_63 ? reg_dpc : _io_rw_addr_9 ? _tvec : reg_sepc) : _tvec;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:483:15, :511:15, :515:15, :522:15, Package.scala:40:38
  assign io_fatc = _system_insn & _insn_rs2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:416:37
  assign io_time = _T_4;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10
  assign io_fcsr_rm = reg_frm;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Cat.scala:30:58
  assign io_interrupt = _T_7 | |_all_interrupts & ~reg_debug & ~_T | reg_singleStepped;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:286:{34,41,52,55,70}, :292:18, :451:34, Cat.scala:30:58
  assign io_interrupt_cause = _T_7 ? 64'h800000000000000D : {58'h0, _all_interrupts[0] ? 6'h0 : _all_interrupts[1] ? 6'h1
                : _all_interrupts[2] ? 6'h2 : _all_interrupts[3] ? 6'h3 : _all_interrupts[4] ? 6'h4 :
                _all_interrupts[5] ? 6'h5 : _all_interrupts[6] ? 6'h6 : _all_interrupts[7] ? 6'h7 :
                _all_interrupts[8] ? 6'h8 : _all_interrupts[9] ? 6'h9 : _all_interrupts[10] ? 6'hA :
                _all_interrupts[11] ? 6'hB : _all_interrupts[12] ? 6'hC : _all_interrupts[13] ? 6'hD :
                _all_interrupts[14] ? 6'hE : _all_interrupts[15] ? 6'hF : _all_interrupts[16] ? 6'h10 :
                _all_interrupts[17] ? 6'h11 : _all_interrupts[18] ? 6'h12 : _all_interrupts[19] ? 6'h13 :
                _all_interrupts[20] ? 6'h14 : _all_interrupts[21] ? 6'h15 : _all_interrupts[22] ? 6'h16 :
                _all_interrupts[23] ? 6'h17 : _all_interrupts[24] ? 6'h18 : _all_interrupts[25] ? 6'h19 :
                _all_interrupts[26] ? 6'h1A : _all_interrupts[27] ? 6'h1B : _all_interrupts[28] ? 6'h1C :
                _all_interrupts[29] ? 6'h1D : _all_interrupts[30] ? 6'h1E : _all_interrupts[31] ? 6'h1F :
                _all_interrupts[32] ? 6'h20 : _all_interrupts[33] ? 6'h21 : _all_interrupts[34] ? 6'h22 :
                _all_interrupts[35] ? 6'h23 : _all_interrupts[36] ? 6'h24 : _all_interrupts[37] ? 6'h25 :
                _all_interrupts[38] ? 6'h26 : _all_interrupts[39] ? 6'h27 : _all_interrupts[40] ? 6'h28 :
                _all_interrupts[41] ? 6'h29 : _all_interrupts[42] ? 6'h2A : _all_interrupts[43] ? 6'h2B :
                _all_interrupts[44] ? 6'h2C : _all_interrupts[45] ? 6'h2D : _all_interrupts[46] ? 6'h2E :
                _all_interrupts[47] ? 6'h2F : _all_interrupts[48] ? 6'h30 : _all_interrupts[49] ? 6'h31 :
                _all_interrupts[50] ? 6'h32 : _all_interrupts[51] ? 6'h33 : _all_interrupts[52] ? 6'h34 :
                _all_interrupts[53] ? 6'h35 : _all_interrupts[54] ? 6'h36 : _all_interrupts[55] ? 6'h37 :
                _all_interrupts[56] ? 6'h38 : _all_interrupts[57] ? 6'h39 : _all_interrupts[58] ? 6'h3A :
                _all_interrupts[59] ? 6'h3B : _all_interrupts[60] ? 6'h3C : _all_interrupts[61] ? 6'h3D :
                {5'h1F, ~(_all_interrupts[62])}} - 64'h8000000000000000;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:285:43, :293:{24,46}, Counters.scala:47:37, :52:27, Mux.scala:31:69, OneHot.scala:39:40
  assign io_bp_0_control_dmode = reg_bp_0_control_dmode;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_action = reg_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_tmatch = reg_bp_0_control_tmatch;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_m = reg_bp_0_control_m;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_s = reg_bp_0_control_s;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_u = reg_bp_0_control_u;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_x = reg_bp_0_control_x;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_w = reg_bp_0_control_w;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_control_r = reg_bp_0_control_r;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, CSR.scala:312:48
  assign io_bp_0_address = reg_bp_0_address;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:1977:10, Package.scala:40:38
endmodule

module BreakpointUnit(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10
  input         io_status_debug,
  input  [1:0]  io_status_prv,
  input         io_bp_0_control_action,
  input  [1:0]  io_bp_0_control_tmatch,
  input         io_bp_0_control_m, io_bp_0_control_s, io_bp_0_control_u,
  input         io_bp_0_control_x, io_bp_0_control_w, io_bp_0_control_r,
  input  [38:0] io_bp_0_address, io_pc, io_ea,
  output        io_xcpt_if, io_xcpt_ld, io_xcpt_st, io_debug_if, io_debug_ld,
  output        io_debug_st);

  wire [3:0] _T = {io_bp_0_control_m, 1'h0, io_bp_0_control_s, io_bp_0_control_u} >> io_status_prv;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:30:68, Cat.scala:30:58
  wire _T_0 = ~io_status_debug & _T[0];	// Breakpoint.scala:30:{35,50,68}
  wire _io_bp_0_control_tmatch_1 = io_bp_0_control_tmatch[1];	// Breakpoint.scala:47:23
  wire _io_bp_0_control_tmatch_0 = io_bp_0_control_tmatch[0];	// Breakpoint.scala:44:36
  wire _T_1 = _io_bp_0_control_tmatch_0 & io_bp_0_address[0];	// Breakpoint.scala:38:{73,83}
  wire _T_2 = _T_1 & io_bp_0_address[1];	// Breakpoint.scala:38:{73,83}
  wire [38:0] _T_3 = {35'h0, _T_2 & io_bp_0_address[2], _T_2, _T_1, _io_bp_0_control_tmatch_0};	// Breakpoint.scala:38:{73,83}, :41:9
  wire [38:0] _T_4 = ~io_bp_0_address | _T_3;	// Breakpoint.scala:41:{24,33}
  wire _T_5 = _io_bp_0_control_tmatch_1 ? io_ea >= io_bp_0_address ^ _io_bp_0_control_tmatch_0 : (~io_ea
                | _T_3) == _T_4;	// Breakpoint.scala:41:{6,9,19}, :44:{8,20}, :47:8
  wire _T_6 = _T_0 & io_bp_0_control_r & _T_5;	// Breakpoint.scala:73:38
  wire _T_7 = _T_0 & io_bp_0_control_w & _T_5;	// Breakpoint.scala:74:38
  wire _T_8 = _T_0 & io_bp_0_control_x & (_io_bp_0_control_tmatch_1 ? io_pc >= io_bp_0_address ^
                _io_bp_0_control_tmatch_0 : (~io_pc | _T_3) == _T_4);	// Breakpoint.scala:41:{6,9,19}, :44:{8,20}, :47:8, :75:38
  assign io_xcpt_if = _T_8 & ~io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:78:37, :80:34
  assign io_xcpt_ld = _T_6 & ~io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:78:{34,37}
  assign io_xcpt_st = _T_7 & ~io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:78:37, :79:34
  assign io_debug_if = _T_8 & io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:80:69
  assign io_debug_ld = _T_6 & io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:78:69
  assign io_debug_st = _T_7 & io_bp_0_control_action;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4068:10, Breakpoint.scala:79:69
endmodule

module ALU(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4210:10
  input         io_dw,
  input  [3:0]  io_fn,
  input  [63:0] io_in2, io_in1,
  output [63:0] io_out, io_adder_out,
  output        io_cmp_out);

  wire _io_fn_3 = io_fn[3];	// ALU.scala:41:29
  wire [63:0] _T = {64{_io_fn_3}} ^ io_in2;	// ALU.scala:61:20
  wire [63:0] _in1_xor_in2 = io_in1 ^ _T;	// ALU.scala:62:28
  wire [63:0] _T_0 = io_in1 + _T + {63'h0, _io_fn_3};	// ALU.scala:63:36, :90:50
  wire _io_in1_63 = io_in1[63];	// ALU.scala:68:15
  wire _io_in2_63 = io_in2[63];	// ALU.scala:68:34
  wire _T_1 = io_fn[0] ^ (_io_fn_3 ? (_io_in1_63 == _io_in2_63 ? _T_0[63] : io_fn[1] ? _io_in2_63 :
                _io_in1_63) : _in1_xor_in2 == 64'h0);	// ALU.scala:43:35, :44:35, :66:36, :67:{8,35}, :68:{8,24,56}, :69:8, :84:18
  wire [31:0] _T_2 = io_dw ? io_in1[63:32] : {32{_io_fn_3 & io_in1[31]}};	// ALU.scala:76:{46,55}, :77:{24,48}, Bitwise.scala:71:12
  wire _T_3 = io_fn == 4'h5 | io_fn == 4'hB;	// ALU.scala:81:{24,35,44}
  wire [63:0] _shin = _T_3 ? {_T_2, io_in1[31:0]} : {io_in1[0], io_in1[1], io_in1[2], io_in1[3], io_in1[4],
                io_in1[5], io_in1[6], io_in1[7], io_in1[8], io_in1[9], io_in1[10], io_in1[11], io_in1[12],
                io_in1[13], io_in1[14], io_in1[15], io_in1[16], io_in1[17], io_in1[18], io_in1[19],
                io_in1[20], io_in1[21], io_in1[22], io_in1[23], io_in1[24], io_in1[25], io_in1[26],
                io_in1[27], io_in1[28], io_in1[29], io_in1[30], io_in1[31], _T_2[0], _T_2[1], _T_2[2],
                _T_2[3], _T_2[4], _T_2[5], _T_2[6], _T_2[7], _T_2[8], _T_2[9], _T_2[10], _T_2[11],
                _T_2[12], _T_2[13], _T_2[14], _T_2[15], _T_2[16], _T_2[17], _T_2[18], _T_2[19], _T_2[20],
                _T_2[21], _T_2[22], _T_2[23], _T_2[24], _T_2[25], _T_2[26], _T_2[27], _T_2[28], _T_2[29],
                _T_2[30], _T_2[31]};	// ALU.scala:79:34, :81:17, Bitwise.scala:102:{21,39,46}, Cat.scala:30:58
  wire [64:0] _T_4 = $signed($signed({_io_fn_3 & _shin[63], _shin}) >>> {59'h0, io_in2[5] & io_dw, io_in2[4:0]});	// ALU.scala:78:{29,33,60}, :82:{35,41,64}, Cat.scala:30:58
  wire _T_5 = io_fn == 4'h6;	// ALU.scala:88:45
  wire [63:0] _out = io_fn == 4'h0 | io_fn == 4'hA ? _T_0 : {63'h0, (io_fn == 4'h2 | io_fn == 4'h3 | io_fn >
                4'hB) & _T_1} | (io_fn == 4'h4 | _T_5 ? _in1_xor_in2 : 64'h0) | (_T_5 | io_fn == 4'h7 ?
                io_in1 & io_in2 : 64'h0) | (_T_3 ? _T_4[63:0] : 64'h0) | (io_fn == 4'h1 ? {_T_4[0],
                _T_4[1], _T_4[2], _T_4[3], _T_4[4], _T_4[5], _T_4[6], _T_4[7], _T_4[8], _T_4[9], _T_4[10],
                _T_4[11], _T_4[12], _T_4[13], _T_4[14], _T_4[15], _T_4[16], _T_4[17], _T_4[18], _T_4[19],
                _T_4[20], _T_4[21], _T_4[22], _T_4[23], _T_4[24], _T_4[25], _T_4[26], _T_4[27], _T_4[28],
                _T_4[29], _T_4[30], _T_4[31], _T_4[32], _T_4[33], _T_4[34], _T_4[35], _T_4[36], _T_4[37],
                _T_4[38], _T_4[39], _T_4[40], _T_4[41], _T_4[42], _T_4[43], _T_4[44], _T_4[45], _T_4[46],
                _T_4[47], _T_4[48], _T_4[49], _T_4[50], _T_4[51], _T_4[52], _T_4[53], _T_4[54], _T_4[55],
                _T_4[56], _T_4[57], _T_4[58], _T_4[59], _T_4[60], _T_4[61], _T_4[62], _T_4[63]} : 64'h0);	// ALU.scala:42:{30,48,59,66}, :81:44, :82:73, :84:18, :85:{18,25}, :88:{18,25,36}, :89:{18,35,44,63}, :90:{35,50,58}, :91:{16,23,34,43}, Bitwise.scala:102:{21,31,39,46}
  assign io_out = io_dw ? _out : {{32{_out[31]}}, _out[31:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4210:10, ALU.scala:96:{37,56,66}, Bitwise.scala:71:12, Cat.scala:30:58
  assign io_adder_out = _T_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4210:10
  assign io_cmp_out = _T_1;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4210:10
endmodule

module MulDiv(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4427:10
  input         clock, reset, io_req_valid,
  input  [3:0]  io_req_bits_fn,
  input         io_req_bits_dw,
  input  [63:0] io_req_bits_in1, io_req_bits_in2,
  input  [4:0]  io_req_bits_tag,
  input         io_kill, io_resp_ready,
  output        io_req_ready, io_resp_valid,
  output [63:0] io_resp_bits_data,
  output [4:0]  io_resp_bits_tag);

  wire         _T;	// Multiplier.scala:167:25
  wire         _T_0;	// Multiplier.scala:166:26
  reg  [2:0]   state;	// Multiplier.scala:45:18
  reg  [3:0]   req_fn;	// Multiplier.scala:47:16
  reg          req_dw;	// Multiplier.scala:47:16
  reg  [63:0]  req_in1;	// Multiplier.scala:47:16
  reg  [63:0]  req_in2;	// Multiplier.scala:47:16
  reg  [4:0]   req_tag;	// Multiplier.scala:47:16
  reg  [6:0]   count;	// Multiplier.scala:48:18
  reg          neg_out;	// Multiplier.scala:49:20
  reg          isMul;	// Multiplier.scala:50:18
  reg          isHi;	// Multiplier.scala:51:17
  reg  [64:0]  divisor;	// Multiplier.scala:52:20
  reg  [129:0] remainder;	// Multiplier.scala:53:22

  `ifndef SYNTHESIS	// Multiplier.scala:45:18
    `ifdef RANDOMIZE_REG_INIT	// Multiplier.scala:45:18
      reg [31:0] _RANDOM;	// Multiplier.scala:45:18
      reg [31:0] _RANDOM_1;	// Multiplier.scala:47:16
      reg [31:0] _RANDOM_2;	// Multiplier.scala:47:16
      reg [31:0] _RANDOM_3;	// Multiplier.scala:47:16
      reg [31:0] _RANDOM_4;	// Multiplier.scala:47:16
      reg [31:0] _RANDOM_5;	// Multiplier.scala:52:20
      reg [31:0] _RANDOM_6;	// Multiplier.scala:52:20
      reg [31:0] _RANDOM_7;	// Multiplier.scala:53:22
      reg [31:0] _RANDOM_8;	// Multiplier.scala:53:22
      reg [31:0] _RANDOM_9;	// Multiplier.scala:53:22
      reg [31:0] _RANDOM_10;	// Multiplier.scala:53:22

    `endif
    initial begin	// Multiplier.scala:45:18
      `INIT_RANDOM_PROLOG_	// Multiplier.scala:45:18
      `ifdef RANDOMIZE_REG_INIT	// Multiplier.scala:45:18
        _RANDOM = `RANDOM;	// Multiplier.scala:45:18
        state = _RANDOM[2:0];	// Multiplier.scala:45:18
        req_fn = _RANDOM[6:3];	// Multiplier.scala:47:16
        req_dw = _RANDOM[7];	// Multiplier.scala:47:16
        _RANDOM_1 = `RANDOM;	// Multiplier.scala:47:16
        _RANDOM_2 = `RANDOM;	// Multiplier.scala:47:16
        req_in1 = {_RANDOM_2[7:0], _RANDOM_1, _RANDOM[31:8]};	// Multiplier.scala:47:16
        _RANDOM_3 = `RANDOM;	// Multiplier.scala:47:16
        _RANDOM_4 = `RANDOM;	// Multiplier.scala:47:16
        req_in2 = {_RANDOM_4[7:0], _RANDOM_3, _RANDOM_2[31:8]};	// Multiplier.scala:47:16
        req_tag = _RANDOM_4[12:8];	// Multiplier.scala:47:16
        count = _RANDOM_4[19:13];	// Multiplier.scala:48:18
        neg_out = _RANDOM_4[20];	// Multiplier.scala:49:20
        isMul = _RANDOM_4[21];	// Multiplier.scala:50:18
        isHi = _RANDOM_4[22];	// Multiplier.scala:51:17
        _RANDOM_5 = `RANDOM;	// Multiplier.scala:52:20
        _RANDOM_6 = `RANDOM;	// Multiplier.scala:52:20
        divisor = {_RANDOM_6[23:0], _RANDOM_5, _RANDOM_4[31:23]};	// Multiplier.scala:52:20
        _RANDOM_7 = `RANDOM;	// Multiplier.scala:53:22
        _RANDOM_8 = `RANDOM;	// Multiplier.scala:53:22
        _RANDOM_9 = `RANDOM;	// Multiplier.scala:53:22
        _RANDOM_10 = `RANDOM;	// Multiplier.scala:53:22
        remainder = {_RANDOM_10[25:0], _RANDOM_9, _RANDOM_8, _RANDOM_7, _RANDOM_6[31:24]};	// Multiplier.scala:53:22
      `endif
    end // initial
  `endif
      wire [63:0] _remainder_63to0 = remainder[63:0];	// Multiplier.scala:77:29, :78:37
      wire _remainder_31 = remainder[31];	// CircuitMath.scala:32:12, Multiplier.scala:77:29
      wire [2:0] _T_1 = {2'h2, ~neg_out};	// CircuitMath.scala:32:10, Multiplier.scala:96:17
      wire _io_req_bits_fn_2 = io_req_bits_fn[2];	// Decode.scala:13:65
      wire _io_req_bits_fn_3 = io_req_bits_fn[3];	// Decode.scala:13:65
      wire _cmdMul = ~_io_req_bits_fn_2 | _io_req_bits_fn_3;	// Decode.scala:13:121, :14:30
      wire _T_2 = {io_req_bits_fn[2], io_req_bits_fn[0]} == 2'h1 | io_req_bits_fn[1] | _io_req_bits_fn_3;	// Decode.scala:13:{65,121}, :14:30
      wire _T_3 = {io_req_bits_fn[3], io_req_bits_fn[0]} == 2'h0;	// Decode.scala:13:{65,121}, Multiplier.scala:144:19
      wire _lhs_sign = (_T_3 | ~_io_req_bits_fn_2 | io_req_bits_fn[1:0] == 2'h0) & (io_req_bits_dw ?
                io_req_bits_in1[63] : io_req_bits_in1[31]);	// Decode.scala:13:{65,121}, :14:30, Multiplier.scala:70:{23,29,38,48}, :144:19
      wire _rhs_sign = (_T_3 | ~_io_req_bits_fn_2) & (io_req_bits_dw ? io_req_bits_in2[63] : io_req_bits_in2[31]);	// Decode.scala:13:121, :14:30, Multiplier.scala:70:{23,29,38,48}
      wire [64:0] _T_4 = remainder[128:64] - divisor;	// Multiplier.scala:77:{29,37}
      wire _T_5 = state == 3'h1;	// Multiplier.scala:80:15
      wire _remainder_63 = remainder[63];	// Multiplier.scala:77:29, :81:20
      wire _divisor_63 = divisor[63];	// Multiplier.scala:77:37, :84:18
      wire _T_6 = state == 3'h4;	// Multiplier.scala:80:15, :90:15
      wire _T_7 = state == 3'h3;	// Multiplier.scala:80:15, :94:15
      wire _T_8 = state == 3'h2;	// Multiplier.scala:80:15, :87:11, :98:15
      wire _T_9 = _T_8 & isMul;	// Multiplier.scala:81:26, :98:26
      wire [64:0] _remainder_129to65 = remainder[129:65];	// Multiplier.scala:77:29, :99:31
      wire [72:0] _T_10 = {{8{divisor[64]}}, divisor} * {{65{remainder[7]}}, remainder[7:0]} + {{8{remainder[129]}},
                _remainder_129to65};	// Multiplier.scala:77:{29,37}, :103:{22,43,52}
      wire [64:0] _T_11 = $signed(65'sh10000000000000000 >>> {59'h0, count[2:0], 3'h0});	// Multiplier.scala:45:18, :106:{46,56,72}
      wire _T_12 = count != 7'h7 & |count & ~isHi & (_remainder_63to0 & ~(_T_11[63:0])) == 64'h0;	// Multiplier.scala:106:{56,91}, :107:{47,81}, :108:{7,13,24,26,37}
      wire [6:0] _T_13 = count + 7'h1;	// Multiplier.scala:106:56, :113:20
      wire _T_14 = _T_8 & ~isMul;	// Multiplier.scala:81:26, :118:{26,29}
      wire _T_15 = _T_4[64];	// Multiplier.scala:122:28
      wire _T_16 = ~(|count) & ~_T_15;	// Multiplier.scala:106:56, :107:81, :123:67, :134:{24,30}
      wire [31:0] _divisor_63to32 = divisor[63:32];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [15:0] _divisor_63to48 = divisor[63:48];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [7:0] _divisor_63to56 = divisor[63:56];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_63to60 = divisor[63:60];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_55to52 = divisor[55:52];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [7:0] _divisor_47to40 = divisor[47:40];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_47to44 = divisor[47:44];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_39to36 = divisor[39:36];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [15:0] _divisor_31to16 = divisor[31:16];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [7:0] _divisor_31to24 = divisor[31:24];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_31to28 = divisor[31:28];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_23to20 = divisor[23:20];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [7:0] _divisor_15to8 = divisor[15:8];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_15to12 = divisor[15:12];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [3:0] _divisor_7to4 = divisor[7:4];	// CircuitMath.scala:35:17, Multiplier.scala:77:37
      wire [4:0] _T_17 = |_divisor_63to32 ? {|_divisor_63to48, |_divisor_63to48 ? {|_divisor_63to56,
                |_divisor_63to56 ? {|_divisor_63to60, |_divisor_63to60 ? (_divisor_63 ? 2'h3 : divisor[62]
                ? 2'h2 : {1'h0, divisor[61]}) : divisor[59] ? 2'h3 : divisor[58] ? 2'h2 : {1'h0,
                divisor[57]}} : {|_divisor_55to52, |_divisor_55to52 ? (divisor[55] ? 2'h3 : divisor[54] ?
                2'h2 : {1'h0, divisor[53]}) : divisor[51] ? 2'h3 : divisor[50] ? 2'h2 : {1'h0,
                divisor[49]}}} : {|_divisor_47to40, |_divisor_47to40 ? {|_divisor_47to44, |_divisor_47to44
                ? (divisor[47] ? 2'h3 : divisor[46] ? 2'h2 : {1'h0, divisor[45]}) : divisor[43] ? 2'h3 :
                divisor[42] ? 2'h2 : {1'h0, divisor[41]}} : {|_divisor_39to36, |_divisor_39to36 ?
                (divisor[39] ? 2'h3 : divisor[38] ? 2'h2 : {1'h0, divisor[37]}) : divisor[35] ? 2'h3 :
                divisor[34] ? 2'h2 : {1'h0, divisor[33]}}}} : {|_divisor_31to16, |_divisor_31to16 ?
                {|_divisor_31to24, |_divisor_31to24 ? {|_divisor_31to28, |_divisor_31to28 ? (divisor[31] ?
                2'h3 : divisor[30] ? 2'h2 : {1'h0, divisor[29]}) : divisor[27] ? 2'h3 : divisor[26] ? 2'h2
                : {1'h0, divisor[25]}} : {|_divisor_23to20, |_divisor_23to20 ? (divisor[23] ? 2'h3 :
                divisor[22] ? 2'h2 : {1'h0, divisor[21]}) : divisor[19] ? 2'h3 : divisor[18] ? 2'h2 :
                {1'h0, divisor[17]}}} : {|_divisor_15to8, |_divisor_15to8 ? {|_divisor_15to12,
                |_divisor_15to12 ? (divisor[15] ? 2'h3 : divisor[14] ? 2'h2 : {1'h0, divisor[13]}) :
                divisor[11] ? 2'h3 : divisor[10] ? 2'h2 : {1'h0, divisor[9]}} : {|_divisor_7to4,
                |_divisor_7to4 ? (divisor[7] ? 2'h3 : divisor[6] ? 2'h2 : {1'h0, divisor[5]}) : divisor[3]
                ? 2'h3 : divisor[2] ? 2'h2 : {1'h0, divisor[1]}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, Decode.scala:14:30, Multiplier.scala:77:37
      wire [31:0] _remainder_63to32 = remainder[63:32];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [15:0] _remainder_63to48 = remainder[63:48];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [7:0] _remainder_63to56 = remainder[63:56];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_63to60 = remainder[63:60];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_55to52 = remainder[55:52];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [7:0] _remainder_47to40 = remainder[47:40];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_47to44 = remainder[47:44];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_39to36 = remainder[39:36];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [15:0] _remainder_31to16 = remainder[31:16];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [7:0] _remainder_31to24 = remainder[31:24];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_31to28 = remainder[31:28];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_23to20 = remainder[23:20];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [7:0] _remainder_15to8 = remainder[15:8];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_15to12 = remainder[15:12];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [3:0] _remainder_7to4 = remainder[7:4];	// CircuitMath.scala:35:17, Multiplier.scala:77:29
      wire [4:0] _T_18 = |_remainder_63to32 ? {|_remainder_63to48, |_remainder_63to48 ? {|_remainder_63to56,
                |_remainder_63to56 ? {|_remainder_63to60, |_remainder_63to60 ? (_remainder_63 ? 2'h3 :
                remainder[62] ? 2'h2 : {1'h0, remainder[61]}) : remainder[59] ? 2'h3 : remainder[58] ? 2'h2
                : {1'h0, remainder[57]}} : {|_remainder_55to52, |_remainder_55to52 ? (remainder[55] ? 2'h3
                : remainder[54] ? 2'h2 : {1'h0, remainder[53]}) : remainder[51] ? 2'h3 : remainder[50] ?
                2'h2 : {1'h0, remainder[49]}}} : {|_remainder_47to40, |_remainder_47to40 ?
                {|_remainder_47to44, |_remainder_47to44 ? (remainder[47] ? 2'h3 : remainder[46] ? 2'h2 :
                {1'h0, remainder[45]}) : remainder[43] ? 2'h3 : remainder[42] ? 2'h2 : {1'h0,
                remainder[41]}} : {|_remainder_39to36, |_remainder_39to36 ? (remainder[39] ? 2'h3 :
                remainder[38] ? 2'h2 : {1'h0, remainder[37]}) : remainder[35] ? 2'h3 : remainder[34] ? 2'h2
                : {1'h0, remainder[33]}}}} : {|_remainder_31to16, |_remainder_31to16 ? {|_remainder_31to24,
                |_remainder_31to24 ? {|_remainder_31to28, |_remainder_31to28 ? (_remainder_31 ? 2'h3 :
                remainder[30] ? 2'h2 : {1'h0, remainder[29]}) : remainder[27] ? 2'h3 : remainder[26] ? 2'h2
                : {1'h0, remainder[25]}} : {|_remainder_23to20, |_remainder_23to20 ? (remainder[23] ? 2'h3
                : remainder[22] ? 2'h2 : {1'h0, remainder[21]}) : remainder[19] ? 2'h3 : remainder[18] ?
                2'h2 : {1'h0, remainder[17]}}} : {|_remainder_15to8, |_remainder_15to8 ?
                {|_remainder_15to12, |_remainder_15to12 ? (remainder[15] ? 2'h3 : remainder[14] ? 2'h2 :
                {1'h0, remainder[13]}) : remainder[11] ? 2'h3 : remainder[10] ? 2'h2 : {1'h0,
                remainder[9]}} : {|_remainder_7to4, |_remainder_7to4 ? (remainder[7] ? 2'h3 : remainder[6]
                ? 2'h2 : {1'h0, remainder[5]}) : remainder[3] ? 2'h3 : remainder[2] ? 2'h2 : {1'h0,
                remainder[1]}}}};	// Cat.scala:30:58, CircuitMath.scala:30:8, :32:{10,12}, :37:22, :38:21, Decode.scala:14:30, Multiplier.scala:77:29
      wire [5:0] _T_19 = {|_divisor_63to32, _T_17} - 6'h1 - {|_remainder_63to32, _T_18};	// CircuitMath.scala:37:22, Multiplier.scala:138:{31,44}
      wire _T_20 = {|_divisor_63to32, _T_17} > {|_remainder_63to32, _T_18};	// Cat.scala:30:58, CircuitMath.scala:37:22, Multiplier.scala:139:33
      wire _T_21 = ~(|count) & ~_T_16 & (|_T_19 | _T_20);	// Multiplier.scala:106:56, :107:81, :134:24, :140:{33,41,53,70}
      wire [5:0] _T_22 = _T_20 ? 6'h3F : _T_19;	// Multiplier.scala:138:31, :142:22
      wire _T_23 = _T & io_req_valid;	// Decoupled.scala:30:37, Multiplier.scala:167:25
      wire [128:0] _T_24 = {_remainder_129to65, _remainder_63to0} >> 6'h0 - {count[2:0], 3'h0};	// Cat.scala:30:58, Multiplier.scala:45:18, :106:56, :109:{27,36}
  always @(posedge clock) begin	// Multiplier.scala:155:11
    if (reset)	// Multiplier.scala:45:18
      state <= 3'h0;	// Multiplier.scala:45:18
    else	// Multiplier.scala:45:18
      state <= _T_23 ? (_lhs_sign | _rhs_sign & ~_cmdMul ? 3'h1 : 3'h2) : io_resp_ready & _T_0 | io_kill ?
                                                3'h0 : _T_14 & count == 7'h40 ? (isHi ? 3'h3 : _T_1) : _T_9 & (_T_12 | count == 7'h7) ?
                                                (isHi ? 3'h3 : 3'h5) : _T_7 ? _T_1 : _T_6 ? 3'h5 : _T_5 ? 3'h2 : state;	// Decoupled.scala:30:37, Multiplier.scala:45:18, :80:15, :87:11, :92:11, :94:15, :96:11, :106:56, :107:47, :108:7, :109:36, :114:{16,25}, :115:{13,19}, :127:17, :128:{13,19}, :150:24, :151:11, :154:{11,17,27,39,42}, :166:26
    if (_T_23) begin	// Multiplier.scala:156:10
      isMul <= _cmdMul;	// Multiplier.scala:155:11
      isHi <= _T_2;	// Multiplier.scala:156:10
    end
    count <= _T_23 ? 7'h0 : _T_14 ? (_T_21 ? {1'h0, _T_22} : _T_13) : _T_9 ? _T_13 : count;	// Decode.scala:14:30, Multiplier.scala:106:56, :113:11, :145:15, :157:11
    neg_out <= _T_23 ? ~_cmdMul & (_T_2 ? _lhs_sign : _lhs_sign != _rhs_sign) : (~_T_14 | ~(_T_16 &
                                ~isHi)) & neg_out;	// Multiplier.scala:96:17, :108:7, :148:{18,38}, :154:42, :158:{13,24,30,57}
    divisor <= _T_23 ? {_rhs_sign, io_req_bits_dw ? io_req_bits_in2[63:32] : {32{_rhs_sign}},
                                io_req_bits_in2[31:0]} : _T_5 & (_divisor_63 | isMul) ? _T_4 : divisor;	// Bitwise.scala:71:12, Cat.scala:30:58, Multiplier.scala:71:{17,43}, :72:15, :77:37, :81:26, :84:24, :85:15, :159:13
    remainder <= _T_23 ? {66'h0, io_req_bits_dw ? io_req_bits_in1[63:32] : {32{_lhs_sign}},
                                io_req_bits_in1[31:0]} : _T_14 ? {1'h0, _T_21 ? {2'h0, {63'h0, _remainder_63to0} << _T_22}
                                : {_T_15 ? remainder[127:64] : _T_4[63:0], _remainder_63to0, ~_T_15}} : _T_9 ?
                                {_T_10[72:8], 1'h0, _T_12 ? _T_24[63:0] : {_T_10[7:0], remainder[63:8]}} : _T_7 ? {66'h0,
                                remainder[128:65]} : _T_6 | _T_5 & (_remainder_63 | isMul) ? {66'h0, 64'h0 -
                                _remainder_63to0} : remainder;	// Bitwise.scala:71:12, Cat.scala:30:58, Decode.scala:14:30, Multiplier.scala:71:{17,43}, :72:15, :77:29, :78:27, :81:26, :82:17, :91:15, :95:{15,27}, :104:38, :108:37, :110:{37,55}, :111:15, :123:{14,24,45,67}, :144:{19,39}, :160:15
    if (_T_23) begin	// Multiplier.scala:161:9
      req_fn <= io_req_bits_fn;	// Multiplier.scala:161:9
      req_dw <= io_req_bits_dw;	// Multiplier.scala:161:9
      req_in1 <= io_req_bits_in1;	// Multiplier.scala:161:9
      req_in2 <= io_req_bits_in2;	// Multiplier.scala:161:9
      req_tag <= io_req_bits_tag;	// Multiplier.scala:161:9
    end
  end // always @(posedge)
  assign _T_0 = state == 3'h5;	// Multiplier.scala:80:15, :92:11, :166:26
  assign _T = state == 3'h0;	// Multiplier.scala:45:18, :80:15, :167:25
  assign io_req_ready = _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4427:10, Multiplier.scala:167:25
  assign io_resp_valid = _T_0;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4427:10, Multiplier.scala:166:26
  assign io_resp_bits_data = req_dw ? _remainder_63to0 : {{32{_remainder_31}}, remainder[31:0]};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4427:10, Bitwise.scala:71:12, Cat.scala:30:58, Multiplier.scala:77:29, :161:9, :165:{27,86}
  assign io_resp_bits_tag = req_tag;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4427:10, Multiplier.scala:161:9
endmodule

module RVCExpander(	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10
  input  [31:0] io_in,
  output [31:0] io_out_bits,
  output [4:0]  io_out_rd, io_out_rs1, io_out_rs2, io_out_rs3,
  output        io_rvc);

  wire _io_in_5 = io_in[5];	// RVC.scala:34:45
  wire _io_in_6 = io_in[6];	// RVC.scala:34:51
  wire [2:0] _io_in_4to2 = io_in[4:2];	// RVC.scala:31:30
  wire [4:0] _T = {2'h1, _io_in_4to2};	// Cat.scala:30:58
  wire [1:0] _io_in_6to5 = io_in[6:5];	// RVC.scala:36:20
  wire [2:0] _io_in_12to10 = io_in[12:10];	// RVC.scala:36:28
  wire [2:0] _io_in_9to7 = io_in[9:7];	// RVC.scala:30:30
  wire [4:0] _T_0 = {2'h1, _io_in_9to7};	// Cat.scala:30:58
  wire _io_in_12 = io_in[12];	// RVC.scala:63:32
  wire [1:0] _io_in_11to10 = io_in[11:10];	// RVC.scala:63:66
  wire _io_in_12_1 = io_in[12];	// RVC.scala:66:30
  wire [1:0] _io_in_11to10_2 = io_in[11:10];	// RVC.scala:66:64
  wire _io_in_12_3 = io_in[12];	// RVC.scala:43:30
  wire [6:0] _T_4 = {7{_io_in_12_3}};	// Bitwise.scala:71:12
  wire [4:0] _io_in_6to2 = io_in[6:2];	// RVC.scala:43:38
  wire [11:0] _T_5 = {_T_4, _io_in_6to2};	// Cat.scala:30:58
  wire [4:0] _io_in_11to7 = io_in[11:7];	// RVC.scala:33:13
  wire [1:0] _io_in_4to3 = io_in[4:3];	// RVC.scala:42:42
  wire _io_in_2 = io_in[2];	// RVC.scala:42:56
  wire _io_in_6_6 = io_in[6];	// Package.scala:18:26
  wire _io_in_5_7 = io_in[5];	// Package.scala:18:26
  wire _io_in_10 = io_in[10];	// RVC.scala:107:42
  wire _io_in_12_8 = io_in[12];	// RVC.scala:123:34
  wire [1:0] _io_in_11to10_9 = io_in[11:10];	// RVC.scala:123:67
  wire _io_in_0 = io_in[0];	// Package.scala:18:26
  wire _io_in_1 = io_in[1];	// Package.scala:19:17
  wire _io_in_15 = io_in[15];	// Package.scala:18:26
  wire _io_in_14 = io_in[14];	// Package.scala:18:26
  wire _io_in_13 = io_in[13];	// Package.scala:18:26
  assign io_out_bits = _io_in_1 ? (_io_in_0 ? io_in : _io_in_15 ? (_io_in_14 ? (_io_in_13 ? {3'h0, _io_in_9to7,
                _io_in_12_8, _io_in_6to2, 8'h13, _io_in_11to10_9, 10'h23} : {4'h0, io_in[8:7], io_in[12],
                _io_in_6to2, 8'h12, io_in[11:9], 9'h23}) : _io_in_13 ? {3'h0, _io_in_9to7, _io_in_12_8,
                _io_in_6to2, 8'h13, _io_in_11to10_9, 10'h27} : {7'h0, _io_in_12_3 ? (|_io_in_6to2 ?
                {_io_in_6to2, _io_in_11to7, 3'h0, _io_in_11to7, 7'h33} : |_io_in_11to7 ? {_io_in_6to2,
                _io_in_11to7, 15'hE7} : {_io_in_6to2 | 5'h1, _io_in_11to7, 15'h73}) : {_io_in_6to2,
                |_io_in_6to2 ? {8'h0, _io_in_11to7, 7'h33} : {_io_in_11to7, |_io_in_11to7 ? 15'h67 :
                15'h1F}}}) : _io_in_14 ? {_io_in_13 ? {3'h0, _io_in_4to2, _io_in_12_3, _io_in_6to5, 11'h13}
                : {4'h0, io_in[3:2], _io_in_12_3, io_in[6:4], 10'h12}, _io_in_11to7, 7'h3} : _io_in_13 ?
                {3'h0, _io_in_4to2, _io_in_12_3, _io_in_6to5, 11'h13, _io_in_11to7, 7'h7} : {6'h0,
                _io_in_12_3, _io_in_6to2, _io_in_11to7, 3'h1, _io_in_11to7, 7'h13}) : _io_in_0 ? (_io_in_15
                ? (_io_in_14 ? {{4{_io_in_12_3}}, _io_in_6to5, _io_in_2, 7'h1, _io_in_9to7, 2'h0,
                _io_in_13, io_in[11:10], _io_in_4to3, _io_in_12_3, 7'h63} : _io_in_13 ? {_io_in_12_3,
                io_in[8], io_in[10:9], _io_in_6, io_in[7], _io_in_2, io_in[11], io_in[5:3],
                {9{_io_in_12_3}}, 12'h6F} : io_in[11] ? (_io_in_10 ? {1'h0, _io_in_6to5 == 2'h0, 7'h1,
                _io_in_4to2, 2'h1, _io_in_9to7, _io_in_12_3 ? {1'h0, _io_in_6_6 ? {1'h1, _io_in_5_7} :
                2'h0} : _io_in_6_6 ? {2'h3, _io_in_5_7} : {_io_in_5_7, 2'h0}, 2'h1, _io_in_9to7, 3'h3,
                _io_in_12_3, 3'h3} : {_T_4, _io_in_6to2, 2'h1, _io_in_9to7, 5'h1D, _io_in_9to7, 7'h13}) :
                {1'h0, _io_in_10, 4'h0, _io_in_12_3, _io_in_6to2, 2'h1, _io_in_9to7, 5'h15, _io_in_9to7,
                7'h13}) : {{3{_io_in_12_3}}, _io_in_14 ? (_io_in_13 ? (~(|_io_in_11to7) | _io_in_11to7 ==
                5'h2 ? {_io_in_4to3, _io_in_5, _io_in_2, _io_in_6, 4'h0, _io_in_11to7, 3'h0, _io_in_11to7,
                |_T_5 ? 7'h13 : 7'h1F} : {{12{_io_in_12_3}}, _io_in_6to2, _io_in_11to7, 3'h3, ~(|_T_5),
                3'h7}) : {{4{_io_in_12_3}}, _io_in_6to2, 8'h0, _io_in_11to7, 7'h13}) : {{4{_io_in_12_3}},
                _io_in_6to2, _io_in_11to7, 3'h0, _io_in_11to7, _io_in_13 ? {4'h3, ~(|_io_in_11to7), 2'h3} :
                7'h13}}) : _io_in_15 ? (_io_in_14 ? (_io_in_13 ? {4'h0, _io_in_6to5, _io_in_12_1, 2'h1,
                _io_in_4to2, 2'h1, _io_in_9to7, 3'h3, _io_in_11to10_2, 10'h23} : {5'h0, _io_in_5,
                _io_in_12, 2'h1, _io_in_4to2, 2'h1, _io_in_9to7, 3'h2, _io_in_11to10, _io_in_6, 9'h23}) :
                _io_in_13 ? {4'h0, _io_in_6to5, _io_in_12_1, 2'h1, _io_in_4to2, 2'h1, _io_in_9to7, 3'h3,
                _io_in_11to10_2, 10'h27} : {5'h0, _io_in_5, _io_in_12, 2'h1, _io_in_4to2, 2'h1,
                _io_in_9to7, 3'h2, _io_in_11to10, _io_in_6, 9'h2F}) : _io_in_14 ? {_io_in_13 ? {4'h0,
                _io_in_6to5, _io_in_12to10, 5'h1, _io_in_9to7, 5'hD} : {5'h0, _io_in_5, _io_in_12to10,
                _io_in_6, 4'h1, _io_in_9to7, 5'h9}, _io_in_4to2, 7'h3} : _io_in_13 ? {4'h0, _io_in_6to5,
                _io_in_12to10, 5'h1, _io_in_9to7, 5'hD, _io_in_4to2, 7'h7} : {2'h0, io_in[10:7],
                io_in[12:11], _io_in_5, _io_in_6, 12'h41, _io_in_4to2, |(io_in[12:5]) ? 7'h13 : 7'h1F};	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, Bitwise.scala:71:12, Cat.scala:30:58, Package.scala:18:26, :19:{12,17}, RVC.scala:22:14, :34:{26,35}, :37:{22,37}, :39:22, :44:{36,42,57,69,76}, :53:{20,22,29}, :77:{20,24}, :86:20, :90:{20,29}, :92:{10,14,21,27}, :103:30, :104:22, :105:43, :107:42, :122:{33,66}, :133:33, :134:{22,27}, :136:47, :137:33, :138:25, :139:10, :162:26
  assign io_out_rd = _io_in_1 ? (_io_in_0 | ~_io_in_15 | _io_in_14 | _io_in_13 ? _io_in_11to7 : _io_in_12_3 ?
                (|_io_in_6to2 ? _io_in_11to7 : 5'h1) : |_io_in_6to2 ? _io_in_11to7 : 5'h0) : _io_in_0 ?
                (_io_in_15 ? (_io_in_14 ? (_io_in_13 ? 5'h0 : _T_0) : _io_in_13 ? 5'h0 : _T_0) :
                _io_in_11to7) : _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, Cat.scala:30:58, Package.scala:19:12, RVC.scala:134:{22,27}, :138:25, :139:10
  assign io_out_rs1 = _io_in_1 ? (_io_in_0 ? io_in[19:15] : _io_in_15 ? (_io_in_14 | _io_in_13 ? 5'h2 :
                _io_in_12_3 | ~(|_io_in_6to2) ? _io_in_11to7 : 5'h0) : _io_in_14 | _io_in_13 ? 5'h2 :
                _io_in_11to7) : _io_in_0 ? (_io_in_15 ? _T_0 : ~_io_in_14 | _io_in_13 ? _io_in_11to7 :
                5'h0) : _io_in_15 | _io_in_14 | _io_in_13 ? _T_0 : 5'h2;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, Cat.scala:30:58, Package.scala:19:12, RVC.scala:20:57, :134:27, :139:10
  assign io_out_rs2 = _io_in_1 ? (_io_in_0 ? io_in[24:20] : _io_in_6to2) : _io_in_0 & _io_in_15 & _io_in_14 ?
                5'h0 : _T;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, Cat.scala:30:58, Package.scala:19:12, RVC.scala:20:79
  assign io_out_rs3 = io_in[31:27];	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, RVC.scala:20:101
  assign io_rvc = io_in[1:0] != 2'h3;	// /Users/andrewy/wsp/circt/integration_test/Dialect/FIRRTL/Regress/RocketCore.fir:4990:10, RVC.scala:162:{20,26}
endmodule

