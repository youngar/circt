// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Test: mutable cells are collapsed during elaboration

func.func @consume(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @mut_ops
rtg.test @mut_ops() {
  %zero = index.constant 0
  %ref = rtg.mut_create %zero : index
  %v = rtg.mut_read %ref : !rtg.mut<index>
  %one = index.constant 1
  %v2 = index.add %v, %one
  rtg.mut_write %ref, %v2 : !rtg.mut<index>
  %v3 = rtg.mut_read %ref : !rtg.mut<index>
  func.call @consume(%v3) : (index) -> ()
}
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// The second read sees the written value (zero + one = 1).
// CHECK-DAG: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @consume([[C1]])

// -----

// Test: multi-shot with mutable state — the handler mutates shared state between
// resumes; each shot of the continuation observes the accumulated value.

rtg.effect @tick : () -> ()
func.func @log_val(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @multi_shot_mut_state
rtg.test @multi_shot_mut_state() {
  %zero = index.constant 0
  %ref  = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @tick(%k: !rtg.continuation<none>) {
      rtg.resume %k : !rtg.continuation<none>
      %one = index.constant 1
      rtg.mut_write %ref, %one : !rtg.mut<index>
      rtg.resume %k : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @tick() : () -> none
      %v = rtg.mut_read %ref : !rtg.mut<index>
      func.call @log_val(%v) : (index) -> ()
      rtg.yield
    }
  }
}
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_read
// CHECK-NOT: rtg.mut_write
// First shot sees initial value (0); second shot sees mutated value (1).
// CHECK: [[C0:%.+]] = rtg.constant 0 : index
// CHECK: func.call @log_val([[C0]])
// CHECK: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @log_val([[C1]])

// -----

// Test: deep handler semantics — two sequential `perform` calls in the same
// body are both routed to the same handler. After the first perform's
// continuation runs, the handler must still be installed for the second
// perform to find. (Shallow handlers would handle only the first; the second
// would error as unhandled.) The handler increments a Mut on each call so we
// can observe the call count.

rtg.effect @ask : () -> index
func.func @record(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @deep_two_performs
rtg.test @deep_two_performs() {
  %zero = index.constant 0
  %ref  = rtg.mut_create %zero : index
  rtg.with_handlers {
    handle @ask(%k: !rtg.continuation<index>) {
      %v   = rtg.mut_read %ref : !rtg.mut<index>
      %one = index.constant 1
      %v1  = index.add %v, %one
      rtg.mut_write %ref, %v1 : !rtg.mut<index>
      rtg.resume %k, %v : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      %a = rtg.perform @ask() : () -> index
      func.call @record(%a) : (index) -> ()
      %b = rtg.perform @ask() : () -> index
      func.call @record(%b) : (index) -> ()
      rtg.yield
    }
  }
}
// Both performs are handled (no orphan-perform error). Handler ran twice;
// counter values 0 and 1 reach @record in order.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-DAG: [[C0:%.+]] = rtg.constant 0 : index
// CHECK: func.call @record([[C0]])
// CHECK-DAG: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @record([[C1]])

// -----

// Test: state visibility through an abort. With a multi-handler "state +
// abort" composition modeled by a Mut for state and a no-resume handler for
// abort, lexical ordering of writes matters: pre-perform writes from the
// body persist; the handler's own writes persist; post-perform writes from
// the body are skipped because the continuation was dropped. This pins the
// observable effect order so a future change cannot accidentally swap the
// handler-vs-body sequencing.

rtg.effect @abort_s : () -> ()
func.func @log_state(%arg0: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @abort_state_ordering
rtg.test @abort_state_ordering() {
  %z   = index.constant 0
  %ref = rtg.mut_create %z : index
  rtg.with_handlers {
    handle @abort_s(%k: !rtg.continuation<none>) {
      // Handler runs after the body's pre-perform write, so this overrides.
      %c1 = index.constant 1
      rtg.mut_write %ref, %c1 : !rtg.mut<index>
      // No resume: body's post-perform write must be skipped.
      rtg.yield
    }
    do {
      %c2 = index.constant 2
      rtg.mut_write %ref, %c2 : !rtg.mut<index>
      rtg.perform @abort_s() : () -> none
      // Skipped:
      %c3 = index.constant 3
      rtg.mut_write %ref, %c3 : !rtg.mut<index>
      rtg.yield
    }
  }
  %final = rtg.mut_read %ref : !rtg.mut<index>
  func.call @log_state(%final) : (index) -> ()
}
// Final state is 1 (handler's write); body's `2` was overwritten by the
// handler, and the post-perform `3` write never ran.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK-NOT: rtg.mut_write
// CHECK: [[C1:%.+]] = rtg.constant 1 : index
// CHECK: func.call @log_state([[C1]])

// -----

// Test: continuation escape via empty set + Mut. An empty
// !rtg.set<!rtg.continuation<T>> requires no initial continuation value and
// can be used as the initial value for a !rtg.mut. The handler stores %k in
// the set (without resuming), and the outer code retrieves and resumes it
// after with_handlers exits. This is "deferred continuation" semantics:
// after_handlers runs first, then after_perform runs when %k is resumed.

rtg.effect @deferred_e : () -> ()
func.func @after_perform_d() -> () { return }
func.func @after_handlers_d() -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @cont_escape_deferred
rtg.test @cont_escape_deferred() {
  %empty = rtg.set_create : !rtg.continuation<none>
  %ref   = rtg.mut_create %empty : !rtg.set<!rtg.continuation<none>>

  rtg.with_handlers {
    handle @deferred_e(%k: !rtg.continuation<none>) {
      // Stash %k without resuming (abort the body).
      %s = rtg.set_create %k : !rtg.continuation<none>
      rtg.mut_write %ref, %s : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
      rtg.yield
    }
    do {
      rtg.perform @deferred_e() : () -> none
      // Skipped by abort — becomes the continuation's remaining ops.
      func.call @after_perform_d() : () -> ()
      rtg.yield
    }
  }

  // Runs before the deferred continuation.
  func.call @after_handlers_d() : () -> ()

  // Retrieve and resume the escaped continuation.
  %s_out = rtg.mut_read %ref : !rtg.mut<!rtg.set<!rtg.continuation<none>>>
  %k_out = rtg.set_select_random %s_out : !rtg.set<!rtg.continuation<none>>
  rtg.resume %k_out : !rtg.continuation<none>
  rtg.yield
}
// after_handlers_d runs first; after_perform_d runs when %k_out is resumed.
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: func.call @after_handlers_d
// CHECK: func.call @after_perform_d

// -----

// Test: perform inside scf.if then-branch. The continuation is bounded to
// the then-block (delimited semantics). Code at the outer block level after
// the scf.if runs normally with the initial state -- it is NOT part of the
// continuation captured at the perform site.

rtg.effect @ev_if : () -> index
func.func @record_if(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @scfif_delimited_continuation
rtg.test @scfif_delimited_continuation() {
  %zero = index.constant 0
  %ref  = rtg.mut_create %zero : index
  %true = arith.constant true
  rtg.with_handlers {
    handle @ev_if(%k: !rtg.continuation<index>) {
      %c7 = index.constant 7
      rtg.resume %k, %c7 : !rtg.continuation<index>, index
      rtg.yield
    }
    do {
      // perform in then-branch; continuation = remainder of then-block only.
      scf.if %true {
        %v = rtg.perform @ev_if() : () -> index
        func.call @record_if(%v) : (index) -> ()
      }
      // This runs regardless: the scf.if finishes, outer block continues.
      // Mut was never written (else-branch not taken), so reads 0.
      %final = rtg.mut_read %ref : !rtg.mut<index>
      func.call @record_if(%final) : (index) -> ()
      rtg.yield
    }
  }
}
// Then-branch: record_if(7). Outer block: record_if(0).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK-NOT: rtg.mut_create
// CHECK: [[C7:%.+]] = rtg.constant 7 : index
// CHECK: func.call @record_if([[C7]])
// CHECK: [[C0:%.+]] = rtg.constant 0 : index
// CHECK: func.call @record_if([[C0]])

// -----

// Test: continuation passed as an effect INPUT argument.
// An escaped !rtg.continuation<index> (captured via set/mut) is forwarded as
// an operand to rtg.perform. The handler receives it as a block arg and
// resumes it directly with value 42 — proving continuations are valid effect
// operands and that the elaborator threads them through state correctly.

rtg.effect @relay_ci  : (!rtg.continuation<index>) -> ()
rtg.effect @supply_ci : () -> index
func.func @record_ci(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @cont_as_effect_input
rtg.test @cont_as_effect_input() {
  // Escape a !rtg.continuation<index> via abort + set/mut.
  %empty_ci = rtg.set_create : !rtg.continuation<index>
  %ref_ci   = rtg.mut_create %empty_ci : !rtg.set<!rtg.continuation<index>>
  rtg.with_handlers {
    handle @supply_ci(%k_s: !rtg.continuation<index>) {
      %s = rtg.set_create %k_s : !rtg.continuation<index>
      rtg.mut_write %ref_ci, %s : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
      rtg.yield
    }
    do {
      %val = rtg.perform @supply_ci() : () -> index
      func.call @record_ci(%val) : (index) -> ()
      rtg.yield
    }
  }
  %s_ci  = rtg.mut_read %ref_ci : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
  %k_held = rtg.set_select_random %s_ci : !rtg.set<!rtg.continuation<index>>

  // Pass the continuation as an effect input operand.
  rtg.with_handlers {
    handle @relay_ci(%cont_arg: !rtg.continuation<index>,
                     %k_r: !rtg.continuation<none>) {
      %c42 = index.constant 42
      rtg.resume %cont_arg, %c42 : !rtg.continuation<index>, index
      rtg.resume %k_r : !rtg.continuation<none>
      rtg.yield
    }
    do {
      rtg.perform @relay_ci(%k_held) : (!rtg.continuation<index>) -> none
      rtg.yield
    }
  }
}
// relay handler resumes the passed continuation with 42 → record_ci(42).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C42:%.+]] = rtg.constant 42 : index
// CHECK: func.call @record_ci([[C42]])

// -----

// Test: effect that RETURNS a continuation (!rtg.continuation<T> as effect
// result type). The handler for @give_cont resumes its own %k with a
// captured continuation value; the test body receives that continuation
// and resumes it with 99 — a second-order "give me a handle to resume later"
// pattern.

rtg.effect @give_cont_co : () -> !rtg.continuation<index>
rtg.effect @make_val_co  : () -> index
func.func @record_co(%v: index) -> () { return }

rtg.target @t : !rtg.dict<> {
  rtg.yield
}

// CHECK-LABEL: rtg.test @effect_returns_continuation
rtg.test @effect_returns_continuation() {
  // Escape a !rtg.continuation<index> to hand to @give_cont_co's handler.
  %empty_co = rtg.set_create : !rtg.continuation<index>
  %ref_co   = rtg.mut_create %empty_co : !rtg.set<!rtg.continuation<index>>
  rtg.with_handlers {
    handle @make_val_co(%k_mv: !rtg.continuation<index>) {
      %s = rtg.set_create %k_mv : !rtg.continuation<index>
      rtg.mut_write %ref_co, %s : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
      rtg.yield
    }
    do {
      %v = rtg.perform @make_val_co() : () -> index
      func.call @record_co(%v) : (index) -> ()
      rtg.yield
    }
  }
  %s_co   = rtg.mut_read %ref_co : !rtg.mut<!rtg.set<!rtg.continuation<index>>>
  %k_held_co = rtg.set_select_random %s_co : !rtg.set<!rtg.continuation<index>>

  // @give_cont_co's continuation arg expects !rtg.continuation<index> as the
  // resume value (since that is the declared effect result type).
  rtg.with_handlers {
    handle @give_cont_co(%k: !rtg.continuation<!rtg.continuation<index>>) {
      rtg.resume %k, %k_held_co : !rtg.continuation<!rtg.continuation<index>>,
                                   !rtg.continuation<index>
      rtg.yield
    }
    do {
      %k_got = rtg.perform @give_cont_co() : () -> !rtg.continuation<index>
      %c99 = index.constant 99
      rtg.resume %k_got, %c99 : !rtg.continuation<index>, index
      rtg.yield
    }
  }
}
// Body resumes the returned continuation with 99 → record_co(99).
// CHECK-NOT: rtg.with_handlers
// CHECK-NOT: rtg.perform
// CHECK-NOT: rtg.resume
// CHECK: [[C99:%.+]] = rtg.constant 99 : index
// CHECK: func.call @record_co([[C99]])
