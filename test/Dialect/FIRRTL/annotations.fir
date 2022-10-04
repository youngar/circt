; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

; An Annotation without a target is promoted to a CircuitTarget Annotation. This
; represents a Scala FIRRTL Compiler `firrtl.annotations.NoTargetAnnotation`.
circuit Foo: %[[{"a":"a"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A legacy `firrtl.annotations.CircuitName` annotation becomes a CircuitTarget
; Annotation.
circuit Foo: %[[{"a":"a","target":"Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A CircuitTarget Annotation is attached to the circuit.
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A legacy `firrtl.annotations.ModuleName` annotation becomes a ModuleTarget
; Annotation
circuit Foo: %[[{"a":"a","target":"Foo.Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; A ModuleTarget Annotation is attached to the correct module.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; A ModuleTarget Annotation can be attached to an ExtModule.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar"}]]
  extmodule Bar:
    input a: UInt<1>
  module Foo:
    input a: UInt<1>
    inst bar of Bar
    bar.a <= a

    ; CHECK-LABEL: firrtl.circuit "Foo" {
    ; CHECK: firrtl.extmodule private @Bar
    ; CHECK-SAME: attributes {annotations = [{a = "a"}]}

; // -----

; A ReferenceTarget, ComponentName, or InstanceTarget pointing at an Instance
; should work.
circuit Foo: %[[
{"a":"a","target":"~Foo|Foo>bar"},
{"b":"b","target":"Foo.Foo.bar"},
{"c":"c","target":"~Foo|Foo/bar:Bar"}
]]
  module Bar:
    skip
  module Foo:
    inst bar of Bar
    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.hierpath  @nla_1 [@Foo::@bar, @Bar]
    ; CHECK: firrtl.module private @Bar
    ; CHECK-SAME annotations = [{c = "c"}]
    ; CHECK: firrtl.module @Foo
    ; CHECK: firrtl.instance bar
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; Test result annotations of InstanceOp.
; Must add inner_sym, if any subfield of a bundle type has nonlocal anchor.
; Otherwise, the nla will be illegal, without any inner_sym.
; Test on port and wire.
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar.a"},
                {"two":null,"target":"~Foo|Foo>bar.b.baz"},
                {"three":null,"target":"~Foo|Foo/bar:Bar>b.qux"},
                {"five":null,"target":"~Foo|Foo/bar:Bar>d.qux"},
                {"four":null,"target":"Foo.Foo.bar.c"}

                ]]
  module Bar:
    input a: UInt<1>
    output b: {baz: UInt<1>, qux: UInt<1>}
    output c: UInt<1>
    wire d: {baz: UInt<1>, qux: UInt<1>}
  module Foo:
    inst bar of Bar

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.hierpath @nla_2 [@Foo::@bar, @Bar]
    ; CHECK: firrtl.hierpath @nla_1 [@Foo::@bar, @Bar]
    ; CHECK: firrtl.module private @Bar
    ; CHECK:      out %b
    ; CHECK-NOT:  sym
    ; CHECK-SAME: [{circt.fieldID = 2 : i64, circt.nonlocal = @nla_1, three}]
    ; CHECK-NEXT: %d = firrtl.wire
    ; CHECK-NOT:  sym
    ; CHECK-SAME: {annotations = [{circt.fieldID = 2 : i64, circt.nonlocal = @nla_2, five}]}
    ; CHECK-SAME: : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
    ; CHECK: %bar_a, %bar_b, %bar_c = firrtl.instance bar
    ; CHECK-SAME: [{one}],
    ; CHECK-SAME: [{circt.fieldID = 1 : i64, two}],
    ; CHECK-SAME: [{four}]

; // -----

; A ReferenceTarget/ComponentName pointing at a CombMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    cmem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: chirrtl.combmem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a memory should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    mem bar:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.mem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; Test result annotations of MemOp.
circuit Foo: %[[{"a":null,"target":"~Foo|Foo>bar.r"},
                {"b":null,"target":"~Foo|Foo>bar.r.data.baz"},
                {"c":null,"target":"~Foo|Foo>bar.w.en"},
                {"d":null,"target":"~Foo|Foo>bar.w.data.qux"}]]
  module Foo:
    mem bar:
      data-type => {baz: UInt<8>, qux: UInt<8>}
      depth => 16
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.mem
    ; CHECK-SAME: portAnnotations = [
    ; CHECK-SAME: [{a}, {b, circt.fieldID = 5 : i64}],
    ; CHECK-SAME: [{c, circt.fieldID = 2 : i64}, {circt.fieldID = 6 : i64, d}]

; // -----

; A ReferenceTarget/ComponentName pointing at a node should work.  This
; shouldn't crash if the node is in a nested block.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.baz"}]]
  module Foo:
    input cond: UInt<1>[2]
    node bar = UInt<1>(0)
    when cond[0]:
      when cond[1]:
        node baz = UInt<1>(0)

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.node
    ; CHECK-SAME: annotations = [{a = "a"}
    ; CHECK: %baz = firrtl.node
    ; CHECK-SAME: annotations = [{b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a wire should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    wire bar: UInt<1>

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.wire
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a register should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.baz"}]]
  module Foo:
    input clock: Clock
    input reset: UInt<1>
    reg bar: UInt<1>, clock
    reg baz: UInt<1>, clock with : (reset => (reset, UInt<1>(0)))

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.reg
    ; CHECK-SAME: annotations = [{a = "a"}]
    ; CHECK: %baz = firrtl.regreset
    ; CHECK-SAME: annotations = [{b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at an SeqMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    smem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: chirrtl.seqmem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar>bar"},{"b":"b","target":"Foo.Foo.foo"}]]
  extmodule Bar:
    input bar: UInt<1>
  module Foo:
    input foo: UInt<1>
    inst bar of Bar
    bar.bar <= foo

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.extmodule private @Bar
    ; CHECK-SAME: [[_:.+]] [{a = "a"}]
    ; CHECK: firrtl.module @Foo
    ; CHECK-SAME: %foo: [[_:.+]] [{b = "b"}]

; // -----

; All types of JSON values should work
circuit Foo: %[[
  {
    "class": "circt.testNT",
    "string": "a",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null": null,
    "object": {
      "foo": "bar"
    },
    "array": [1, 2, 3]
  }
]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations =
    ; CHECK-SAME: array = [1, 2, 3]
    ; CHECK-SAME: boolean = true
    ; CHECK-SAME: float = 3.140
    ; CHECK-SAME: integer = 42
    ; CHECK-SAME: object = {foo = "bar"}
    ; CHECK-SAME: string = "a"

; // -----

; JSON escapes should work.
circuit Foo: %[[{"class": "circt.testNT", "\"":"}]]"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations =

; // -----
; JSON with a JSON-quoted string should be expanded.
circuit Foo: %[[{"class":"circt.testNT","a":"{\"b\":null}"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = {b}, class = "circt.testNT"}]}

; // -----
; JSON with a JSON-quoted number should stay as a number.
circuit Foo: %[[{"class":"circt.testNT","a":"0","b":0,"c":"\"0\""}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK:         firrtl.circuit "Foo"
    ; CHECK-SAME:      a = "0"
    ; CHECK-SAME:      b = 0
    ; CHECK-SAME:      c = "0"

; // -----
;
; A numeric "class" shouldn't crash the parser.

circuit Top : %[[
  {
    "class":"circt.test",
    "target":"~Top|Top>a"
  }
]]
  module Top :
    wire a: UInt<1>
    a is invalid

; CHECK-LABEL: firrtl.circuit "Top"
