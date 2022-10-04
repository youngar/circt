; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

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
