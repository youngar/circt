; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; COM: A non-local Annotation is unsupported.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: Unsupported target (not a local CircuitTarget, ModuleTarget, or ReferenceTarget without subfield or subindex) */ "~Foo|Foo/bar:Bar"}}
circuit Foo: %[[{"a":"a","target":"~Foo|Foo/bar:Bar"}]]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is not an array.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array, but found something else. */}}
circuit Foo: %[{"a":"a"}]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is not an array of objects.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array of objects, but found an array of something else. */}}
circuit Foo: %[[{"a":"a"},[{"b":"b"}]]]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is invalid. (There's a trailing comma.)

; expected-error @+2 {{Failed to parse JSON Annotations}}
; expected-note @+1 {{}}
circuit Foo: %[[{"a":"a"},]]]
  module Foo:
    skip

; // -----

; COM: A custom annotation fails to parse because it is missing fields

; expected-error @+2 {{did not contain required key}}
; expected-note @+1 {{}}
circuit Foo: %[
[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation"
  }
]]
  module Foo:
    skip

; // -----

; COM: An empty target should report an error (and not crash)


; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: invalid target string */}}
circuit Foo: %[[{"target":""}]]
  module Foo:
    skip

; // -----

; COM: A null target should report an error (and not crash)

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: target must be a string type */}}
circuit Foo: %[[{"target":null}]]
  module Foo:
    skip
