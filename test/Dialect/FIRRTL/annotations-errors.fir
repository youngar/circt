; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; Annotations JSON is not an array.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array, but found something else. */}}
circuit Foo: %[{"a":"a"}]
  module Foo:
    skip

; // -----

; Annotations JSON is not an array of objects.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array of objects, but found an array of something else. */}}
circuit Foo: %[[[{"class":"b"}]]]
  module Foo:
    skip

; // -----

; Annotations JSON is invalid. (There's a trailing comma.)

; expected-error @+2 {{Failed to parse JSON Annotations}}
; expected-note @+1 {{Invalid JSON value}}
circuit Foo: %[[{"a":"a"},]]
  module Foo:
    skip
    
; // -----

; An unknown annotation should error.

; expected-error @below {{Unhandled annotation}}
circuit Foo: %[[{"class":"circt.unknown"}]]
  module Foo:
    skip

; // -----

; An incorrect circuit target should report an error.
;
; expected-error @+2 {{circuit name doesn't match annotation}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Bar|Foo>bar"
  }
]]
  module Foo:
    skip

; // -----

; An incorrect circuit target should report an error.
;
; expected-error @+2 {{circuit name doesn't match annotation}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Bar"
  }
]]
  module Foo:
    skip

; // -----

; An empty target string should be illegal.

; expected-error @+2 {{Cannot tokenize annotation path}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":""
  }
]]
  module Foo:

; // -----

; A target that does a subindex of an instance should be illegal.

; expected-error @+2 {{illegal target '~Foo|Foo>bar[0]' indexes into an instance}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>bar[0]"
  }
]]
  module Bar:
  module Foo:
    inst bar of Bar

; // -----

; A target that uses a string for an index should be illegal.

; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>bar[a].baz"
  }
]]
  module Foo:
    ; expected-error @+1 {{Cannot convert 'a' to an integer}}
    wire bar : {baz : UInt<1>, qux : UInt<1>}[2]
    
  
; // -----

; Invalid subindex or subfield targets are checked.

; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>bar[1][42]"
  }
]]
  module Foo:
    input clock : Clock
    ; expected-error @+1 {{index access '42' into non-vector type}}
    reg bar : {baz : UInt<1>, qux : UInt<1>}[2], clock

; // -----

; Invalid subindex or subfield targets are checked.

; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>bar[1].qnx"
  }
]]
  module Foo:
    input clock : Clock
    ; expected-error @+1 {{cannot resolve field 'qnx' in subtype}}
    reg bar : {baz : UInt<1>, qux : UInt<1>}[2], clock


; // -----

; Invalid subindex or subfield targets are checked.

; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>bar[1].baz[1337]"
  }
]]
  module Foo:
    input clock : Clock
    ; expected-error @+1 {{index access '1337' into non-vector type}}
    reg bar : {baz : UInt<1>, qux : UInt<1>}[2], clock


; // -----

; A target on a non-existent module should error.

; expected-error @+2 {{module doesn't exist 'Bar'}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Bar"
  }
]]
  module Foo:
    skip

; // -----

; A target on a non-existent component should error.

; expected-error @+2 {{cannot find name 'x' in Foo}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo>x"
  }
]]
  module Foo:
    skip

; // -----

; A non-local annotation on a non-existent instance should error.

; expected-error @+2 {{cannot find instance 'baz' in 'Foo'}}
; expected-error @+1 {{Unable to resolve target of annotation}}
circuit Foo: %[[
  {
    "class":"circt.test",
    "target":"~Foo|Foo/baz:Bar"
  }
]]
  module Bar:
    skip
  module Foo:
    inst bar of Bar

; // -----

; expected-error @+1 {{Unable to apply annotation}}
circuit LocalOnlyAnnotation: %[[
  {
     "class":"circt.testLocalOnly",
     "target":"~LocalOnlyAnnotation|LocalOnlyAnnotation/foo:Foo>w"
  }
]]
  module Foo:
    ; expected-error @+2 {{targeted by a non-local annotation}}
    ; expected-note @+1 {{see current annotation}}
    wire w : UInt<1>
  module LocalOnlyAnnotation:
    inst foo of Foo
  
; // -----

; expected-error @+1 {{Unable to apply annotation}}
circuit DontTouchOnNonReferenceTarget: %[[
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
     "target":"~DontTouchOnNonReferenceTarget|Submodule"
  },
  {
    "class":"firrtl.transforms.DontTouchAnnotation",
     "target":"~DontTouchOnNonReferenceTarget|DontTouchOnNonReferenceTarget>submodule"
  }
]]
  module Submodule:
    skip
  module DontTouchOnNonReferenceTarget:
    inst submodule of Submodule