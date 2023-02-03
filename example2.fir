circuit Foo:
  extmodule Bar_w2:
    input a: UInt<2>
    defname = Bar
    parameter w = 3

  extmodule Bar_w3:
    input a: UInt<3>
    defname = Bar
    parameter w = 3

  module Foo:
    input a: UInt<1>

    inst bar of Bar_w2
    bar.a <= a
    inst baz of Bar_w3
    baz.a <= a