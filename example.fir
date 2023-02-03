circuit Main:
  module Main:
    parameter n: Nat
    output o: UInt<n>
    o <= UInt<n>(0)
