// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-aos-to-soa))' | FileCheck %s

firrtl.circuit "AOSToSOATestCircuit" {
  // CHECK-LABEL: @Test1
  firrtl.module @Test1() {

  }

  // CHECK-LABEL: @Test2
  firrtl.module @Test2() {

  }
}
