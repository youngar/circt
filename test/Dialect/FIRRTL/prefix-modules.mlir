// RUN: circt-opt --pass-pipeline="firrtl.circuit(firrtl-prefix-modules)" %s | FileCheck %s

// Check that the circuit is updated when the main module is updated.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
  }
}


// Check that the circuit is not updated if the annotation is non-inclusive.
// CHECK: firrtl.circuit "Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
  }
}


// Check that basic module prefixing is working.
firrtl.circuit "Top" {
  // The annotation should be removed.
  // CHECK:  firrtl.module @Top() {
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Check that memories are renamed.
firrtl.circuit "Top" {
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
    // CHECK: name = "T_ram"
    %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}

// Check that memory modules are renamed.
// CHECK-LABEL: firrtl.circuit "MemModule"
firrtl.circuit "MemModule" {
  // CHECK: firrtl.memmodule @T_MWrite_ext
  firrtl.memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataType = !firrtl.uint<42>, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.module @MemModule()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    // CHECK: firrtl.instance MWrite_ext  @T_MWrite_ext
    %0:4 = firrtl.instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
}

// Check that external modules are not renamed.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.extmodule @ExternalModule
  firrtl.extmodule @ExternalModule()

  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    firrtl.instance ext @ExternalModule()
  }
}


// Check that the module is not cloned more than necessary.
firrtl.circuit "Top0" {
  firrtl.module @Top0()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  firrtl.module @Top1()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Complex nested test.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK: firrtl.instance test @T_Aardvark()
    firrtl.instance test @Aardvark()

    // CHECK: firrtl.instance test @T_Z_Zebra()
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Aardvark
  firrtl.module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK: firrtl.instance test @T_A_Z_Zebra()
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Z_Zebra
  // CHECK: firrtl.module @T_A_Z_Zebra
  firrtl.module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}


// Updates should be made to a Grand Central interface to add a "prefix" field
// and the annotations associated with the parent and companion should have
// their "name" field prefixed.
// CHECK-LABEL: firrtl.circuit "GCTInterfacePrefix"
// CHECK-SAME:    name = "MyView", prefix = "FOO_"
firrtl.circuit "GCTInterfacePrefix"
  attributes {annotations = [{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "MyInterface",
    elements = [],
    id = 0 : i64,
    name = "MyView"}]}  {
  // CHECK:      firrtl.module @FOO_MyView_companion
  // CHECK-SAME:   name = "FOO_MyView"
  firrtl.module @MyView_companion()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.ViewAnnotation",
      id = 0 : i64,
      name = "MyView",
      type = "companion"}]} {}
  // CHECK:      firrtl.module @FOO_DUT
  // CHECK-SAME:   name = "FOO_MyView"
  firrtl.module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation",
       id = 0 : i64,
       name = "MyView",
       type = "parent"},
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    firrtl.instance MyView_companion  @MyView_companion()
  }
  firrtl.module @GCTInterfacePrefix() {
    firrtl.instance dut @DUT()
  }
}

// CHECK: firrtl.circuit "T_NLATop"
firrtl.circuit "NLATop" {

  firrtl.nla @nla [#hw.innerNameRef<@NLATop::@test>, #hw.innerNameRef<@Aardvark::@test>, @Zebra]
  firrtl.nla @nla_1 [#hw.innerNameRef<@NLATop::@test>,#hw.innerNameRef<@Aardvark::@test_1>, @Zebra]
  // CHECK: firrtl.nla @nla [#hw.innerNameRef<@T_NLATop::@test>, #hw.innerNameRef<@T_Aardvark::@test>, @T_A_Z_Zebra]
  // CHECK: firrtl.nla @nla_1 [#hw.innerNameRef<@T_NLATop::@test>, #hw.innerNameRef<@T_Aardvark::@test_1>, @T_A_Z_Zebra]
  // CHECK: firrtl.module @T_NLATop
  firrtl.module @NLATop()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK:  firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @T_Aardvark()
    firrtl.instance test  sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"} ]}@Aardvark()

    // CHECK: firrtl.instance test2 @T_Z_Zebra()
    firrtl.instance test2 @Zebra()
  }

  // CHECK: firrtl.module @T_Aardvark
  firrtl.module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK:  firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @T_A_Z_Zebra()
    firrtl.instance test sym @test {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]}@Zebra()
    firrtl.instance test1 sym @test_1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}@Zebra()
  }

  // CHECK: firrtl.module @T_Z_Zebra
  // CHECK: firrtl.module @T_A_Z_Zebra
  firrtl.module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}

// Prefixes should be applied to Grand Central Data or Mem taps.  Check that a
// multiply instantiated Data/Mem tap is cloned ("duplicated" in Scala FIRRTL
// Compiler terminology) if needed.  (Note: multiply instantiated taps are
// completely untrodden territory for Grand Central.  However, the behavior here
// is the exact same as how normal modules are cloned.)
//
// CHECK-LABLE: firrtl.circuit "GCTDataMemTapsPrefix"
firrtl.circuit "GCTDataMemTapsPrefix" {
  // CHECK:      firrtl.extmodule @FOO_DataTap
  // CHECK-SAME:   defname = "FOO_DataTap"
  firrtl.extmodule @DataTap()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapsAnnotation"}],
      defname = "DataTap"}
  // The Mem tap should be prefixed with "FOO_" and cloned to create a copy
  // prefixed with "BAR_".
  //
  // CHECK:      firrtl.extmodule @FOO_MemTap
  // CHECK-SAME:   defname = "FOO_MemTap"
  // CHECK:      firrtl.extmodule @BAR_MemTap
  // CHECK-SAME:   defname = "BAR_MemTap"
  firrtl.extmodule @MemTap(
    out mem: !firrtl.vector<uint<1>, 1>
      [#firrtl.subAnno<fieldID = 1, {
        class = "sifive.enterprise.grandcentral.MemTapAnnotation",
        id = 0 : i64,
        word = 0 : i64}>])
    attributes {defname = "MemTap"}
  // Module DUT has a "FOO_" prefix.
  firrtl.module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    // CHECK: firrtl.instance d @FOO_DataTap
    firrtl.instance d @DataTap()
    // CHECK: firrtl.instance m @FOO_MemTap
    %a = firrtl.instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  // Module DUT2 has a "BAR_" prefix.
  firrtl.module @DUT2()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "BAR_",
       inclusive = true}]} {
    // CHECK: firrtl.instance m @BAR_MemTap
    %a = firrtl.instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  firrtl.module @GCTDataMemTapsPrefix() {
    firrtl.instance dut @DUT()
    firrtl.instance dut @DUT2()
  }
}

// Test the the NonLocalAnchor is properly updated.
// CHECK-LABEL: firrtl.circuit "FixNLA" {
  firrtl.circuit "FixNLA"   {
    firrtl.nla @nla_1 [#hw.innerNameRef<@FixNLA::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
    // CHECK:   firrtl.nla @nla_1 [#hw.innerNameRef<@FixNLA::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
    firrtl.nla @nla_2 [#hw.innerNameRef<@FixNLA::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@s1>]
    // CHECK:   firrtl.nla @nla_2 [#hw.innerNameRef<@FixNLA::@foo>, #hw.innerNameRef<@X_Foo::@bar>, #hw.innerNameRef<@X_Bar::@baz>, #hw.innerNameRef<@X_Baz::@s1>]
    firrtl.nla @nla_3 [#hw.innerNameRef<@FixNLA::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
    // CHECK:   firrtl.nla @nla_3 [#hw.innerNameRef<@FixNLA::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
    firrtl.nla @nla_4 [#hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
    // CHECK:       firrtl.nla @nla_4 [#hw.innerNameRef<@X_Foo::@bar>, #hw.innerNameRef<@X_Bar::@baz>, @X_Baz]
    // CHECK-LABEL: firrtl.module @FixNLA()
    firrtl.module @FixNLA() {
      firrtl.instance foo sym @foo  {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @Foo()
      firrtl.instance bar sym @bar  {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_3, class = "circt.nonlocal"}]} @Bar()
      // CHECK:   firrtl.instance foo sym @foo  {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @X_Foo()
      // CHECK:   firrtl.instance bar sym @bar  {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_3, class = "circt.nonlocal"}]} @Bar()
    }
    firrtl.module @Foo() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "X_"}]} {
      firrtl.instance bar sym @bar  {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}, {circt.nonlocal = @nla_4, class = "circt.nonlocal"}]} @Bar()
    }
    // CHECK-LABEL:   firrtl.module @X_Foo()
    // CHECK:         firrtl.instance bar sym @bar  {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}, {circt.nonlocal = @nla_4, class = "circt.nonlocal"}]} @X_Bar()

    // CHECK-LABEL:   firrtl.module @Bar()
    firrtl.module @Bar() {
      firrtl.instance baz sym @baz  {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}, {circt.nonlocal = @nla_3, class = "circt.nonlocal"}, {circt.nonlocal = @nla_4, class = "circt.nonlocal"}]} @Baz()
      // CHECK:     firrtl.instance baz sym @baz  {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_3, class = "circt.nonlocal"}]} @Baz()
    }
    // CHECK-LABEL: firrtl.module @X_Bar()
    // CHECK:       firrtl.instance baz sym @baz  {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}, {circt.nonlocal = @nla_4, class = "circt.nonlocal"}]} @X_Baz()

    firrtl.module @Baz() attributes {annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}, {circt.nonlocal = @nla_4, class = "nla_4"}]} {
      %mem_MPORT_en = firrtl.wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
    }
    // CHECK-LABEL: firrtl.module @X_Baz()
    // CHECK-SAME:  annotations = [{circt.nonlocal = @nla_4, class = "nla_4"}]
    // CHECK:       %mem_MPORT_en = firrtl.wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
    // CHECK:       firrtl.module @Baz()
    // CHECK_SAME:  annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}]
    // CHECK:       %mem_MPORT_en = firrtl.wire sym @s1  : !firrtl.uint<1>
    
  }

// Test that NonLocalAnchors are properly updated with memmodules.
firrtl.circuit "Test"   {
  // CHECK: firrtl.nla @nla_1 [#hw.innerNameRef<@Test::@foo1>, #hw.innerNameRef<@A_Foo1::@bar>, @A_Bar]
  firrtl.nla @nla_1 [#hw.innerNameRef<@Test::@foo1>, #hw.innerNameRef<@Foo1::@bar>, @Bar]
  // CHECK: firrtl.nla @nla_2 [#hw.innerNameRef<@Test::@foo2>, #hw.innerNameRef<@B_Foo2::@bar>, @B_Bar]
  firrtl.nla @nla_2 [#hw.innerNameRef<@Test::@foo2>, #hw.innerNameRef<@Foo2::@bar>, @Bar]

  firrtl.module @Test() {
    firrtl.instance foo1 sym @foo1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @Foo1()
    firrtl.instance foo2 sym @foo2 {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @Foo2()
  }

  firrtl.module @Foo1() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "A_"}]} {
    firrtl.instance bar sym @bar {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @Bar()
  }

  firrtl.module @Foo2() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "B_"}]} {
    firrtl.instance bar sym @bar {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]} @Bar()
  }

  // CHECK: firrtl.memmodule @A_Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}]
  // CHECK: firrtl.memmodule @B_Bar() attributes {annotations = [{circt.nonlocal = @nla_2, class = "test2"}]
  firrtl.memmodule @Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}, {circt.nonlocal = @nla_2, class = "test2"}], dataType = !firrtl.uint<1>, depth = 16 : ui64, extraPorts = [], maskBits = 0 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32,  writeLatency = 1 : ui32}
}
