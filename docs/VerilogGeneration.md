# Verilog and SystemVerilog Generation

[Verilog](https://en.wikipedia.org/wiki/Verilog) and
 [SystemVerilog](https://en.wikipedia.org/wiki/SystemVerilog) are critical
 components of the hardware design tool
ecosystem, but generating syntatically correct Verilog that is acceptable by a
wide range
of tools is a challenge -- and generating "good looking" output even more so.
This document describes CIRCT's approach and support for generating Verilog and
SystemVerilog, some of the features and capabilities provided, and information
about the internal layering of the related subsystems.

## Why is this hard?

One of the goals of CIRCT is to insulate "front end" authors from the details of
Verilog generation.  We would like to see innovation at the authoring level, and
the problems in that space are quite different than the challenges of creating
syntactically correct Verilog.

Further, the Verilog/SystemVerilog languages were primarily designed to be a
human-authored programming language, and have evolved over the years with many
new and exciting features.  At the same time, the industry is full of critical
EDA tools - but these have mixed support for different language features.  Open
source tools in particular have [mixed
support](https://chipsalliance.github.io/sv-tests-results/) for new features.
Different emission styles also impact [simulator performance](http://www.sunburst-design.com/papers/CummingsDVCon2019_Yikes_SV_Coding_rev1_0.pdf)
and have many other considerations.  We would like clients of CIRCT to be insulated from this complexity where possible.

Beyond the capabilities of different tools, in many cases the output of CIRCT
is run through various "linters" that look for antipatterns or possible bugs in
the output.  While it is difficult to work with arbitrary 3rd party linters, we
would like the output of CIRCT-based tools to be as "lint clean by definition"
as possible.

Finally, our goal is for the generated Verilog to be as readable and polished as
possible - some users of CIRCT generate IP that is sold to customers, and the
quality of the generated Verilog directly reflects on the quality of the
corresponding products.  This means that small details, including indentation
and use of the correct idioms is important.

## Controlling output style with `LoweringOptions`

The primary interface to control the output style from a CIRCT-based tool is
through the [`circt::LoweringOptions`](https://github.com/llvm/circt/blob/main/include/circt/Support/LoweringOptions.h)
structure.  It contains a number of properties (e.g. `emittedLineLength` or
`disallowLocalVariables`) that affect lowering and emission of Verilog -- in
this case, what length of lines the emitter should aim for (e.g. 80 columns
wide, 120 wide, etc), and whether the emitter is allowed to use `automatic
logic` declarations in nested blocks or not.

The defaults in `LoweringOptions` are set up to generate aethetically pleasing
output, and to use the modern features of SystemVerilog where possible.  Client
tools and frontends can change these, e.g. if they need to generate standard
Verilog for older tools.

Command line tools generally provide a `--lowering-options=` flag that
allows end-users to override the defaults or the front-end provided features.
If you're using `firtool` for example, you can pass
`--lowering-options=emittedLineLength=200` to change the line length.  This can
be useful for experimentation, or when a frontend doesn't have other ways to
control the output.

The current set of "tool capability" Lowering Options is:

 * `noAlwaysComb` (default=`false`).  If true, emits `sv.alwayscomb` as Verilog
   `always @(*)` statements.  Otherwise, print them as `always_comb`.
 * `exprInEventControl` (default=`false`).   If true, expressions are
   allowed in the sensitivity list of `always` statements, otherwise they are
   forced to be simple wires. Some EDA tools rely on these being simple wires.
 * `disallowPackedArrays` (default=`false`).  If true, eliminate packed arrays
   for tools that don't support them (e.g. Yosys).
 * `disallowLocalVariables` (default=`false`).  If true, do not emit
   SystemVerilog locally scoped "automatic" or logic declarations - emit top
   level wire and reg's instead.
 * `verifLabels` (default=`false`).  If true, verification statements
   like `assert`, `assume`, and `cover` will always be emitted with a label. If
   the statement has no label in the IR, a generic one will be created. Some EDA
   tools require verification statements to be labeled.
  
The current set of "style" Lowering Options is:

 * `emittedLineLength` (default=`90`).  This is the target width of lines in an
   emitted Verilog source file in columns.
 * `locationInfoStyle` (default=`plain`).  This option controls emitted location
   information style.  The available styles are:
   * `plain`: `// perf/regress/AndNot.fir:3:10, :7:{10,17}`
   * `wrapInAtSquareBracket`: `// @[perf/regress/AndNot.fir:3:10, :7:{10,17}]`
 * `disallowPortDeclSharing` (default=`false`).  If true, emit one port per
   declaration.  Instead of `input a,\n b` this will produce
   `input a,\n input b`.  When false, ports are emitted using the same
   declaration when possible.

### Specifying `LoweringOptions` in a front-end HDL tool

The [`circt::LoweringOptions` struct itself](https://github.com/llvm/circt/blob/main/include/circt/Support/LoweringOptions.h) 
is very simple: it projects each of the lowering options as a boolean, integer
or other property.  This allows C++ code to set up and query these properties
with a natural and easy to use API.

That said, this struct is merely a convenience  the actual truth is encoded into
the IR as a `circt.loweringOptions` string attribute on the top level
`builtin.module` declaration.  Any frontend can set these options by setting
this attribute on the IR that they generate.

## Adding new Lowering Options

Adding new `LoweringOptions` is pretty easy, but we want to be able to scale to
having lots of these and want them to remain as consistent as we can.  Please
follow these guidelines when adding new things:

1) Don't use `LoweringOptions` to change the semantics of IR nodes in
   ExportVerilog.  Instead, add new IR nodes to model the different semantic
   concepts that you need, and query `LoweringOptions` to decide what construct
   to lower to.

2) Make the default setting of the flag generate modern and clean SystemVerilog
   code.  Flags should make the output more conservative/verbose/boring.

3) Name boolean options with active verb and reuse the existing ones (e.g.
   `disallow`) where possible.  Try to make new options consistently named.

4) Consider making `ExportVerilog` reject invalid constructs with an error
   message - a compiler bug that causes CIRCT to generate the wrong construct
   is pretty certain to be better diagnosed by CIRCT itself than by the EDA tool
   that consumes the output.

5) Keep this documentation up to date.

## Using the Verilog Exporter in a PassManager pipeline

When building a new compiler, you get to decide what order to run passes in,
and the order of passes can greatly affect the quality of the generated IR.
There are many ways to do this, but we'd recommend you follow the example of a
well maintained tool in CIRCT (e.g. `firtool`).  Something like this as the end
of your pipeline should work well:

```
  // Optional: perform general cleanups and structure the modules in a
  // consistent way.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(sv::createHWCleanupPass());
  modulePM.addPass(createCSEPass());
  modulePM.addPass(createSimpleCanonicalizerPass());

  // Required: Legalize unsupported operations within modules.  Do not run
  // passes after this that aren't aware of LoweringOptions.
  modulePM.addPass(sv::createHWLegalizeModulesPass());

  // Optional: Tidy up the IR to improve verilog emission quality.
  modulePM.addPass(sv::createPrettifyVerilogPass());

  // Actually export the module.
  exportVerilog(theModule, ...);
```

## Signal naming

ExportVerilog checks all signal (and instance) names for keyword conflicts and
duplicated names. Whenever these conditions are encountered, ExportVerilog will
change the name to avoid the conflict.

### Ops with explicit names

The `sv.reg` operation, `sv.wire` operation, and the various instance operations
in the hw dialect have a `name` attribute which gets used.

### Out-of-line expressions ("temporaries")

Expressions are sometimes not emitted inlined (out-of-line) for a variety of
reasons. These wire (or `automatic logic`) names or existence is **not
guaranteed** to be stable. (Meaning they could change for *any* reason.) They,
therefore, should not be relied upon for anything but local waveform debugging.
In general, any name with prefixed with an underscore should not be relied upon.

These names come from the following sources, listed here in proirity order:
1. The `sv.namehint` dialect attribute (which can be attached to any operation).
2. If ExportVerilog has a rule to derive a name based on the operation and
operands, it will do so. (e.g. `hw.extract` operations get name
`_<operandName>_<highBit>to<lowBit>` as of writing).
3. `_T` or `_T_x` wherein `x` is a number.

## `exportVerilog` Internals

It turns out that producing syntactically correct Verilog that is also pretty is
really hard.  As such, we've taken a few steps to improve separation of concerns
and thus simplify the implementation of Verilog emission.  It is important to
understand the division of responsibilities between these components when adding
new features or fixing bugs in the existing code.

In particular, we split responsibilities between three major components, which
are run in this order:

1) The optional [`PrettifyVerilog` pass](https://github.com/llvm/circt/blob/main/lib/Dialect/SV/Transforms/PrettifyVerilog.cpp).
   It is never required for correctness,  but has a major impact on
   "prettiness", and should always be used in practice.
2) The mandatory [`PrepareForEmission` logic] that is built into the
   `exportVerilog` function, and is thus mandatory.
3) The core [`ExportVerilog`](https://github.com/llvm/circt/blob/main/lib/Translation/ExportVerilog/ExportVerilog.cpp) logic, which handles printing out
   of Verilog source code.

The first two of these are highly parameterized on `LoweringOptions`, and we're
trying to minimize the complexity in the last one.  Let's discuss each of them
in inverse order.

### The core `ExportVerilog` logic

The core of `ExportVerilog` walks the IR and prints out syntactically correct
verilog to an `llvm::raw_ostream`.

TODO: Talk about line splitting, preorder traversal, NameCollector, etc.  Cross
block references always go through a temporary.  This doesn't support cyclic
graph region references in the top level of a hw.module.

Because the Prepass logic has already been run, it knows it doesn't have to
handle invalid output - it will have already been lowered by Prepass or other
things earlier in the pipeline.  As such, it can just diagnose any invalid
things that may have slipped through with an error.

### The `PrepareForEmission` logic built into `ExportVerilog`

This functionality is a logically distinct lowering pass that happens to run as
part of ExportVerilog (so it cannot be forgotten or drift away from the
exporter).  It is structured as a lowering pass that rewrites invalid constructs
as lower level ones, e.g. injecting explicit wires in places to break cyclic
combinational logic circuits or duplicating array indexing operations into the
same block as the uses (so they'll be sure to be inlined).

`PrepareForEmission` also collects some information about local names that are
used by the emitter later.

### `PrettifyVerilog` Internals

`PrettifyVerilog` is an optional pass that is run right before the Verilog
emitter.  It introduces prettier but non-canonical forms of expressions that
align with user expectations: for example, we print "a-1" instead of "a+255".

When in SystemVerilog mode, this pass sinks expressions into nested regions
(e.g. into procedural if regions) whenever it can to encourage inline emission
of subexpressions.  It also moves instances to the end of the module, which
eliminates temporaries and makes the output more predictable.
