#!/bin/bash
##===- utils/equiv-rtl.sh - Formal Equivalence via yosys------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script checks two input verilog files for equivalence using yosys.
#
# Usage equiv-rtl.sh File1.v File2.v TopLevelModuleName
#
##===----------------------------------------------------------------------===##
if [ "$4" != "" ]; then
    mdir=$4
else
    mdir=.
fi

echo "==== First One ===="
/usr/bin/time yosys -q -p "
 read_verilog $1
 rename $3 top1
 proc
 memory
 flatten top1
 hierarchy -libdir $mdir -top top1
 read_verilog $2
 rename $3 top2
 proc
 memory
 flatten top2

 equiv_make top1 top2 equiv
 hierarchy -top equiv
 clean -purge
 opt -purge
 equiv_simple -short
 equiv_induct
 equiv_status -assert
"
if [ $? -eq 0 ]
then
  echo "PASS,INDUCT"
  #exit 0
fi


echo "==== Second One ===="
echo "Comparing $1 and $2 with $3 Missing Dir $mdir"
/usr/bin/time yosys -q -p "
 read_verilog $1
 rename $3 top1
 proc
 memory
 flatten top1
 hierarchy -libdir $mdir -top top1
 read_verilog $2
 rename $3 top2
 proc
 memory
 flatten top2

 equiv_make top1 top2 equiv
 hierarchy -top equiv
 clean -purge
 opt -purge
 equiv_simple -undef
 equiv_induct -undef
 equiv_status -assert
"
if [ $? -eq 0 ]
then
  echo "PASS,INDUCT"
  #exit 0
fi

#repeat with sat
echo "==== Third One ===="
echo "Trying SAT $1 and $2 with $3 Missing Dir $mdir"
/usr/bin/time yosys -q -p "
 read_verilog $1
 rename $3 top1
 proc
 memory
 flatten top1
 hierarchy -top top1
 read_verilog $2
 rename $3 top2
 proc
 memory
 flatten top2
 opt
 miter -equiv -make_assert -flatten top1 top2 equiv
 hierarchy -top equiv
 opt
 sat -prove-asserts -seq 4 -verify
"
if [ $? -eq 0 ]
then
  echo "PASS,SAT"
  #exit 0
fi

echo "FAIL"
exit 1
