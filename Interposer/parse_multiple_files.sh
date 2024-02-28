#!/bin/bash
for mf in `find -name '*ptx'`; do
	echo $mf
	./../../../ptx_parser/parse_ptx_v2.py $mf ../my_cublas_ptx/$mf 
done
