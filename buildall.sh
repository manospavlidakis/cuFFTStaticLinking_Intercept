#!/bin/bash
for mf in `find -name 'CMakeLists.txt'`; do                                                          cd `dirname $mf`
	#mkdir -p build
	#cd build
	#~/cmake-3.24.0/bin/cmake ..
	make -j
	pwd
	cd -
	pwd
done

