#!/bin/bash
#cd 1d_r2c
#for i in 1 2 3 do
#do
#	./run.sh &> 1d_r2c_${i}
#done
#cd ../

#cd 2d_c2r
#for i in 1 2 3 
#do
#	./run.sh &> 2d_c2r_${i}
#done
#cd ../
cd 3d_c2c
for i in 1 2 3 do
do
	./run.sh &> 3d_c2c_${i}
done
cd ../
