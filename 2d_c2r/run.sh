#echo "JIT+MODIFIED"
#export CUDA_FORCE_PTX_JIT=1; time LD_PRELOAD=./../Interposer/libmylib_allocator.so  ./2d_c2r_example
echo "**************************************"
echo "JIT+NATIVE"
export CUDA_FORCE_PTX_JIT=1; ./2d_c2r_example

#echo "**************************************"
#echo "NATIVE"
#export CUDA_FORCE_PTX_JIT=0; ./a.out
