#echo "JIT+MODIFIED"
#export CUDA_FORCE_PTX_JIT=1; time LD_PRELOAD=./../Interposer/libmylib_allocator.so  ./1d_r2c_example
echo "**************************************"
echo "JIT+NATIVE"
export CUDA_FORCE_PTX_JIT=1; ./1d_r2c_example

#echo "**************************************"
#echo "NATIVE"
#export CUDA_FORCE_PTX_JIT=0; ./a.out
