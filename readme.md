
# project construct
```
src:
積分Gridの生成
dft_lebedev.cpp
dft_atomGrid.cpp
dft_partition_gpu.cu

積分Gridの離散化
dft_aoEVAL.cpp
dft_aoEVAL_kernel.cu
dft_shell_eval.cpp

積分GridのAPI
dft_genGrid.cpp

DFTのSCF
LDA.py


```
# using method

## prepare mol

copy mol to get_coords.py

```
python get_coords.py
```

then copy the [mol_name].txt to 

work/xyz

## cmake

```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native ..
cmake --build . -j
```



## generate Grids and Aos

```
cd ./work
./test_atomgrid H2O sto-3g
```



H2O 分子の名前

sto-3g 基底関数







