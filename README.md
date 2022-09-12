# A Fused Architecture Supporting GEMM and SPMM on FPGA

The kernel is reproduced from https://github.com/eejlny/gemm_spmm

The host program is contained in `host.cpp`. The `xcl2.hpp` and `host.h` should be included when you build host.exe.

The kernel is contained in `kernelMatrixmult.cpp` and `kernelMatrixmult.h` in which multiple parameters are defined.

## Before You Start

Make sure you set up Vitis and XRT environment first so that libraries can be included on the system path.

## Build the Kernel

First you need to compile the kernel code using `-c` option and the Xilinx object `.xo` file will be generated.

Then you can link the Xilinx object `.xo` file to the platform using `-l` or `--link` option.
This will generate the Xilinx binary `.xclbin` file which will be used to program the FPGA.

```
v++ -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k mmult_top -o'kernelmult.hw.xo' kernelMatrixmult.cpp kernelMatrixmult.h xcl2.hpp
v++ -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 --link kernelmult.hw.xo --config mmult_top.cfg -o'kernelmult.hw.xclbin'
```

## Build the Host

Arbitrary precision type is used in this accelerator. The header file can be found in [HLS_arbitrary_Precision_Types](https://github.com/Xilinx/HLS_arbitrary_Precision_Types).

You can clone this and add to your working path. You can include it when building the host using `-I` option.

By running the following command, you can get `host.exe`.

```
g++ -g -std=c++14 -I$XILINX_XRT/include -L${XILINX_XRT}/lib/ -I/HLS_arbitrary_Precision_Types/include -o host.exe host.cpp host.h -lOpenCL -pthread -lrt -lstdc++
```

## Run the Application

When you successfully running commands above, you should have two files `kernelmult.hw.xclbin` and `host.exe` in your folder.

In both modes, one of matrices is read from data file so you should have this data file in the right format, such as [weights_byte_099.csv](https://github.com/RhysHuo/Fused_Multicores/blob/main/gemm_weights_byte_099.csv)for GEMM or [weights_byte_099.csr](https://github.com/RhysHuo/Fused_Multicores/blob/main/spmm_weights_byte_099.csr) for SPMM.

You can run this application using the following command:

```
./host.exe kernelmult.hw.xclbin <core_number> <mode> <data_file> <SN> <SM> <SP>
```
This accelerator can support up to 4 cores.

You can choose the mode between GEMM and SPMM. `0` for GEMM, `1` for SPMM.

If you choose GEMM mode, then you can set any matrix size `<SN>` `<SM>` `<SP>` up to the maximum `MAX_N`  `MAX_M`  `MAX_P`.
Matrix size : `A: SN*SM` `B: SM*SP` `C: SN*SP`

If you choose SPMM mode, then these three values `<SN>` `<SM>` `<SP>` are irrelevant and can be set to any values since in SPMM mode the matrix size is up to the data file you input.
