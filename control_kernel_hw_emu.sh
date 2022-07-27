git clone https://github.com/RhysHuo/Fused_Vitis_Updata01.git
cd Fused_Vitis_Updata01
cp kernelMatrixmult_sw.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Vitis_Updata01
v++ -t hw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k kernelmult1_sw -o'kernelmult1.hw_emu.xo' kernelMatrixmult_sw.cpp kernelMatrixmult.h xcl2.hpp
