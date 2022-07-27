git clone https://github.com/RhysHuo/Fused_Multicores.git
cd Fused_Multicores
cp kernelMatrixmult_sw.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Multicores
v++ -t hw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k mmult_top_sw -o'kernelmult.hw_emu.xo' kernelMatrixmult_sw.cpp kernelMatrixmult.h xcl2.hpp
