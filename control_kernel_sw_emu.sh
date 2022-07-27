git clone https://github.com/RhysHuo/Fused_Multicores.git
cd Fused_Multicores
cp kernelMatrixmult_sw.cpp ..
cp kernelMatrixmult.h ..
cd ..
rm -rf Fused_Multicores
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -c -k mmult_top_sw -o'kernelmult.sw_emu.xo' kernelMatrixmult_sw.cpp kernelMatrixmult.h xcl2.hpp
v++ -t sw_emu --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 --link kernelmult.sw_emu.xo --config mmult_top_sw.cfg -o'kernelmult.sw_emu.xclbin'
