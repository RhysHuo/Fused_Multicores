/*Copyright (c) [2021] [Jose Nunez-Yanez (eejlny@bristol.ac.uk)]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
* This file has been written at University of Bristol
* for the HOPWARE/MINET project
*
* 
* author    : Jose Nunez-Yanez eejlny@bristol.ac.uk
* date      : 1 October 2021
*/

#ifndef KERNELMATRIXMULT_H_
#define KERNELMATRIXMULT_H_

#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <fstream>
#include <hls_stream.h>

#include "ap_int.h"



#define DTYPE_LENGTH 32 //8//32 //8 //32//32 //8//32//8
#define MAX_N    2048 //2048 //64
#define MAX_M    2048 //2048 //384 //1536 //384 //1536 // 384 //48 //768 //96 //384 //96 //384 //96 //384// 96
#define MAX_P    32768 //2048 //512 //64//1//64//1

#define A_HEIGHT   MAX_N
#define A_WIDTH    MAX_M

#define B_HEIGHT   MAX_M
#define B_WIDTH    MAX_P

#define C_HEIGHT   MAX_N
#define C_WIDTH    MAX_P

#define A_HEIGHT_BLOCK  1// 4096 //(512/4)
#define B_WIDTH_BLOCK 128 //16 //32 //64 //64 //128 // 64 //64 //64 //8//8// //16//32//1//32//1//32//1// 1//32//(128/4)
#define B_BLOCK_PARALLEL 1
#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 
#define C_WIDTH_BLOCK   B_WIDTH_BLOCK

#define ENABLE_GEMM
#define ENABLE_SPMM
//#define ENABLE_SCALING
#define ENABLE_TRANSPOSE


typedef ap_int<DTYPE_LENGTH> DTYPE;
typedef ap_int<32> DTYPE_OUT;

const static int II = 1;

const static int ROW_SIZE_MAX        = (1024);
const static int ROW_SIZE_THREAD_MAX = (1024);
const static int COL_SIZE_MAX        = (1024);

void dsp_kernel_sw(
	DTYPE a_value,
	DTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],
	ap_int<32> b_row,
	ap_int<8> zero_point_lhs,
	ap_int<8> zero_point_rhs,
	DTYPE_OUT acc[B_WIDTH_BLOCK]
);

void compute_sw(
	ap_uint<2> mode, 
	ap_int<8> zero_point_lhs,  
	ap_int<8> zero_point_rhs, 
	int N, 
	int M, 
	int P, 
	DTYPE* A, 
	DTYPE* B, 
	hls::stream<DTYPE_OUT> C_fifo[B_WIDTH_BLOCK],
	int B_index, 
	int B_index_loop, 
	int tail,
	int *rowPtr,
	int *columnIndex,
	DTYPE *values
);

void scale_sw(
	ap_int<32> *quantized_multiplier, 
	ap_int<32> *shift, 
	ap_int<32> *bias, 
	ap_int<8> zero_point_dst, 
	ap_int<8> clamp_max,
	ap_int<8> clamp_min,
	int N, 
	int M, 
	int P, 
	hls::stream<DTYPE_OUT> C_fifo[C_WIDTH_BLOCK],
	int B_index, 
	int B_index_loop,
	int tail,
	hls::stream<DTYPE_OUT> write_fifo[C_WIDTH_BLOCK]
);

void writec_sw(
	int N,
	int P, 
	hls::stream<DTYPE_OUT> write_fifo[C_WIDTH_BLOCK], 
	DTYPE* C,
	int array_c_adjust,
	int B_index, 
	int B_index_loop,
	int tail
);

void mmult_wrapper_sw(
	ap_uint<2> mode, 
	ap_int<32> *quantized_multiplier, 
	ap_int<32> *shift, 
	ap_int<32> *bias,  
	ap_int<32> bias_count, 
	ap_int<8> zero_point_lhs,  
	ap_int<8> zero_point_rhs, 
	ap_int<8> zero_point_dst, 
	ap_int<8> clamp_max,
	ap_int<8> clamp_min,
	int N, 
	int M, 
	int P, 
	DTYPE* A, 
	DTYPE* B, 
	DTYPE* C, 
	int array_c_adjust, 
	int B_index, 
	int B_index_loop, 
	int tail,
	int *rowPtr,
	int *columnIndex,
	DTYPE *values
);

void mmult_top_sw(
	ap_uint<2> mode, 
	ap_int<32> *quantized_multiplier, 
	ap_int<32> *shift, 
	ap_int<32> *bias,  
	ap_int<32> bias_count, 
	ap_int<8> zero_point_lhs,  
	ap_int<8> zero_point_rhs, 
	ap_int<8> zero_point_dst, 
	ap_int<8> clamp_max,
	ap_int<8> clamp_min,
	int N, 
	int M, 
	int P, 
	DTYPE* A, 
	DTYPE* B, 
	DTYPE* C,
	int array_c_adjust,
	int *rowPtr,
	int *columnIndex,
	DTYPE *values,
	int nnz
);

extern "C" {

	void kernelmult1_sw(
		int cores,
		ap_uint<2> mode,
		ap_int<32> *quantized_multiplier,
		ap_int<32> *shift,
		ap_int<32> *bias,
		ap_int<32> bias_count,
		ap_int<8> zero_point_lhs,
		ap_int<8> zero_point_rhs,
		ap_int<8> zero_point_dst,
		ap_int<8> clamp_max,
		ap_int<8> clamp_min,
		DTYPE *array_a,
		DTYPE *array_b,
		DTYPE *array_c,
		DTYPE *values,
		int *colIndices,
		int *rowPtr,
		int nnz,
		int N,
		int M,
		int P
	);

}

#endif 
