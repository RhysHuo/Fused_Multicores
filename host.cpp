#include <iostream>
#include <sstream> // std::stringstream
#include <algorithm>
#include <chrono>
#include <vector>
#include "xcl2.hpp"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include "host.h"

int SN, SM, SP;

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

//gemm
static void init_arrays_gemm(DTYPE *B, DTYPE *C_sw, DTYPE *C)
{
    for (int i = 0; i < SM; i++) {    
        for (int j = 0; j < SP; j++) {
        	B[i * SP + j] =  0x01;
        }
    }
    for (int i = 0; i < SN; i++) {
        for (int j = 0; j < SP; j++) {
			C_sw[i * SP + j] = 0;
			C[i * SP + j] = 0;
		}
	}
}

static void load_arrays_byte_gemm(DTYPE *A, std::ifstream& myFile)
{
	// Make sure the file is open
	if(!myFile.is_open()) throw std::runtime_error("Could not open byte file");

	// Helper vars
	std::string line;
	int val;
	int val_count=0;
	DTYPE array_val;

    for (int i = 0; i < SN; i++) {
    	// Read data, line by line
    	std::getline(myFile, line);

	    // Create a stringstream of the current line
	    std::stringstream ss(line);

        for (int j = 0; j < SM; j++) {

	        //fill one array val
        	array_val = 0;
	        for(int z =0; z< DTYPE_LENGTH/8; z++)
	        {
	        	// Extract each integer
	        	ss >> val;
	        	array_val = (array_val << 8) + val;

	            // If the next token is a comma, ignore it and move on
	            if(ss.peek() == ',') ss.ignore();
	        }
	        A[i * SM + j] = array_val;
	        val_count++;
	    }
    }
    std::cout << "(BYTE) Val count " << val_count << std::endl;
}

void mmult_golden_byte(DTYPE *A, DTYPE *B, DTYPE *C)
{
	for (int row = 0; row < SN; row++) {
		for (int col = 0; col < SP; col++) {
			DTYPE result = 0;
			for (int k = 0; k < SM; k++) {
				for(int z = 0; z < DTYPE_LENGTH; z+=8) {
					DTYPE A_temp1 = A[row*SM+k];
					DTYPE B_temp1 = B[k*SP+col];
					ap_int<8> A_val = A_temp1.range(z+7,z);
					ap_int<8> B_val = B_temp1.range(z+7,z);
					result += A_val * B_val;
				}
			}
			//C[row*SP+col] = result;
			C[row+col*SN] = result;
		}
	}
}

//spmm
void init_arrays_spmm(DTYPE *x, int row, int col)
{
    for (int i = 0; i < row; i++) {
        //for (int j = 0; j < (col>>2); j++) {
	for (int j = 0; j < col; j++) {
            //x[i*(col>>2)+j] = 0x01010101;
		x[i*col+j] = 0x01010101;
        }
    }
}

void golden_spmm_byte(DTYPE *values, int *row_ptr, int *col_indices, DTYPE *x, int no_vectors, DTYPE *y, int row_size, int col_size) {

	int nvc = 0, i = 0, j = 0, rowStart = 0, rowEnd = row_size;

	DTYPE y0 = 0;
	int last_j = 0;
	for (nvc = 0; nvc < no_vectors; nvc++) {
		for (i = rowStart; i < rowEnd; ++i) {
			y0 = 0;
			for (j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
				for(int z = 0; z < DTYPE_LENGTH; z+=8) {
				    	DTYPE values_val1 = values[j];
					ap_int<8> values_val = values_val1.range(z+7,z);
					int x_value = nvc*col_size+col_indices[j];
					int x_up = x_value >> 2;
					int x_down = (x_value & 0x3);
					y0 += values_val * x[x_up].range(x_down*8+7,x_down*8);
				}
			}
			y[nvc*row_size+i] = y0;
		}
	}
}

//both
static int result_check(DTYPE *y, DTYPE *y_golden, int row, int col)
{
	for (int i = 0; i < row * col; i++) {
		if (y_golden[i] != y[i]) {
			std::cout 	<< "Mismatch: data index= " << i << " golden = " << y_golden[i]
						<< ", kernel = " << y[i] << std::endl;
			return 1;
		}
	}
    std::cout 	<< "TEST PASSED !" <<  std::endl;
	return 0;
}

//main
int main(int argc, char** argv) {

    if (argc != 8) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << " cores" << " mode: 0 for gemm / 1 for spmm" << " file" << " N" << " M" << " P" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
	int S_cores = atoi(argv[2]);
	if(S_cores > 7) {
		std::cout 	<< "No enough cores (7 in total), please re-enter core number." <<  std::endl;
		return EXIT_FAILURE;
	}
	int core_count = (S_cores&0x7);
    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
	std::string cu_id;
    std::string krnl_name = "mmult_top";
    std::vector<cl::Kernel> krnls(S_cores);
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
			for (int i = 0; i < core_count; i++) {
                cu_id = std::to_string(i + 1);
                std::string krnl_name_full = krnl_name + ":{" + "mmult_top_" + cu_id + "}";
                printf("Creating a kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), i);
                // Here Kernel object is created by specifying kernel name along with
                // compute unit.
                // For such case, this kernel object can only access the specific
                // Compute unit
                OCL_CHECK(err, krnls[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
            }
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
	
	ap_uint<2> spmm = atoi(argv[3]);
	FILE *fp_input;
   	fp_input = fopen(argv[4], "r");
	
    ap_int<32> bias_count = 0;
	ap_int<8> zero_point_lhs = 0;
	ap_int<8> zero_point_rhs = 0;
	ap_int<8> zero_point_dst = 0;
	ap_int<8> clamp_max = 127;
	ap_int<8> clamp_min = -128;
    int nnz = 512;
	int row_size = 0;
    int col_size = 0;
	
	if(spmm) {
	    
        if (fp_input != NULL) {
            char line_1[1000];
            if(fgets(line_1, sizeof(line_1), fp_input) != NULL){
                sscanf(line_1, "%d %d %d", &row_size, &col_size, &nnz);
            }
        }
        else {
            std::cout << "Error with input file name" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int no_vectors = 512;

    if(spmm){
        SN = row_size;
        SM = col_size;
        SP = no_vectors;
    }
    else{
        SN = atoi(argv[5]);
        SM = atoi(argv[6]);
        SP = atoi(argv[7]);
    }
	
	DTYPE *array_a;
	DTYPE *array_b;
	DTYPE *array_values;
	DTYPE *array_c;
	DTYPE *array_c_golden;
	DTYPE_OUT *quantized_multiplier;
    DTYPE_OUT *shift;
    DTYPE_OUT *bias;
    int *array_colIndices;
    int *array_rowPtr;
	int size = 1;
	
	posix_memalign((void **)&array_a, 4096, size * SN * SM * sizeof(DTYPE));
	posix_memalign((void **)&array_b, 4096, size * SM * SP * sizeof(DTYPE));
	posix_memalign((void **)&array_values, 4096, size * SN * SM * sizeof(DTYPE));
	posix_memalign((void **)&array_c, 4096, size * SN * SP * sizeof(DTYPE));
	posix_memalign((void **)&array_c_golden, 4096, size * SN * SP * sizeof(DTYPE));
	posix_memalign((void **)&quantized_multiplier, 4096, size * SN * sizeof(DTYPE_OUT));
	posix_memalign((void **)&shift, 4096, size * SN * sizeof(DTYPE_OUT));
	posix_memalign((void **)&bias, 4096, size * SN * sizeof(DTYPE_OUT));
	posix_memalign((void **)&array_colIndices, 4096, size * nnz * sizeof(int));
	if(spmm) {
		posix_memalign((void **)&array_rowPtr, 4096, size * (SN + 1) * sizeof(int));
	}
	else {
		posix_memalign((void **)&array_rowPtr, 4096, size * SN * sizeof(int));
	}
	
	/*
	printf("array_a = %x \n", array_a);
	printf("array_b = %x \n", array_b);
	printf("array_values = %x \n", array_values);
	printf("array_c = %x \n", array_c);
	printf("array_c_golden = %x \n", array_c_golden);
	printf("quantized_multiplier = %x \n", quantized_multiplier);
	printf("shift = %x \n", shift);
	printf("bias = %x \n", bias);
	printf("array_colIndices = %x \n", array_colIndices);
	printf("array_rowPtr = %x \n", array_rowPtr);
	printf("array_c - array_values = %x %d \n", array_c - array_values, array_c - array_values);
	printf("sizeof(DTYPE) = %x %d \n", sizeof(DTYPE), sizeof(DTYPE));
	*/
	
	int array_c_adjust = SN;
	int N_block = SN;
	int P_block = SP / core_count;
	int P_tail = SP % core_count;
	int bias_offset = 0;
	
	std::vector<cl::Buffer> buffer_array_a(core_count);
	std::vector<cl::Buffer> buffer_array_b(core_count);
	std::vector<cl::Buffer> buffer_array_values(core_count);
	std::vector<cl::Buffer> buffer_array_c(core_count);
	std::vector<cl::Buffer> buffer_quantized_multiplier(core_count);
	std::vector<cl::Buffer> buffer_shift(core_count);
	std::vector<cl::Buffer> buffer_bias(core_count);
	std::vector<cl::Buffer> buffer_array_colIndices(core_count);
	std::vector<cl::Buffer> buffer_array_rowPtr(core_count);
	
	/*
	DTYPE *array_b_block;
	DTYPE *array_c_block;
	posix_memalign((void **)&array_b_block, 4096, SM * SP * sizeof(DTYPE)/core_count);
	posix_memalign((void **)&array_c_block, 4096, SN * SP * sizeof(DTYPE)/core_count);
	*/
	
	for(int i = 0; i < core_count; i++) {
		//array_b_block = (DTYPE*)(array_b + i*P_block*SM);
		//array_c_block = (DTYPE*)(array_c + i*P_block*SN);
		//OCL_CHECK(err, buffer_array_b[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR  , SM * SP * sizeof(DTYPE)/core_count, array_b_block, &err));
		//OCL_CHECK(err, buffer_array_c[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR  , SN * SP * sizeof(DTYPE)/core_count, array_c_block, &err));
		if((P_tail != 0) && (i == (core_count - 1))) {
			OCL_CHECK(err, buffer_array_b[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR  , SM * (P_block + P_tail) * sizeof(DTYPE), array_b + i*P_block*SM, &err));
			OCL_CHECK(err, buffer_array_c[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR  , SN * (P_block + P_tail) * sizeof(DTYPE), array_c + i*P_block*SN, &err));
		}
		else {
			OCL_CHECK(err, buffer_array_b[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR  , SM * P_block * sizeof(DTYPE), array_b + i*P_block*SM, &err));
			OCL_CHECK(err, buffer_array_c[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR  , SN * P_block * sizeof(DTYPE), array_c + i*P_block*SN, &err));
		}
		OCL_CHECK(err, buffer_array_a[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * SM * sizeof(DTYPE), array_a, &err));
		OCL_CHECK(err, buffer_array_values[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * SM * sizeof(DTYPE), array_values, &err));
		OCL_CHECK(err, buffer_quantized_multiplier[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * sizeof(DTYPE_OUT), quantized_multiplier, &err));
		OCL_CHECK(err, buffer_shift[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * sizeof(DTYPE_OUT), shift, &err));
		OCL_CHECK(err, buffer_bias[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * sizeof(DTYPE_OUT), bias, &err));
		OCL_CHECK(err, buffer_array_colIndices[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , nnz * sizeof(int), array_colIndices, &err));
		if(spmm) {
			OCL_CHECK(err, buffer_array_rowPtr[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , (SN + 1) * sizeof(int), array_rowPtr, &err));
		}
		else {
			OCL_CHECK(err, buffer_array_rowPtr[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR , SN * sizeof(int), array_rowPtr, &err));
		}
	}
	
	if(spmm)
        init_arrays_spmm(array_b, SM, SP);
    else
        init_arrays_gemm(array_b, array_c_golden, array_c);
	
	for(int i = 0; i < SN; i++)
	{
		quantized_multiplier[i] = 1;
		shift[i] = 0;
		bias[i] = 0;
	}
	
	std::cout << "init_arrays completed." << std::endl;

	// load arrays
    if(spmm){
        int r;
        int c;
        DTYPE v;

        if (fp_input != NULL) {
            char line_2[1000];
            int line_number = 0;
                while (fgets(line_2, sizeof(line_2), fp_input) != NULL) {
                if (line_number < nnz) {
                    sscanf(line_2, "%d %d", &c, &v);
                    array_colIndices[line_number] = c;
                    //std::cout << "array_colIndices = " << array_colIndices[line_number] << std::endl;
                    array_values[line_number] = v;
                    //std::cout << "array_values = " << array_values[line_number] << std::endl;
                }
                else {
                    sscanf(line_2, "%d", &r);
                    array_rowPtr[line_number - nnz] = r;
                    //std::cout << "array_rowPtr = " << array_rowPtr[line_number - nnz] << std::endl;
                }
                line_number++;
            }
        }
    }
    else {
		std::ifstream myFile(argv[4]);
		load_arrays_byte_gemm(array_a, myFile);
    }
        
	std::cout << "Load data completed." << std::endl;
	
	for(int i = 0; i < core_count; i++) {
		int narg = 0;
		OCL_CHECK(err, err = krnls[i].setArg(narg++, spmm));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_quantized_multiplier[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_shift[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_bias[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, bias_count));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, zero_point_lhs));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, zero_point_rhs));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, zero_point_dst));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, clamp_max));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, clamp_min));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, N_block));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, SM));
		if(i != (core_count - 1)) {
			OCL_CHECK(err, err = krnls[i].setArg(narg++, P_block));
		}
		else {
			OCL_CHECK(err, err = krnls[i].setArg(narg++, P_block+P_tail));
			//std::cout << "check point ----001 " << std::endl;
		}
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_a[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_b[i]));
		//std::cout << "check point ----002 " << std::endl;
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_c[i]));
		//std::cout << "check point ----003 " << std::endl;
		OCL_CHECK(err, err = krnls[i].setArg(narg++, array_c_adjust));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_rowPtr[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_colIndices[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_array_values[i]));
		OCL_CHECK(err, err = krnls[i].setArg(narg++, nnz));
		
		OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_a[i], buffer_array_b[i], buffer_array_values[i], buffer_quantized_multiplier[i], buffer_shift[i], buffer_bias[i], buffer_array_colIndices[i], buffer_array_rowPtr[i]}, 0));
	}
	OCL_CHECK(err, err = q.finish());
	
	std::cout << "Running : Kernel." << std::endl;

	auto fpga_begin = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < core_count; i++) {
        // Launch the kernel
        OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
    }
	OCL_CHECK(err, err = q.finish());
	
	for (int i = 0; i < core_count; i++) {
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_array_c[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
    }
	OCL_CHECK(err, err = q.finish());
    
	auto fpga_end = std::chrono::high_resolution_clock::now();
	std::cout << "Complete : Kernel execution." << std::endl;
	
	/*
    std::cout << "Start : mmult_golden." << std::endl;
    auto cpu_begin = std::chrono::high_resolution_clock::now();
    
	if(spmm)
        golden_spmm_byte(
            array_values,
            array_rowPtr,
            array_colIndices,
            array_b,
            SP,
            array_c_golden,
            SN,
            SM
        );
    else
        mmult_golden_byte(array_a, array_b, array_c_golden);

	auto cpu_end = std::chrono::high_resolution_clock::now();
	std::cout << "Complete : mmult_golden." << std::endl;
  	
	
    // Compare the results of the Device to the simulation
    std::cout << "Start : result_check." << std::endl;

    if(result_check(array_c, array_c_golden, SN, SP))
        return 1;
	*/
	
	std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
	//std::chrono::duration<double> cpu_duration = cpu_end - cpu_begin;
	//float fpga_throughput = (double) numRuns*3*nbytes / fpga_duration.count() / (1024.0*1024.0);
	//float cpu_throughput  = (double) numRuns*3*nbytes / cpu_duration.count() / (1024.0*1024.0);
	
	std::cout << std::endl;
	std::cout << "----------------------------------------------------------------------------"   << std::endl;
	std::cout << "         Performance  " << std::endl;
	//std::cout << "          Total data: " << total << " MBits" << std::endl;
	std::cout << "          FPGA Time : " << fpga_duration.count() * 1000.0 << " ms" << std::endl;
	//std::cout << "     FPGA Throughput: " << total / fpga_duration.count() << " MBits/s" << std::endl;
	//std::cout << "FPGA PCIe Throughput: " << (2*total) / fpga_duration.count() << " MBits/s" << std::endl;
	//std::cout << "           CPU Time : " << cpu_duration.count() * 1000.0 << " ms" << std::endl;
	//std::cout << "       FPGA Speedup : " << cpu_duration.count() / fpga_duration.count() << " x" << std::endl;
	std::cout << "----------------------------------------------------------------------------"   << std::endl;
	
}
