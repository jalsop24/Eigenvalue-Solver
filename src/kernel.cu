#include <pycuda-complex.hpp>
//#include <math.h>
//#include <complex>
//#include <cmath>

__global__ void gpu_product( float* X, float* Y, float factor, int Nx, float* V) {

	int index = blockIdx.x * blockDim.x + threadIdx.x; // CUDA index

	if (index < Nx) { //check if index is within the wavefunction
		
		//float temp = V[index];
		
		//Y[index] = temp;
		
		Y[index] = - factor * float(2.0) * X[index]; //( V[index] ) * X[index];
		
		if (index + 1 < Nx){ 
			Y[index] += factor * X[index + 1];
		}
		
		
		if (index - 1 >= 0){
			Y[index] += factor * X[index - 1];
		}

	}
}
