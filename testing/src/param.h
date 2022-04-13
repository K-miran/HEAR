#include <iostream>
#include <vector>
using namespace std;

class Param_conv2d {
  public:
    static long memoryscale;
    
    // ML-param
    static double epsilon;
    static int DIM_FILTERS;         // dimension of kernels in the conv layers
    static int DIM2_FILTERS;        // = dim * dim
    static int npacked;
    static int shift;
    static int logshift;
    static vector<int> POOL_SIZE;   // total size of average pooling
    
    // rotation amounts for homomorphic evaluation
    static vector<vector<int>> steps_conv;
    static vector<int> steps_conv1;
    static vector<int> steps_conv2;
    static vector<int> steps_conv3;
    static vector<int> steps_giant;
    static vector<int> steps_pool;
    static vector<int> steps_interlacing;
    static int steps_size;
    static int steps_halfsize;
    static int steps_giant_size;
};


class Param_conv1d {
  public:
    static long memoryscale;
    
    // ML-param
    static double epsilon;
    static int DIM_FILTERS;            // dimension of kernels in the conv layers
    static int npacked;
    static int shift;
    static int logshift;
    static vector<int> NUM_CHANNELS;   // number of output channels
    static vector<int> POOL_SIZE;      // total size of average pooling
    
    // rotation amounts for homomorphic evaluation
    static vector<vector<int>> steps_conv;
    static vector<int> steps_conv1;
    static vector<int> steps_conv2;
    static vector<int> steps_conv3;
    static vector<int> steps_giant;
    static vector<int> steps_pool;
    static vector<int> steps_interlacing;
    static int steps_size;
    static int steps_halfsize;
    static int steps_giant_size;
    
    Param_conv1d() {}
};

// parameters for HEAR
class Param_HEAR {
public:
    static size_t poly_modulus_degree;  // N, degree of ring polynomials
    
    // ciphertext moduli
    static int logq0;
    static int logq;
    static int logqc;
    static int logp0;
    static double qscale;
    static double qcscale;
    
    static vector<int> ker_poly_lvl;   // Encoding level for kernels, ker_poly_lvl[l]: for the (l+1)-th conv layer
};

// parameters for Fast-HEAR
class Param_FHEAR {
public:
    static size_t poly_modulus_degree;  // N, degree of ring polynomials
    
    // ciphertext moduli
    static int logq0;
    static int logq;
    static int logqc;
    static int logqc_small;
    static int logp0;
    static double qscale;
    static double qcscale;
    static double qcscale_small;
    
    static vector<int> ker_poly_lvl;   // Encoding level for kernels, ker_poly_lvl[l]: for the (l+1)-th conv layer
};

void get_modulus_chain(vector<int>& bit_sizes_vec, int logq0, int logq, int logqc, int logp0);
void get_modulus_chain(vector<int>& bit_sizes_vec, int logq0, int logq, int logqc, int logqc_small, int logp0);
void print_modulus_chain(vector<int> bit_sizes_vec);
