#include "param.h"
#include <cmath>

/*--------------------*/
// CONV1d
/*--------------------*/

long Param_conv1d::memoryscale = (1 << 20);

// 3D-filter: (DIM_FILTER * DIM_FILTER) * (DEPTH_FILTER)
double Param_conv1d::epsilon = 0.000;
int Param_conv1d::DIM_FILTERS = 3;
int Param_conv1d::npacked = 16;     // the number of results in a single ciphertext
int Param_conv1d::shift = 512;      // dim * dim = 32 * 16, r
int Param_conv1d::logshift = 9;     // log2(shift)

vector<int> Param_conv1d::POOL_SIZE = {2, 2, 120};

// Required rotation amounts
int Param_conv1d::steps_size = 2;
int Param_conv1d::steps_halfsize = 1;       // Param::steps_size/2
int Param_conv1d::steps_giant_size = 15;    // Param::steps_giant.size()

vector<vector<int>> Param_conv1d::steps_conv = {
    {-1, 1},
    {-1 * 2, 1 * 2},
    {-1 * 4, 1 * 4},
};

vector<int> Param_conv1d::steps_giant = {
    (1 << 9), (2 << 9), (3 << 9), (4 << 9), (5 << 9),
    (6 << 9), (7 << 9), (8 << 9), (9 << 9), (10 << 9),
    (11 << 9), (12 << 9), (13 << 9), (14 << 9), (15 << 9)
}; // rotations of giant-step

vector<int> Param_conv1d::steps_pool = {4 * 1, 4 * 2, 4 * 4, 4 * 8, 4 * 16, 4 * 32, 4 * 64};
vector<int> Param_conv1d::steps_interlacing = {-3};


/*--------------------*/
// CONV2d
/*--------------------*/
long Param_conv2d::memoryscale = (1 << 20);

// 3D-filter: (DIM_FILTER * DIM_FILTER) * (DEPTH_FILTER)
double Param_conv2d::epsilon = 0.000;
int Param_conv2d::DIM_FILTERS = 3;
int Param_conv2d::DIM2_FILTERS = Param_conv2d::DIM_FILTERS * Param_conv2d::DIM_FILTERS;
int Param_conv2d::npacked = 16;     // the number of results in a single ciphertext
int Param_conv2d::shift = 512;      // dim * dim = 32 * 16, r
int Param_conv2d::logshift = 9;     // log2(shift)

vector<int> Param_conv2d::POOL_SIZE = {4, 4, 24};

// Required rotation amounts
int Param_conv2d::steps_size = 8;
int Param_conv2d::steps_halfsize = 4;     // Param::steps_size/2
int Param_conv2d::steps_giant_size = 15;  // Param::steps_giant.size()

vector<vector<int>> Param_conv2d::steps_conv = {
    {-17, -16, -15, -1, 1, 15, 16, 17},
    {-17 * 2, -16 * 2, -15 * 2, -1 * 2, 1 * 2, 15 * 2, 16 * 2, 17 * 2},
    {-17 * 4, -16 * 4, -15 * 4, -1 * 4, 1 * 4, 15 * 4, 16 * 4, 17 * 4},
};

vector<int> Param_conv2d::steps_giant = {
    (1 << 9), (2 << 9), (3 << 9), (4 << 9), (5 << 9),
    (6 << 9), (7 << 9), (8 << 9), (9 << 9), (10 << 9),
    (11 << 9), (12 << 9), (13 << 9), (14 << 9), (15 << 9)
}; // rotations of giant-step

vector<int> Param_conv2d::steps_pool = {4 * 2, 16 * 4 * 2, 16 * 4 * 4};
vector<int> Param_conv2d::steps_interlacing = {-3, -18, -19, -33, -35, -48, -49, -50, -51};


/*--------------------*/
// HE-params for HEAR
// logQ = 7*logqc + 3 * logq
/*--------------------*/

size_t Param_HEAR::poly_modulus_degree = (1 << 14);

int Param_HEAR::logq = 31;
int Param_HEAR::logqc = 31;

int Param_HEAR::logq0 = Param_HEAR::logq + 2;
int Param_HEAR::logp0 = Param_HEAR::logq0;      // bit-length of special prime needed for KS

double Param_HEAR::qscale = pow(2.0, Param_HEAR::logq);
double Param_HEAR::qcscale = pow(2.0, Param_HEAR::logqc);

vector<int> Param_HEAR::ker_poly_lvl = {10, 7, 4, 1};   // naive

void get_modulus_chain(vector<int>& bit_sizes_vec, int logq0, int logq, int logqc, int logp0)
{
    vector<int> temp = {
        logq0,
        logqc,                  // FC
        logqc, logq, logqc,     // B3
        logqc, logq, logqc,     // B2
        logqc, logq, logqc,     // B1
        logp0
    };
    
    bit_sizes_vec = temp;
}

/*--------------------*/
// HE-params for Fast-HEAR
// logQ = 9*logqc + 3 * logq
/*--------------------*/

size_t Param_FHEAR::poly_modulus_degree = (1 << 14);

int Param_FHEAR::logq = 31;
int Param_FHEAR::logqc = 31;
int Param_FHEAR::logqc_small = 28;  // should be larger than 25 (for correctness)

int Param_FHEAR::logq0 = Param_FHEAR::logq + 2;
int Param_FHEAR::logp0 = Param_FHEAR::logq0 ;

double Param_FHEAR::qscale = pow(2.0, Param_FHEAR::logq);
double Param_FHEAR::qcscale = pow(2.0, Param_FHEAR::logqc);
double Param_FHEAR::qcscale_small = pow(2.0, Param_FHEAR::logqc_small);

vector<int> Param_FHEAR::ker_poly_lvl = {12, 8, 4, 1};

void get_modulus_chain(vector<int>& bit_sizes_vec, int logq0, int logq, int logqc, int logqc_small, int logp0)
{
    vector<int> temp = {
        logq0,
        logqc,                  // FC
        logqc, logq, logqc,     // B3
        logqc_small,
        logqc, logq, logqc,     // B2 (act2 + conv2)
        logqc_small,            // pre-processing
        logqc, logq, logqc,     // B1 (act1 + conv1)
        logp0
    };
    
    bit_sizes_vec = temp;
}

void print_modulus_chain(vector<int> bit_sizes_vec)
{
    std::cout << "| --->  modulus_chain = {" ;
    for(int i = 0; i < bit_sizes_vec.size() - 1; ++i){
        cout << bit_sizes_vec[i] << ",";
    }
    cout << bit_sizes_vec[bit_sizes_vec.size() - 1] << "}" << endl;
}
