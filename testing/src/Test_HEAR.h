#include <vector>
#include "utils.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;


class TestHEAR1d
{
public:
    // default constructor
    TestHEAR1d() {}

    void hecnn(dmat &output, dmat input, int id_st, int id_end,
               vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
               string mode1, string mode2, string mode3, int nch, string method);
};


class TestHEAR2d
{
public:
    // default constructor
    TestHEAR2d() {}

    void hecnn(dmat &output, dmat input, int id_st, int id_end,
               vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
               string mode1, string mode2, string mode3, int nch, string method);
    
    void hecnn_threading(dmat &output, dmat input, int id_st, int id_end,
               vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
               string mode1, string mode2, string mode3, int nch, string method, int NUM_THREADS);
};
