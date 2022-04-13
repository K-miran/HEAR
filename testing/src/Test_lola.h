#include <vector>
#include "utils.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;


// Implementation of LoLA
class TestLoLA
{
    public:
        // default constructor
        TestLoLA() {}
        
        void hecnn1d(dmat &output, dmat input, int id_st, int id_end,
           vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch);
    
        void hecnn2d(dmat &output, dmat input, int id_st, int id_end,
               vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch);
};

