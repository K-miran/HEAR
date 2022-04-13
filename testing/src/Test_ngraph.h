#include <vector>
#include "utils.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;

// Implementation of nGraph2
class TestnGraph
{
    public:
        TestnGraph() {}
        
        void hecnn1d(dmat &output, dmat input, int id_st, int id_end,
           vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch);
    
        void hecnn2d(dmat &output, dmat input, int id_st, int id_end,
               vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch);
};

// Implementation of unencrypted computation
class TestPlain
{
    public:
        TestPlain() {}
        
        void Eval_Conv2d(vector<vector<vector<double>>> &res,
                         vector<vector<vector<double>>> input, vector<dten> ker, int batch_size);
        void Eval_BN_ActPoly2d(vector<vector<vector<double>>> &res, dmat poly);
        void Eval_Average2d(vector<vector<vector<double>>> &res, vector<vector<vector<double>>> input);
        void Eval_Global_Average2d(vector<double> &res, vector<vector<vector<double>>> input);
        void Eval_Dense2d(vector<double> &output, vector<double> &input, dmat dense_ker, dvec dense_bias);
    
        void hecnn2d(dmat &output, dmat input, int id_st, int id_end,
               vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, int nch);
};
