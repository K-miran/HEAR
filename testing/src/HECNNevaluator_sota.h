#include <vector>
#include "utils.h"
#include "thread.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;

// Encoding for nGraph-HE2
namespace seal {
    class HEnGraphCNN{
        public:
        HEnGraphCNN(std::shared_ptr<SEALContext> context);
        
            void encode(vector<uint64_t>& destination, double input, double scale, int coeff_mod_count, MemoryPoolHandle pool = MemoryManager::GetPool());
        
      
            void prepare_network1d(vector<vector<vector<vector<vector<uint64_t>>>>> &ker_plain,
                             vector<vector<vector<vector<uint64_t>>>> &act_plain,
                             vector<vector<vector<uint64_t>>> &dense_ker_plain, vector<vector<uint64_t>> &dense_bias_plain,
                             vector<dten> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, vector<int> NB_CHANNELS);
        
            void prepare_network2d(vector<vector<vector<vector<vector<vector<uint64_t>>>>>> &ker_plain,
                                 vector<vector<vector<vector<uint64_t>>>> &act_plain,
                                 vector<vector<vector<uint64_t>>> &dense_ker_plain, vector<vector<uint64_t>> &dense_bias_plain,
                                 vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias, vector<int> NB_CHANNELS);
        
        
        private:
            std::shared_ptr<SEALContext> context_{ nullptr };
            MemoryPoolHandle pool_ = MemoryManager::GetPool(mm_prof_opt::FORCE_NEW, true);
        
    };
}

// Evaluation for nGraph-HE2
class HEnGraphEval{
public:
    Evaluator& evaluator;
    RelinKeys& relin_keys;
    
    HEnGraphEval(Evaluator& evaluator, RelinKeys& relin_keys):  evaluator(evaluator), relin_keys(relin_keys) {}
    
    void Eval_Conv1d(vector<vector<Ciphertext>> &res,
                   vector<vector<Ciphertext>> ct, vector<vector<vector<vector<uint64_t>>>> plain);
    
    void Eval_Puctured_Conv1d(vector<vector<Ciphertext>> &res,
                              vector<vector<Ciphertext>> ct, vector<vector<vector<vector<uint64_t>>>> plain);
    
    void Eval_Conv2d(vector<vector<vector<Ciphertext>>> &res,
                   vector<vector<vector<Ciphertext>>> ct, vector<vector<vector<vector<vector<uint64_t>>>>> plain);
    
    void Eval_Puctured_Conv2d(vector<vector<vector<Ciphertext>>> &res,
                   vector<vector<vector<Ciphertext>>> ct, vector<vector<vector<vector<vector<uint64_t>>>>> plain);
    
    void Eval_BN_ActPoly1d(vector<vector<Ciphertext>> &ct, vector<vector<vector<uint64_t>>> plain);
    void Eval_BN_ActPoly2d(vector<vector<vector<Ciphertext>>> &ct, vector<vector<vector<uint64_t>>> plain);
    
    void Eval_Average1d(vector<vector<Ciphertext>> &ct);
    void Eval_Global_Average1d(vector<vector<Ciphertext>> &ct);
    void Eval_Average2d(vector<vector<vector<Ciphertext>>> &ct);
    void Eval_Global_Average2d(vector<vector<vector<Ciphertext>>> &ct);
    
    void Eval_Dense1d(vector<vector<Ciphertext>> &res, vector<vector<Ciphertext>> ct,
                    vector<vector<vector<uint64_t>>> dense_ker_plain, vector<vector<uint64_t>> dense_bias_plain);
    void Eval_Dense2d(vector<vector<vector<Ciphertext>>> &res, vector<vector<vector<Ciphertext>>> ct,
                    vector<vector<vector<uint64_t>>> dense_ker_plain, vector<vector<uint64_t>> dense_bias_plain);
};
