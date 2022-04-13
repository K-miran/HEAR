#include <vector>
#include "utils.h"
#include "thread.h"
#include "seal/seal.h"
#include "HECNNencryptor_conv2d.h"

using namespace std;
using namespace seal;

class HECNNeval{
public:
    Evaluator& evaluator;
    RelinKeys& relin_keys;
    GaloisKeys& gal_keys;
    HECNNenc& hecnnenc;
    
    HECNNeval(Evaluator& evaluator, RelinKeys& relin_keys, GaloisKeys& gal_keys, HECNNenc& hecnnenc):
                evaluator(evaluator), relin_keys(relin_keys), gal_keys(gal_keys), hecnnenc(hecnnenc) {}

    // Genetate the rotated ciphertexts 
    void generate_rotations_conv1(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct);
    void generate_rotations_conv1(vector<Ciphertext> &res, Ciphertext ct, string mode);
    void generate_rotations_conv1_wo_hoisting(vector<Ciphertext> &res, Ciphertext ct, string mode);
    void generate_rotations_conv_fully(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block);
    void generate_rotations_conv_fully_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block);
    void generate_rotations_conv_babygiant(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block, string mode);
    void generate_rotations_conv_babygiant_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block, string mode);
    
    void generate_rotations_conv_fully_wo_hoisting(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block);
    void generate_rotations_conv_babygiant_wo_hoisting(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block, string mode);
    
    // Evaluation of BN-Act
    void Eval_BN_ActPoly_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly);
    void Eval_BN_ActPoly_Fast_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly);
    void Eval_BN_ActPoly_Fast_inplace_wo_lazyres(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly);
    
    // Evaluation of the average pooling
    void Eval_Avg_inplace(vector<Ciphertext> &ct, int conv_block);
    
    // Evaluation of the fully-connected layer
    void Eval_Dense(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch);
    void Eval_Dense_Fast(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch);
    void Eval_Dense_Fast_Light(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch);
    void Eval_Dense_Fast_wo_lazyres(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int nch);
    
    // Interlace the input ciphertexts
    void interlace_ctxts(vector<Ciphertext> &res, vector<Ciphertext> &ct, Plaintext zero_one_poly, int conv_block);
    void interlace_ctxts_wo_lazyres(vector<Ciphertext> &res, vector<Ciphertext> &ct, Plaintext zero_one_poly, int conv_block);
    
    // Debug
    void Debug_encrypt(Ciphertext res);
    void Debug_Conv1(vector<Ciphertext> res, string param_dir);
    void Debug_Act1(vector<Ciphertext> res);
    void Debug_B1(vector<Ciphertext> res, int NB_CHANNELS);
    void Debug_Conv2(vector<Ciphertext> res, string param_dir);
    void Debug_B2(vector<Ciphertext> res);
    void Debug_Conv3(vector<Ciphertext> res, string param_dir);
    void Debug_B3(vector<Ciphertext> res);
    void Eval_Conv1(vector<Ciphertext> &res, vector<Ciphertext> xCipher, vector<vector<vector<Plaintext>>> ker_poly, string mode);
    void Eval_Conv2(vector<Ciphertext> &res2, vector<Ciphertext> res1, vector<vector<vector<Plaintext>>> ker_poly, string mode);
};

