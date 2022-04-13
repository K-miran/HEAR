#include <vector>
#include "utils.h"
#include "thread.h"
#include "seal/seal.h"
#include "HECNNencryptor_conv1d.h"

using namespace std;
using namespace seal;

class HECNNeval1d{
public:
    Evaluator& evaluator;
    RelinKeys& relin_keys;
    GaloisKeys& gal_keys;
    HECNNenc1d& hecnnenc;
    
    HECNNeval1d(Evaluator& evaluator, RelinKeys& relin_keys, GaloisKeys& gal_keys, HECNNenc1d& hecnnenc):
                evaluator(evaluator), relin_keys(relin_keys), gal_keys(gal_keys), hecnnenc(hecnnenc) {}

    // Genetate the rotated ciphertexts 
    void generate_rotations_conv1(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct);
    void generate_rotations_conv1(vector<Ciphertext> &res, Ciphertext ct, string mode);
    void generate_rotations_conv_fully(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block);
    void generate_rotations_conv_fully_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block);
    void generate_rotations_conv_babygiant(vector<vector<Ciphertext>> &res, vector<Ciphertext> ct, int conv_block, string mode);
    void generate_rotations_conv_babygiant_light(vector<vector<vector<Ciphertext>>> &res, vector<Ciphertext> ct, int conv_block, string mode);
    
    // Evaluation of BN-Act
    void Eval_BN_ActPoly_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly);
    void Eval_BN_ActPoly_Fast_inplace(vector<Ciphertext> &ct, vector<vector<Plaintext>> poly);
   
    // Evaluation of the average pooling
    void Eval_Avg_inplace(vector<Ciphertext> &ct, int conv_block);
    
    // Evaluation of the fully-connected layer
    void Eval_Dense(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int NB_CHANNELS);
    void Eval_Dense_Fast(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int NB_CHANNELS);
    void Eval_Dense_Fast_Light(Ciphertext &res, vector<Ciphertext> ct, vector<vector<Plaintext>> ker_poly, Plaintext bias_poly, int NB_CHANNELS);
    
    // Print out the encrypted result by decryption
    void Debug_encrypt(Ciphertext res, const int len = 2);
    void Debug_Conv(vector<Ciphertext> res, int index, string param_dir, int NB_CHANNELS);
    void Debug_Act1(vector<Ciphertext> res);
    void Debug_Block(vector<Ciphertext> res, int index, int NB_CHANNELS);

    // Interlace the input ciphertexts
    void interlace_ctxts(vector<Ciphertext> &res, vector<Ciphertext> &ct, Plaintext zero_one_poly, int conv_block);
};

