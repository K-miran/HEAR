#include <vector>
#include "utils.h"
#include "thread.h"
#include "seal/seal.h"

using namespace std;
using namespace seal;

class HECNNenc{
public:
    Encryptor& encryptor;
    Decryptor& decryptor;
    CKKSEncoder& encoder;
    
    HECNNenc(Encryptor& encryptor, Decryptor& decryptor, CKKSEncoder& encoder):  encryptor(encryptor), decryptor(decryptor), encoder(encoder) {}

    // Encryption of input data
    void encryptdata(vector<Ciphertext> &res, vector<vector<vector<double>>> data, double scale);
    void encryptdata_packed(Ciphertext &res, vector<vector<vector<double>>> data, double scale);
    
    // Decryption of ciphertext
    void decrypt_vector(vector<double> &dmsg, Ciphertext ct);
  
    // Encoding weights for the first convolutional layer
    void encode_conv1(vector<vector<vector<Plaintext>>> &ker_poly, ften kernel, vector<int> NB_CHANNELS, double scale, int level);
    void encode_conv1(vector<vector<vector<Plaintext>>> &ker_poly, ften kernel, string mode, vector<int> NB_CHANNELS, double scale, int level);
    
    // Encoding weights for the second/their convolutional layer
    void encode_conv(vector<vector<vector<Plaintext>>> &ker_poly, ften kernel, int conv_block,
                     string mode, vector<int> NB_CHANNELS, double scale, int level);
    void encode_conv_interlaced(vector<vector<vector<Plaintext>>> &ker_poly, ften kernel,
                                int conv_block, string mode, vector<int> NB_CHANNELS, double scale, int level);
   
    // Encoding of activation
    void encode_act1(vector<vector<Plaintext>> &act_poly, dmat real_poly, int NB_CHANNELS,
                     double qscale, double qcscale, int level);
    void encode_act(vector<vector<Plaintext>> &act_poly, dmat real_poly, int conv_block, int NB_CHANNELS,
                    double qscale, double qcscale, int level);
    
    // Encoding weights for the fully-connected layer
    void encode_dense_kerpoly(vector<vector<Plaintext>> &dense_ker_poly, dmat dense_ker, int NB_CHANNELS, double scale, int level);
    void encode_dense_biaspoly(Plaintext &bias_poly, dvec dense_bias, double scale);
    
    // Prepare the network parameters
    void prepare_network(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                        vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                        vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                        vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly,
                        vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                        string mode1, string mode2, string mode3, vector<int> NB_CHANNELS);
    
    void prepare_network_interlaced(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                        vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                        vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                        vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly, vector<Plaintext> &zero_one_poly,
                        vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                        string mode1, string mode2, string mode3, vector<int> NB_CHANNELS, vector<int> modulus_bits);
    
    void prepare_network_naive(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                        vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                        vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                        vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly,
                        vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                        string mode1, string mode2, string mode3, vector<int> NB_CHANNELS);
    
    void prepare_network_interlaced_naive(vector<vector<vector<Plaintext>>>&ker1, vector<vector<Plaintext>>&act1,
                        vector<vector<vector<Plaintext>>>&ker2, vector<vector<Plaintext>>&act2,
                        vector<vector<vector<Plaintext>>>&ker3, vector<vector<Plaintext>>&act3,
                        vector<vector<Plaintext>> &dense_ker_poly, Plaintext &dense_bias_poly, vector<Plaintext> &zero_one_poly,
                        vector<ften> ker, vector<dmat> real_poly, dmat dense_ker, dvec dense_bias,
                        string mode1, string mode2, string mode3, vector<int> NB_CHANNELS, vector<int> modulus_bits);

};


