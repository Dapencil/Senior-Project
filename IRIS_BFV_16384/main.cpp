#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "Util.h"
#include "seal/seal.h"
#include "NTL/ZZ.h"
#include "senior.pb.h"

using namespace std;
using namespace seal;
using namespace NTL;

void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
        // std::cout << coeff_modulus[i].value() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

vector<vector<double>> readBinaryInputFile(const string &filename, int numRows, int numCols)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return {};
    }

    // Read the binary data into a flat vector of doubles
    string serialized_data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    InputVector vector_proto;

    if (!vector_proto.ParseFromString(serialized_data))
    {
        cerr << "Failed to parse serialized data." << std::endl;
        return {};
    }
    vector<vector<double>> input_data;

    for (int i = 0; i < numRows; ++i)
    {
        vector<double> row;
        for (int j = 0; j < numCols; ++j)
        {
            row.push_back(vector_proto.values(((j * numRows) + i)));
        }
        input_data.push_back(row);
    }
    return input_data;
}

void writePredictionFile(const string &filename, const vector<int> &pred)
{
    PredictionBFV prediction;

    for (const auto &value : pred)
    {
        prediction.add_values(value);
    }

    // Serialize the Prediction message to a binary file
    fstream output(filename, ios::out | ios::binary);
    if (!prediction.SerializeToOstream(&output))
    {
        cerr << "Failed to write prediction to binary file." << endl;
    }
}

uint64_t transformDoubleToUInt(double number, size_t scale, const Modulus &plain_modulus)
{
    int roundNum = floor(number * scale);
    if (roundNum < 0)
    {
        return (uint64_t)(plain_modulus.value() + roundNum);
    }
    return roundNum;
}

ZZ transformUintToDouble(const ZZ number, const ZZ &plain_modulus)
{
    if (number * 2 > plain_modulus)
    {
        return plain_modulus - number;
    }
    return number;
}

vector<int> argmax_vertical(const vector<vector<ZZ>> &matrix, size_t batch_size)
{
    vector<int> maxIndices;

    int numRows = matrix.size();
    int numCols = batch_size;
    cout << "Argmax" << endl;
    cout << numRows << endl;
    cout << numCols << endl;

    for (int col = 0; col < numCols; ++col)
    {
        ZZ maxVal = matrix[0][col];
        int maxIndex = 0;

        for (int row = 1; row < numRows; ++row)
        {
            if (matrix[row][col] > maxVal)
            {
                maxVal = matrix[row][col];
                maxIndex = row;
            }
        }
        maxIndices.push_back(maxIndex);
    }

    return maxIndices;
}

class BFVDense
{
public:
    // Constructor
    BFVDense(vector<vector<double>> weights,
             vector<double> biases,
             size_t batch_size,
             bool is_apply_activation = true)
        : weights_(weights),
          biases_(biases),
          input_size_(weights[0].size()),
          output_size_(weights.size()),
          batch_size_(batch_size),
          is_apply_activation_(is_apply_activation)
    {
    }

    size_t get_input_size() const { return input_size_; }
    size_t get_output_size() const { return output_size_; }

    vector<Ciphertext> forward(const vector<Ciphertext> &input,
                               const BatchEncoder &encoder_,
                               const Evaluator &evaluator_,
                               const RelinKeys &relin_keys,
                               const Modulus &plain_modulus,
                               size_t layer_index)
    {

        vector<Ciphertext> result;
        Ciphertext output_vector[output_size_];
        Plaintext encoded_weight, encoded_bias;
        size_t rowCnt = output_size_;
        size_t colCnt = input_size_;
        size_t bias_scale = pow(32, pow(2, layer_index + 1));

        // matrix multiplication
        for (size_t i = 0; i < rowCnt; i++)
        {
            for (size_t j = 0; j < colCnt; j++)
            {
                Ciphertext mul_result, add_result;
                encoder_.encode(vector<uint64_t>(batch_size_, transformDoubleToUInt(weights_[i][j], 32, plain_modulus)), encoded_weight);
                evaluator_.multiply_plain(input[j], encoded_weight, mul_result);
                evaluator_.relinearize_inplace(mul_result, relin_keys);
                if (j == 0)
                {
                    output_vector[i] = Ciphertext(mul_result);
                }
                else
                {
                    evaluator_.add(output_vector[i], mul_result, add_result);
                    output_vector[i] = add_result;
                }
            }
            // add bias
            Ciphertext add_bias_result;

            encoder_.encode(vector<uint64_t>(batch_size_, transformDoubleToUInt(biases_[i], bias_scale, plain_modulus)), encoded_bias);
            evaluator_.add_plain(output_vector[i], encoded_bias, add_bias_result);
            output_vector[i] = add_bias_result;

            // activation function
            if (is_apply_activation_)
            {
                evaluator_.square_inplace(output_vector[i]);
                evaluator_.relinearize_inplace(output_vector[i], relin_keys);
            }

            result.push_back(output_vector[i]);
        }
        return result;
    }

private:
    size_t input_size_;
    size_t output_size_;
    size_t batch_size_;

    vector<vector<double>> weights_;
    vector<double> biases_;

    bool is_apply_activation_;
};

class EncryptedModel
{
public:
    EncryptedModel(const PublicKey &public_key,
                   const RelinKeys &relin_keys,
                   size_t poly_modulus_degree,
                   const Modulus &plain_modulus)
        : public_key_(public_key),
          relin_keys_(relin_keys),
          context_(create_context(poly_modulus_degree, plain_modulus)),
          evaluator_(context_),
          encoder_(context_),
          plain_modulus_(plain_modulus)
    {
    }

    void addLayer(const BFVDense &layer)
    {
        if (!layers_.empty() && layers_.back().get_output_size() != layer.get_input_size())
            throw invalid_argument("Input size does not match the expected size.");
        layers_.push_back(layer);
    }

    vector<Ciphertext> predict(const vector<Ciphertext> &input)
    {
        vector<Ciphertext> current_input = input;
        // for (auto &layer : layers_)
        for (size_t index = 0; index < layers_.size(); index++)
        {
            cout << "Start Forward layer.." << endl;
            auto start = chrono::high_resolution_clock::now();
            current_input = layers_[index].forward(current_input, encoder_, evaluator_, relin_keys_, plain_modulus_, index);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end - start);
            cout << "Layer Execution time: " << duration.count() << " seconds" << endl;
        }
        return current_input;
    }

private:
    SEALContext context_;
    PublicKey public_key_;
    RelinKeys relin_keys_;
    Evaluator evaluator_;
    BatchEncoder encoder_;
    Modulus plain_modulus_;

    vector<BFVDense> layers_;

    SEALContext create_context(size_t poly_modulus_degree, Modulus plain_modulus)
    {
        EncryptionParameters parms(scheme_type::bfv);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(plain_modulus);

        return SEALContext(parms);
    }
};

int main()
{
    // ==== Create Computation Environment
    size_t poly_modulus_degree = 16384;
    vector<Modulus> t = PlainModulus::Batching(poly_modulus_degree, {25, 25, 25, 25});
    cout << "t_0: " << t[0].value() << endl;
    cout << "t_1: " << t[1].value() << endl;
    cout << "t_2: " << t[2].value() << endl;
    cout << "t_3: " << t[3].value() << endl;

    // Env 1
    EncryptionParameters parms_1(scheme_type::bfv);
    parms_1.set_poly_modulus_degree(poly_modulus_degree);
    parms_1.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms_1.set_plain_modulus(t[0]);

    SEALContext context_1(parms_1);
    KeyGenerator keygen_1(context_1);
    SecretKey secret_key_1 = keygen_1.secret_key();
    PublicKey public_key_1;
    RelinKeys relin_keys_1;
    keygen_1.create_public_key(public_key_1);
    keygen_1.create_relin_keys(relin_keys_1);
    Encryptor encryptor_1(context_1, public_key_1);
    Evaluator evaluator_1(context_1);
    Decryptor decryptor_1(context_1, secret_key_1);
    BatchEncoder batch_encoder_1(context_1);
    print_parameters(context_1);

    // Env 2
    EncryptionParameters parms_2(scheme_type::bfv);
    parms_2.set_poly_modulus_degree(poly_modulus_degree);
    parms_2.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms_2.set_plain_modulus(t[1]);

    SEALContext context_2(parms_2);
    KeyGenerator keygen_2(context_2);
    SecretKey secret_key_2 = keygen_2.secret_key();
    PublicKey public_key_2;
    keygen_2.create_public_key(public_key_2);
    RelinKeys relin_keys_2;
    keygen_2.create_relin_keys(relin_keys_2);
    Encryptor encryptor_2(context_2, public_key_2);
    Evaluator evaluator_2(context_2);
    Decryptor decryptor_2(context_2, secret_key_2);
    BatchEncoder batch_encoder_2(context_2);
    print_parameters(context_2);

    // Env 3
    EncryptionParameters parms_3(scheme_type::bfv);
    parms_3.set_poly_modulus_degree(poly_modulus_degree);
    parms_3.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms_3.set_plain_modulus(t[2]);

    SEALContext context_3(parms_3);
    KeyGenerator keygen_3(context_3);
    SecretKey secret_key_3 = keygen_3.secret_key();
    PublicKey public_key_3;
    keygen_3.create_public_key(public_key_3);
    RelinKeys relin_keys_3;
    keygen_3.create_relin_keys(relin_keys_3);
    Encryptor encryptor_3(context_3, public_key_3);
    Evaluator evaluator_3(context_3);
    Decryptor decryptor_3(context_3, secret_key_3);
    BatchEncoder batch_encoder_3(context_3);
    print_parameters(context_3);

    // Env 4
    EncryptionParameters parms_4(scheme_type::bfv);
    parms_4.set_poly_modulus_degree(poly_modulus_degree);
    parms_4.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms_4.set_plain_modulus(t[3]);

    SEALContext context_4(parms_4);
    KeyGenerator keygen_4(context_4);
    SecretKey secret_key_4 = keygen_4.secret_key();
    PublicKey public_key_4;
    keygen_4.create_public_key(public_key_4);
    RelinKeys relin_keys_4;
    keygen_4.create_relin_keys(relin_keys_4);
    Encryptor encryptor_4(context_4, public_key_4);
    Evaluator evaluator_4(context_4);
    Decryptor decryptor_4(context_4, secret_key_4);
    BatchEncoder batch_encoder_4(context_4);
    print_parameters(context_4);

    // === INPUT PREPARATION ===
    size_t batch_size = 30;
    size_t feature_dim = 4;
    size_t scale = 32;
    vector<vector<double>> inputs = readBinaryInputFile("../test_iris.bin", feature_dim, batch_size);

    vector<vector<vector<uint64_t>>> splits;

    for (size_t k = 0; k < t.size(); k++)
    {
        vector<vector<uint64_t>> tInputs;
        for (size_t i = 0; i < feature_dim; i++)
        {
            vector<uint64_t> row;
            for (size_t j = 0; j < batch_size; j++)
            {
                row.push_back(transformDoubleToUInt(inputs[i][j], scale, t[k]));
            }
            tInputs.push_back(row);
        }
        splits.push_back(tInputs);
    }

    vector<Ciphertext> encrypted_inputs_1, encrypted_inputs_2, encrypted_inputs_3, encrypted_inputs_4;
    Plaintext encoded_row;
    Ciphertext encrypted_row;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        batch_encoder_1.encode(splits[0][i], encoded_row);
        encryptor_1.encrypt(encoded_row, encrypted_row);
        encrypted_inputs_1.push_back(encrypted_row);

        batch_encoder_2.encode(splits[1][i], encoded_row);
        encryptor_2.encrypt(encoded_row, encrypted_row);
        encrypted_inputs_2.push_back(encrypted_row);

        batch_encoder_3.encode(splits[2][i], encoded_row);
        encryptor_3.encrypt(encoded_row, encrypted_row);
        encrypted_inputs_3.push_back(encrypted_row);

        batch_encoder_4.encode(splits[3][i], encoded_row);
        encryptor_4.encrypt(encoded_row, encrypted_row);
        encrypted_inputs_4.push_back(encrypted_row);
    }
    cout << "Complete prepare input" << endl;

    cout << "Noise Budget" << endl;
    cout << decryptor_1.invariant_noise_budget(encrypted_inputs_1[0]) << endl;
    cout << decryptor_2.invariant_noise_budget(encrypted_inputs_2[0]) << endl;
    cout << decryptor_3.invariant_noise_budget(encrypted_inputs_3[0]) << endl;
    cout << decryptor_4.invariant_noise_budget(encrypted_inputs_4[0]) << endl;

    // === MODEL PREPARATION ===
    extern vector<vector<double>> weight_square_0;
    extern vector<double> bias_square_0;

    extern vector<vector<double>> weight_square_2;
    extern vector<double> bias_square_2;

    extern vector<vector<double>> weight_square_4;
    extern vector<double> bias_square_4;

    BFVDense layer_1 = BFVDense(weight_square_0, bias_square_0, batch_size);
    BFVDense layer_2 = BFVDense(weight_square_2, bias_square_2, batch_size);
    BFVDense layer_3 = BFVDense(weight_square_4, bias_square_4, batch_size, false);

    EncryptedModel model_1 = EncryptedModel(public_key_1, relin_keys_1, poly_modulus_degree, t[0]);
    EncryptedModel model_2 = EncryptedModel(public_key_2, relin_keys_2, poly_modulus_degree, t[1]);
    EncryptedModel model_3 = EncryptedModel(public_key_3, relin_keys_3, poly_modulus_degree, t[2]);
    EncryptedModel model_4 = EncryptedModel(public_key_4, relin_keys_4, poly_modulus_degree, t[3]);

    model_1.addLayer(layer_1);
    model_1.addLayer(layer_2);
    model_1.addLayer(layer_3);

    model_2.addLayer(layer_1);
    model_2.addLayer(layer_2);
    model_2.addLayer(layer_3);

    model_3.addLayer(layer_1);
    model_3.addLayer(layer_2);
    model_3.addLayer(layer_3);

    model_4.addLayer(layer_1);
    model_4.addLayer(layer_2);
    model_4.addLayer(layer_3);

    vector<Ciphertext> ct_result_1, ct_result_2, ct_result_3, ct_result_4;
    auto start = chrono::high_resolution_clock::now();
    cout << "Predicting model_1" << endl;
    ct_result_1 = model_1.predict(encrypted_inputs_1);

    cout << "Predicting model_2" << endl;
    ct_result_2 = model_2.predict(encrypted_inputs_2);

    cout << "Predicting model_3" << endl;
    ct_result_3 = model_3.predict(encrypted_inputs_3);

    cout << "Predicting model_4" << endl;
    ct_result_4 = model_4.predict(encrypted_inputs_4);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Model Execution time: " << duration.count() << " seconds" << endl;

    cout << "Noise Budget" << endl;
    cout << decryptor_1.invariant_noise_budget(ct_result_1[0]) << endl;
    cout << decryptor_2.invariant_noise_budget(ct_result_2[0]) << endl;
    cout << decryptor_3.invariant_noise_budget(ct_result_3[0]) << endl;
    cout << decryptor_4.invariant_noise_budget(ct_result_4[0]) << endl;

    size_t num_labels = 3;
    vector<vector<uint64_t>> result_1, result_2, result_3, result_4;
    for (size_t i = 0; i < num_labels; i++)
    {
        Plaintext plain;
        vector<uint64_t> a;
        decryptor_1.decrypt(ct_result_1[i], plain);
        batch_encoder_1.decode(plain, a);
        result_1.push_back(a);

        decryptor_2.decrypt(ct_result_2[i], plain);
        batch_encoder_2.decode(plain, a);
        result_2.push_back(a);

        decryptor_3.decrypt(ct_result_3[i], plain);
        batch_encoder_3.decode(plain, a);
        result_3.push_back(a);

        decryptor_4.decrypt(ct_result_4[i], plain);
        batch_encoder_4.decode(plain, a);
        result_4.push_back(a);
    }

    vector<vector<ZZ>> result;
    for (size_t i = 0; i < num_labels; i++)
    {
        vector<ZZ> b;
        for (size_t j = 0; j < batch_size; j++)
        {
            ZZ a1, a2, a3, a4;
            a1 = ZZ(result_1[i][j]);
            a2 = ZZ(result_2[i][j]);
            a3 = ZZ(result_3[i][j]);
            a4 = ZZ(result_4[i][j]);

            ZZ t1, t2, t3, t4;
            t1 = ZZ(t[0].value());
            t2 = ZZ(t[1].value());
            t3 = ZZ(t[2].value());
            t4 = ZZ(t[3].value());

            CRT(a1, t1, a2, t2);
            CRT(a1, t1, a3, t3);
            CRT(a1, t1, a4, t4);
            b.push_back(transformUintToDouble(a1, t1));
        }
        result.push_back(b);
    }

    cout << "Analysis Result (Instance 1)" << endl;
    cout << result[0][1] << endl;
    cout << result[1][1] << endl;
    cout << result[2][1] << endl;

    vector<int> pred = argmax_vertical(result, batch_size);
    printVector(pred, batch_size);
    writePredictionFile("../pred.bin", pred);

    return 0;
}