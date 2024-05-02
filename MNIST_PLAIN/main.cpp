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

vector<vector<double>> plus_bias(const vector<vector<double>> &input, const vector<double> bias)
{
    vector<vector<double>> result;

    for (size_t row = 0; row < input.size(); row++)
    {
        vector<double> tempRow;
        for (size_t col = 0; col < input[0].size(); col++)
        {
            tempRow.push_back(input[row][col] + bias[row]);
        }
        result.push_back(tempRow);
    }

    return result;
}

vector<int> argmax_vertical(const vector<vector<double>> &matrix, size_t batch_size)
{
    vector<int> maxIndices;

    int numRows = matrix.size();
    int numCols = batch_size;
    cout << "Argmax" << endl;
    cout << numRows << endl;
    cout << numCols << endl;

    for (int col = 0; col < numCols; ++col)
    {
        double maxVal = matrix[0][col];
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

class Dense
{
public:
    // Constructor
    Dense(vector<vector<double>> weights,
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

    vector<vector<double>> forward(const vector<vector<double>> &input)
    {
        size_t rowCnt = output_size_;
        size_t colCnt = input_size_;
        vector<vector<double>> result(rowCnt, vector<double>(batch_size_, 0.0));

        // matrix multiplication
        for (size_t i = 0; i < rowCnt; i++)
        {
            for (size_t k = 0; k < batch_size_; k++)
            {
                for (size_t j = 0; j < colCnt; j++)
                {
                    result[i][k] += weights_[i][j] * input[j][k];
                }
                // add bias
                result[i][k] += biases_[i];
                // activation
                if (is_apply_activation_)
                {
                    result[i][k] *= result[i][k];
                }
            }
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

class Model
{
public:
    Model()
    {
    }

    void addLayer(const Dense &layer)
    {
        if (!layers_.empty() && layers_.back().get_output_size() != layer.get_input_size())
            throw invalid_argument("Input size does not match the expected size.");
        layers_.push_back(layer);
    }

    vector<vector<double>> predict(const vector<vector<double>> &input)
    {
        vector<vector<double>> current_input = input;
        for (auto &layer : layers_)
        {
            cout << "Start Forward layer.." << endl;
            auto start = chrono::high_resolution_clock::now();
            current_input = layer.forward(current_input);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(end - start);
            cout << "Layer Execution time: " << duration.count() << " seconds" << endl;
        }
        return current_input;
    }

private:
    vector<Dense> layers_;
};

int main()
{
    // === INPUT PREPARATION ===
    size_t batch_size = 10000;
    size_t feature_dim = 784;
    vector<vector<double>> inputs = readBinaryInputFile("../test_mnist_nonpre.bin", feature_dim, batch_size);

    // === MODEL PREPARATION ===
    extern vector<vector<double>> weight_square_0;
    extern vector<double> bias_square_0;

    extern vector<vector<double>> weight_square_2;
    extern vector<double> bias_square_2;

    extern vector<vector<double>> weight_square_4;
    extern vector<double> bias_square_4;

    extern vector<vector<double>> weight_square_6;
    extern vector<double> bias_square_6;

    Dense layer_1 = Dense(weight_square_0, bias_square_0, batch_size);
    Dense layer_2 = Dense(weight_square_2, bias_square_2, batch_size);
    Dense layer_3 = Dense(weight_square_4, bias_square_4, batch_size);
    Dense layer_4 = Dense(weight_square_6, bias_square_6, batch_size, false);

    Model model = Model();

    model.addLayer(layer_1);
    model.addLayer(layer_2);
    model.addLayer(layer_3);
    model.addLayer(layer_4);

    auto start = chrono::high_resolution_clock::now();
    cout << "Predicting model_1" << endl;
    vector<vector<double>> result = model.predict(inputs);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Model Execution time: " << duration.count() << " seconds" << endl;

    vector<int> pred = argmax_vertical(result, batch_size);
    printVector(pred, batch_size);
    writePredictionFile("../pred_plain_mnist_nonpre.bin", pred);

    return 0;
}
