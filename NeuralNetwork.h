#pragma once
#include <vector>
#include "matrix.h"
class NeuralNetwork
{
public:
	NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes);
	NeuralNetwork(const NeuralNetwork& other) = delete;
	NeuralNetwork& operator= (const NeuralNetwork& other) = delete;
	NeuralNetwork(NeuralNetwork&& other) = delete;
	NeuralNetwork& operator= (NeuralNetwork&& other) = delete;
	~NeuralNetwork();

	std::vector<double> Predict(std::vector<double> input);
	void Train(std::vector<double> input, std::vector<double> target);
private:
	int m_InputNodes;
	int m_HiddenNodes;
	int m_OutputNodes;
	Matrix m_Weights_ih;
	Matrix m_Weights_ho;
	Matrix m_Bias_h;
	Matrix m_Bias_o;
	float m_LearningRate;
};

