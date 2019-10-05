#include "pch.h"
#include "NeuralNetwork.h"
#include "matrix.h"
#include <math.h>
double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double dsigmoid(double y)
{
	return y * (1 - y);
}

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) :
	m_InputNodes(input_nodes),
	m_HiddenNodes(hidden_nodes),
	m_OutputNodes(output_nodes)
{
	//Weights input to hidden
	m_Weights_ih = Matrix(m_HiddenNodes, m_InputNodes);
	//Weights hidden to output
	m_Weights_ho = Matrix(m_OutputNodes, m_HiddenNodes);

	//Start with random values
	m_Weights_ho.Randomize();
	m_Weights_ih.Randomize();

	//Our hidden biases
	m_Bias_h = Matrix(m_HiddenNodes, 1);
	//Our output biases
	m_Bias_o = Matrix(m_OutputNodes, 1);

	//Start with random values
	m_Bias_h.Randomize();
	m_Bias_o.Randomize();

	//Learning rate
	m_LearningRate = 0.1f;
}

NeuralNetwork::~NeuralNetwork()
{
	
}

std::vector<double> NeuralNetwork::Predict(std::vector<double> input)
{
	//Transform the vector inputs to matrix
	Matrix inputs = Matrix::fromVector(input);
	//Calaculate the hidden layer by Matrix multiplication with the weights
	Matrix hidden = m_Weights_ih * inputs;
	//Add the biases
	hidden += m_Bias_h;

	//On every value we actually need the sigmoid value
	hidden.map(sigmoid);

	//Repeat the same but with hidden to output
	Matrix output = m_Weights_ho * hidden;
	output += m_Bias_o;

	output.map(sigmoid);

	//Return and transform to vector
	return Matrix::toVector(output);
}

void NeuralNetwork::Train(std::vector<double> input, std::vector<double> target)
{
	//Transform the vector inputs to matrix
	Matrix inputs = Matrix::fromVector(input);
	//Calaculate the hidden layer by Matrix multiplication with the weights
	Matrix hidden = m_Weights_ih * inputs;
	//Add the biases
	hidden += m_Bias_h;

	//On every value we actually need the sigmoid value
	hidden.map(sigmoid);

	//Repeat the same but with hidden to output
	Matrix output = m_Weights_ho * hidden;
	output += m_Bias_o;

	output.map(sigmoid);

	//Receive the actual target vector and transofrm to Matrix
	Matrix targets = Matrix::fromVector(target);

	//Calculate the "Cost" or the error between target en output
	Matrix outputError = targets - output;

	//Copy the output to gradient for calculations
	Matrix gradient = output;
	//dsigmoid every value
	gradient.map(dsigmoid);
	//Multiply with the error value and then with the learningrate(So we dont overshoot it we set an learning rate the speed we want to learn at)
	gradient *= outputError;
	gradient *= m_LearningRate;
	
	//To get the hidden deltas we need to Transpose our current hidden Matrix to a new Transposed matrix
	Matrix hiddenT = hidden.transpose();
	//We now can caltulate our dealtas by multipling our gradient with hiddenT
	Matrix weight_ho_deltas = gradient * hiddenT;

	//We apply our change to the weights and bias of hidden->Output
	m_Weights_ho += weight_ho_deltas;

	m_Bias_o += gradient;

	//Now we want to do the same thing but with out input->hidden
	//First transpose our weights
	Matrix weight_hoT = m_Weights_ho.transpose();
	//Then calulate the hidden error with the transpose and our calculated outputError
	Matrix hiddenError = weight_hoT * outputError;
	//Now we have the hidden error do the same thing as above with output but with the hidden Matrix and hiddenError
	Matrix hiddenGradient = hidden;
	hiddenGradient.map(dsigmoid);
	hiddenGradient *= hiddenError;
	hiddenGradient *= m_LearningRate;
	//Again we do the exact same as above only with input and hidden instaid of output and hidden
	Matrix inputsT = inputs.transpose();
	Matrix weight_ih_deltas = hiddenGradient * inputsT;

	m_Weights_ih += weight_ih_deltas;
	m_Bias_h += hiddenGradient;

	//Our weights and biases have learned and we will now guess closer to our target
}

