// NeuralNetworkFrameWork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "NeuralNetwork.h"
#include <time.h>
#include <vector>
#include <iostream>
#include <string>
using namespace std;


int main()
{
	srand(time(time_t(NULL)));
	NeuralNetwork* myNetwork = new NeuralNetwork(2, 2, 1);

	std::vector<double> vector = { 1,0 };
	std::vector<double> target = { 0.5 };

	std::vector<double> output = myNetwork->Predict(vector);

	for (size_t i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << std::endl;
	}
	bool stop = true;
	while (stop)
	{
		myNetwork->Train(vector, target);

		output = myNetwork->Predict(vector);

		for (size_t i = 0; i < output.size(); i++)
		{
			std::cout << output[i] << std::endl;
			if (abs(output[i] - target[i]) < 0.0001)
			{
				stop = false;
			}
			
		}
	}
	delete myNetwork;
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
