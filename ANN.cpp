/**
 * @file ANN.cpp
 *
 * @author RogerNi(NI Ronghao)
 * Contact: nironghao@gmail.com
 *
 */

#include "ANN.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <chrono>

ANN::ANN(std::vector<int> ns_of_l) : testX(nullptr), testY(nullptr), auto_save(false), cfg(ns_of_l)
{
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();//seed
	std::default_random_engine dre(seed);//engine
	std::uniform_real_distribution<float> di(-1, 1);//distribution

	for (int i = 0; i < ns_of_l.size() - 1; ++i) {
		//weights.push_back(Matrix(ns_of_l.at(i) + 1, ns_of_l.at(i + 1)));
		std::vector<float> v1(ns_of_l.at(i)*ns_of_l.at(i + 1));
		std::generate(v1.begin(), v1.end(), [&] { return di(dre); });
		weights.push_back(v1);
		std::vector<float> v2(ns_of_l.at(i + 1));
		std::generate(v2.begin(), v2.end(), [&] { return di(dre); });
		biases.push_back(v2);
	}
}

ANN::ANN(std::string inputPath) : testX(nullptr), testY(nullptr), auto_save(false)
{
	std::string line;
	std::ifstream infile(inputPath);
	if (infile.is_open())
	{
		std::vector<std::vector<float>> temp_mat;
		int lastLayer = 0;
		// read cfg
		std::getline(infile, line);
		std::stringstream line_stream(line);
		std::string token;
		std::vector<float> temp_row;
		while (line_stream >> token)
		{
			cfg.push_back(std::stof(token));
		}

		// read weights

		for (int i = 0; i < cfg.size()-1;++i)
		{
			std::getline(infile, line);
			std::vector<float> w;
			std::stringstream line_stream(line);
			std::string token;
			std::vector<float> temp_row;
			while (line_stream >> token)
			{
				w.push_back(std::stof(token));
			}
			weights.push_back(w);
		}

		// read bias
		for (int i = 0; i < cfg.size() - 1; ++i)
		{
			std::getline(infile, line);
			std::vector<float> b;
			std::stringstream line_stream(line);
			std::string token;
			std::vector<float> temp_row;
			while (line_stream >> token)
			{
				b.push_back(std::stof(token));
			}
			biases.push_back(b);
		}


	}
}

std::vector<float> ANN::predict(const std::vector<float>& input)
{

	std::vector<std::vector<float>> as;
	const std::vector<float> * lastIn = &input;
	for (int c = 0; c < cfg.size() - 1; ++c)
	{
		as.push_back(sigmoid(dot(*lastIn, weights[c], 1, cfg[c], cfg[c + 1]) + biases[c]));
		lastIn = &as[as.size() - 1];
	}
	return *(--(as.end()));
}

int ANN::positionPredict(std::vector<float>& input)
{
	std::vector<float> out = predict(input);
	return std::distance(out.begin(), std::max_element(out.begin(), out.end()));
}


void ANN::writeTo(std::string outPath)
{
	std::ofstream outFile(outPath);
	for (auto & c: cfg)
	{
		outFile << c << "\t";
	}
	outFile << "\n";
	for (auto & w : weights)
	{
		for (auto & num : w)
		{
			outFile << num << "\t";
		}
		outFile << "\n";
	}
	for (auto & b: biases)
	{
		for (auto & num: b)
		{
			outFile << num << "\t";
		}
		outFile << "\n";
	}
}

std::vector<float> ANN::dot(const std::vector<float>& m1, const std::vector<float>& m2, const int m1_rows, const int m1_columns, const int m2_columns)
{
	std::vector <float> output(m1_rows*m2_columns);

#pragma omp parallel for
	for (int row = 0; row < m1_rows; ++row) {
		for (int col = 0; col != m2_columns; ++col) {
			output[row * m2_columns + col] = 0.f;
			for (int k = 0; k != m1_columns; ++k) {
				output[row * m2_columns + col] += m1[row * m1_columns + k] * m2[k * m2_columns + col];
			}
		}
	}

	return output;
}

std::vector<float> operator-(const std::vector<float>& m1, const std::vector<float>& m2)
{
	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> difference(VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		difference[i] = m1[i] - m2[i];
	};

	return difference;
}

void ANN::setTestData(std::vector<std::vector<float>>& x, std::vector<std::vector<float>>& y)
{
	testX = &x;
	testY = &y;
}

float ANN::testAccuracy()
{
	float error = 0;
#pragma  omp parallel for
	for (int i = 0; i < testX->size(); ++i) {
		if (this->positionPredict((*testX)[i]) != std::distance((*testY)[i].begin(), std::find((*testY)[i].begin(), (*testY)[i].end(), 1)))
		{
#pragma omp critical
			error++;
		}
	}
	return 1 - (error / testX->size());
}


void ANN::set_auto_save(bool save)
{
	auto_save = save;
}



void ANN::train( std::vector<std::vector<float>>& input,  std::vector<std::vector<float>>& base, float lr, int epoch, int batch_size)//input data, output data,learing rate, epoch and batch size
{
	std::cout << "Learning rate: " << lr << std::endl;
	std::cout << "Epoch Num: " << epoch << std::endl;
	std::cout << "Minibatch_size: " << batch_size << std::endl;
	int inputSize = input.size();

	int batch_num = input.size() / batch_size;
	std::cout << "Batch number: " << batch_num << std::endl;
	if (input.size() % batch_size != 0) {
		std::cout << "Batch Size Invalid!" << std::endl;
		return;
	}


	for (int e = 0; e < epoch; ++e) {
		auto seed = unsigned(std::time(0));
		std::srand(seed);
		std::random_shuffle(input.begin(), input.end());

		std::srand(seed);
		std::random_shuffle(base.begin(), base.end());


		float epoch_loss = 0;
		//clock_t tStart = clock();
		std::chrono::steady_clock sc;
		auto start = sc.now();
		for (int i = 0; i < batch_num; i++)
		{
			//std::cout << "batch " << i << " :" << std::endl;
			//int j = (rand() % (batch_num)) + 0;
			//std::cout << j;
			std::vector<float> b_X;
			std::vector<float> b_y;
			for (int j = 0;j<batch_size;++j)
			{
				b_X.insert(b_X.end(), input[i*batch_size + j].begin(), input[i*batch_size + j].end());
				b_y.insert(b_y.end(), base[i*batch_size + j].begin(), base[i*batch_size + j].end());
			}
			epoch_loss += trainOneBatch(b_X, b_y, batch_size, lr);
		}
		auto end = sc.now();
		auto time_span = static_cast<std::chrono::duration<double>>(end - start);
		printf("Time taken for this epoch: %.2fs\n", time_span.count());
		epoch_loss /= batch_size;
		std::cout << "Train Loss: " << epoch_loss << std::endl;
		if (testX != nullptr)
			std::cout << "Test Accuracy: " << testAccuracy() << std::endl;
		if (auto_save)
		{
			std::string file = "model_auto_saved_at_" + std::to_string(e + 1);
			this->writeTo(file);
			std::cout << "Model saved to " << file << std::endl;
		}
	}
	//std::cout << batch[0][1];


}

std::vector<float> ANN::sigmoid(const std::vector<float>& m1)
{
	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> output(VECTOR_SIZE);


	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		output[i] = 1 / (1 + exp(-m1[i]));
	}

	return output;
}

std::vector<float> ANN::sigmoid_d(const std::vector<float>& m1)
{
	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> output(VECTOR_SIZE);


	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		output[i] = m1[i] * (1 - m1[i]);
	}

	return output;
}



float ANN::trainOneBatch(const std::vector<float>& input, const std::vector<float>& base,int batch_size,  float lr)
{
	// Feed forward
	std::vector<std::vector<float>> as;
	as.push_back(input);
	const std::vector<float> * lastIn = &input;
	for (int c = 0; c < cfg.size()-1; ++c)
	{
		std::vector<float> temp_bias;
		for (int b = 0; b< batch_size;++b)
		{
			temp_bias.insert(temp_bias.end(), biases[c].begin(), biases[c].end());
		}
		as.push_back(sigmoid(dot(*lastIn, weights[c], batch_size, cfg[c], cfg[c + 1])+ temp_bias));
		lastIn = &as[as.size() - 1];
	}


	// Back propagation
	std::vector<float> dyhat = (as[as.size()-1] - base);
	std::vector<std::vector<float>> dWs;
	std::vector<std::vector<float>> dbs;
	std::vector<std::vector<float>> dzs;
	std::vector<float> * lastZ = &dyhat;

	float loss = 0.0;
	for (unsigned k = 0; k < batch_size * cfg[cfg.size()-1]; ++k) {
		loss += dyhat[k] * dyhat[k];
	}
	loss /= batch_size;

	for (int c = cfg.size() -2 ; c >=0;--c)
	{
		dWs.push_back(dot(transpose(&((as[c])[0]), batch_size, cfg[c]), *lastZ, cfg[c], batch_size, cfg[c+1]));
		dbs.push_back(dot(std::vector<float>(batch_size,1), *lastZ, 1, batch_size, cfg[c + 1]));
		if (c==0)
			break;
		dzs.push_back(dot(*lastZ, transpose(&((weights[c])[0]), cfg[c], cfg[c+1]), batch_size, cfg[c+1], cfg[c]) * sigmoid_d(as[c]));
		lastZ = &dzs[dzs.size() - 1];
	}

	// Updating the parameters
	for (int c = 0; c < cfg.size() - 1; ++c)
	{
		weights[cfg.size() - 2 - c] = weights[cfg.size() - 2 - c] - lr * dWs[c];
		biases[cfg.size() - 2 - c] = biases[cfg.size() - 2 - c] - lr * dbs[c];
	}
	return loss;
}

std::vector <float> transpose(float *m, const int C, const int R) {
	std::vector <float> mT(C*R);

	for (unsigned n = 0; n != C * R; n++) {
		unsigned i = n / C;
		unsigned j = n % C;
		mT[n] = m[R*j + i];
	}

	return mT;
}

std::vector <float> operator+(const std::vector <float>& m1, const std::vector <float>& m2) {
	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> sum(VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		sum[i] = m1[i] + m2[i];
	};

	return sum;
}

std::vector <float> operator*(const std::vector <float>& m1, const std::vector <float>& m2) {
	const unsigned long VECTOR_SIZE = m1.size();
	std::vector <float> product(VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		product[i] = m1[i] * m2[i];
	};

	return product;
}

std::vector <float> operator*(const float m1, const std::vector <float>& m2) {
	const unsigned long VECTOR_SIZE = m2.size();
	std::vector <float> product(VECTOR_SIZE);

	for (unsigned i = 0; i != VECTOR_SIZE; ++i) {
		product[i] = m1 * m2[i];
	};

	return product;
}