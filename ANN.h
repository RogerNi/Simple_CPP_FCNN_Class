/**
 * @file ANN.h
 *
 * @author RogerNi(NI Ronghao)
 * Contact: nironghao@gmail.com
 *
 */

#pragma once
#ifndef ANN_H
#define ANN_H
#include <vector>
#include <string>

class ANN
{
public:
	ANN(std::vector<int>);
	ANN(std::string);
	std::vector<float> predict(const std::vector<float>&);
	int positionPredict(std::vector<float>&);
	void train( std::vector<std::vector<float>>&,  std::vector<std::vector<float>>&, float, int, int); // need contain dividing function and call trainOneBatch to train
	std::vector <float> sigmoid(const std::vector <float>& m1);
	std::vector <float> sigmoid_d(const std::vector <float>& m1);
	float trainOneBatch(const std::vector<float>& input, const std::vector<float>& base,int, float lr);

	void writeTo(std::string);

	std::vector <float> dot(const std::vector <float>& m1, const std::vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns);


	void setTestData(std::vector<std::vector<float>>& x, std::vector<std::vector<float>>& y);
	float testAccuracy();

	void setLossFunc(std::string);
	void set_auto_save(bool);

private:
	std::vector<int> cfg;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
	std::vector<std::vector<float>>* testX;
	std::vector<std::vector<float>>* testY;
	bool auto_save;
};

std::vector <float> operator-(const std::vector <float>& m1, const std::vector <float>& m2);
std::vector <float> transpose(float *m, const int C, const int R);
std::vector <float> operator+(const std::vector <float>& m1, const std::vector <float>& m2);
std::vector <float> operator*(const std::vector <float>& m1, const std::vector <float>& m2);
std::vector <float> operator*(const float m1, const std::vector <float>& m2);

#endif
