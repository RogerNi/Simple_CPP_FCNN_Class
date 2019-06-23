#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <time.h>
#include <algorithm>
#include "ANN.h"
#include <omp.h>

using namespace std;

int main() {

	//omp_set_num_threads(1);

	bool training = false;

	cout << (training ? "Training Mode:" : "Testing Mode:") << endl;

	vector< vector<float> > X_train;
	vector< vector<float> > y_train;

	if (training) {

		ifstream myfile("train.txt");

		if (myfile.is_open())
		{
			cout << "Loading data ...\n";
			string line;
			while (getline(myfile, line))
			{
				vector<float> y_default(10, 0);
				int x, y;
				vector<float> X;
				stringstream ss(line);
				ss >> y;

				y_default[y] = 1;
				y_train.push_back(y_default);

				for (int i = 0; i < 28 * 28; i++) {
					ss >> x;
					X.push_back(x / 255.0);
				}
				X_train.push_back(X);
			}

			myfile.close();
			cout << "Loading data finished.\n";
		}
		else
			cout << "Unable to open file" << '\n';
	}

	//Testing
	ifstream testFile("test.txt");
	vector<vector<float>> test_X;
	vector<vector<float>> test_Y;

	if (testFile.is_open())
	{
		cout << "Loading testing Data ...\n";
		string line;
		while (getline(testFile, line))
		{
			vector<float> y_default(10, 0);
			int x, y;
			vector<float> X;
			stringstream ss(line);
			ss >> y;

			y_default[y] = 1;
			test_Y.push_back(y_default);

			for (int i = 0; i < 28 * 28; i++) {
				ss >> x;
				X.push_back(x / 255.0);
			}
			test_X.push_back(X);
		}

		testFile.close();
		cout << "Loading testing data finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';

	// For training
	if (training) {
		vector<int> cfg = { 28 * 28,100,10 };
		ANN net(cfg);
		net.setTestData(test_X, test_Y);
		//net.set_auto_save(true);
		net.train(X_train, y_train, 0.01, 100, 64);
		net.writeTo("Final_Model");
	}

	// For validation
	if (!training) {
		ANN net("Final_Model");
		net.setTestData(test_X, test_Y);
		cout << "The accuracy of the model is: " << net.testAccuracy() << endl;
	}



	return 0;
}
