# Simple_CPP_FCNN_Class
This a simple C++ fully connected neural network class for Course COMP3046.

## Codes
+ `ANN.h` : header file for ANN class
+ `ANN.cpp` : implementation of ANN class
+ `ANN_Run.cpp` : sample driving program for ANN class
  
## Main Functions Supported
+ Defining any configuration of FCNN
+ Parrellel computation on CPU
+ Save and Load trained model
+ Output Loss and Testing accuracy
+ Output Epoch time spent

## Explanation on `ANN_Run.cpp`

### Set Configuration of the Neural Network
```
vector<int> cfg = { 28 * 28,100,10 };
```
The numbers in the vector indicates the numbers of neurons on each layer. In the sample file, there are (28 * 28 =) 784 neurons on input layer, 100 neurons on first (the only) hidden layers, and 10 neurons on output layers.

Construct an ANN instance:

```
ANN net(cfg);
```

### Set Testing data
```
net.setTestData(test_X, test_Y);
```
`test_X` and `test_Y` are `vector<vector<float>>` that store the testing X data and testing Y data.

### Train
```
net.train(X_train, y_train, 0.01, 100, 64);
```

Parameter|Explanation
---|---
`X_train`| `vector< vector<float> >` containing training X data
`y_train`| `vector< vector<float> >` containing training y data
`0.01`| learning rate
`100`| Epochs to train
`64`| Batch size

### Write Model to File
```
net.writeTo("Final_Model");
```

### Load Model from File
```
ANN net("Final_Model");
```

### Test Accuracy
```
net.testAccuracy()
```
Returns accuracy

## Acknowledgement
This is a course project for COMP3046 in HKBU. Thanks to course instructor and teaching assistant.

## Author
NI Ronghao ([ RogerNi - GitHub ](https://github.com/RogerNi))
