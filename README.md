Preprocess:
  This part is designed to check the dataset for null or missing values, clean the dataset of any wrong values. 
And if the data is numeric, it will be standardized. If the data is categorical, it will be converted to numerical values.

Neural network:
1.  It is a neural network with two hidden layers. The number of nodes of each layer can be modified. 
2.  The learning rate, iteration times, training rate also can be modified. 
3.  Backpropagation algorithm is applied here, and momentum factor it added to reduce the error of large weights, which is proved effective.
4.  The activation function for hidden layer and outputs node is the sigmoid function. 
5.  The error is calculated as the sum of squire of the difference between target and output. That is the reason more data points leads to more errors.
