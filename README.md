# Neural-Network
This project is still underdevelopment, and its purpose to get a greater understanding on how neural networks actually work. 
There are three main files, MNIST_NN, InitialStart, and BackProp. 

MNIST_NN is the main driver program, while the other two define functions and variables for the main driver program.

InitialStart reads the MNIST data files and stores them in a 2 dimensional array. Each image is stored as (784, 1) instead of (28,28). This is to make 
matrix multiplicationa and feeding information "forward" in the network easier. The file is also responsible for forming the weights
and bias of the network. I am quite happy with the programs ability to read directly from the MNIST dataset, which was a pain to get right, since there
was a lack of documentation on how to extract the data into python on a mac.

BackProp file is what improves the network. Currently, the back propogation does improve the neural network's performance, but is not fully optimized.
There are definitely things that can be done to increase speed and accuracy of the neural network, and I am still studing how to do these things.

Through this project, I self taught python, numpy, gradient descent, directional derivatives, activation functions, and neural network performance.
