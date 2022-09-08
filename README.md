# Neural-Network
This project is still underdevelopment, and its purpose to get a greater understanding on how neural networks actually work. 
Because of this, the performance and formatting of the code is not the primary concern. 
However I do plan on cleaning up the program once I finally acheived the desired output. I believe the main short coming of the current program
is poor use of functions, which have no out put or input and instead only accessing global variables. 
There are three main files, MNIST_NN, InitialStart, and BackProp. 

MNIST_NN is the main driver program, while the other two define functions and variables for the main driver program.

InitialStart reads the MNIST data files and stores them in a 2 dimensional array. Each image is stored as (1, 784) instead of (28,28). This is to make 
matrix multiplicationa and feeding information "forward" in the network easier. The file is also responsible for forming the weights
and bias of the network. I am quite happy with the programs ability to read directly from the MNIST dataset, which was a pain to get right, since there
was a lack of documentation on how to extract the data into python on a mac.

BackProp, which is the file that is still underdevelopment, is what will improve the network. The main struggle with back propogation is keeping
track of the partial derivatives used, which is required in gradient decent. 

Through this project, I self taught python, numpy, gradient descent and directional derivatives.
