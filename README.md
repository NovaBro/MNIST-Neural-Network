# Neural-Network
This project is still underdevelopment, and its purpose to get a greater understanding on how neural networks actually work. 
There are three main files, MNIST_NN, InitialStart, and BackProp. 

UserInputFile is the main driver program and gives a visual interface to create and change the neural network. 
The GUI created also alows the user to draw thier own number for the neural network to classify!
Here is the the picture of the GUI (It may not be the prettiest I know) with default values.
<img width="800" alt="Screen Shot 2023-06-03 at 7 56 32 PM" src="https://github.com/NovaBro/MNIST-Neural-Network/assets/57100555/aa46da5b-df73-4c1c-aeb7-b1a6099746e3">

MNIST_NN utilizes the InitialStart and BackPropV2 file to create the neural network.
When completed, it creates a graph showing how the cost of the neural network progressed through each epoch or iteration.
<img width="801" alt="Screen Shot 2023-06-03 at 7 57 41 PM" src="https://github.com/NovaBro/MNIST-Neural-Network/assets/57100555/52e40425-6cea-4a16-ae24-d30c86a63d32">

InitialStart reads the MNIST data files and stores them in a 2 dimensional array. Each image is stored as (784, 1) instead of (28,28). This is to make 
matrix multiplicationa and feeding information "forward" in the network easier. The file is also responsible for forming the weights
and bias of the network. I am quite happy with the programs ability to read directly from the MNIST dataset, which was a pain to get right, since there
was a lack of documentation on how to extract the data into python on a mac.

BackPropV2 file is what improves the network. Currently, the back propogation does improve the neural network's performance, but is not fully optimized.
There are definitely things that can be done to increase speed and accuracy of the neural network, and I am still studing how to do these things.

Through this project, I self taught python, numpy, gradient descent, directional derivatives, activation functions, and neural network performance.
