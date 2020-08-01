# PYNQ-based-traffic-sign-recognition



This project mainly completes the identification of traffic signs on PYNQ. Our work is as follows.
1. First, we use the cv library to process the size of the picture to get the same size data, and store the data in the list. 
2. Second, we use Pytorch to write a binary convolutional neural network for traffic sign recognition, and complete the training to obtain a model for traffic sign recognition. 
3. Then, we use HLS to design a convolutional neural network reasoning framework based on the built convolutional neural network model.
4. Finally, we import the model parameters under the Pytroch framework into HLS through parameter files, and deploy them in PYNQ-Z2 to realize the recognition of traffic signs on the embedded board.

Next we will expand one by one.
1. Use the cv library to unify the picture size.
We mainly use cv2.imread and cv2.resize functions to read pictures and unify their sizes. We unified the image size to (32, 32, 3). For details, please see the file "Sourcecode\Pytorch\data\DataSet_Process.py".

2. Design and train convolutional neural network based on Pytorch.
The convolutional neural network we designed includes a feature extraction part and a classification part.
        
        self.features = nn.Sequential(
            #32-3+1 = 30
            #30 / 2 = 15
            nn.Conv2d(3, 128, kernel_size= 3, stride= 1, padding= 0),
            nn.MaxPool2d(2, stride=2, return_indices=False),
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=2, return_indices=False),
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=1, return_indices=False),
        )

        self.classifier = nn.Sequential(
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            BinaryConv2d(128, 128, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, num_cls, 1, stride= 1, padding= 0),
            nn.Softmax(dim=1)
        )
3. Use HLS to design forward inference process of convolutional neural network.
In the lib.cpp file, we define the parameter types used in the project; 
In the parameter.cpp file, we define the parameters needed for the entire convolutional neural network;
The conv.cpp, pool.cpp, and relu.cpp files respectively define the convolutional layer, pooling layer and activation layer;
In the operate.cpp file, some operations outside the main network framework are defined, such as obtaining results from the output;
In the model.cpp file, call the functions in the above files to build the entire convolutional network model;
In the cnn_top.cpp file, the reasoning model is called at the top layer.

For the model in the Pytorch framework, we export the parameters in the model to the TXT file, and then read the parameters in the c_testbench file and import the HLS model.

4. Board test.
To be carried out.
