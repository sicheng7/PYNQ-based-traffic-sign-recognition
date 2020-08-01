# PYNQ-based-traffic-sign-recognition



This project mainly completes the identification of traffic signs on PYNQ. Our work is as follows.
1. First, we use the cv library to process the size of the picture to get the same size data, and store the data in the list. 
2. Second, we use Pytorch to write a binary convolutional neural network for traffic sign recognition, and complete the training to obtain a model for traffic sign recognition. 
3. Then, we use HLS to design a convolutional neural network reasoning framework based on the built convolutional neural network model.
4. Finally, we import the model parameters under the Pytroch framework into HLS through parameter files, and deploy them in PYNQ-Z2 to realize the recognition of traffic signs on the embedded board.

Next we will expand one by one.
1. Use the cv library to unify the picture size.
We mainly use cv2.imread and cv2.resize functions to read pictures and unify their sizes. We unified the image size to (32, 32, 3). For details, please see the file "Sourcecode\Pytorch\data\DataSet_Process.py".

2. 
