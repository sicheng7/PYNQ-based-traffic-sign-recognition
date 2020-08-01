import torch
import numpy as np
model = torch.load("save_Model_IQ")

for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
    if(name == "features.0.weight" or name == "classifier.3.weight"):
        print(parameters)
        np.save(name+".npy", parameters.cpu().detach().numpy())
    else:
        print(parameters.sign())
        np.save(name+".npy", parameters.sign().cpu().detach().numpy())

for name,parameters in model.named_parameters():
    param = np.load(name+".npy")
    file = open(name +".txt", 'w+')
    print(param.shape)
    for N in range(param.shape[0]):
        for ch in range(param.shape[1]):
            for w in range(param.shape[2]):
                for h in range(param.shape[3]):
                    file.write(str(param[N][ch][w][h]));
                    file.write(' ');