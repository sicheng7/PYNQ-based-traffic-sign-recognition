
print("#**************************构建模型*******************************")
import torch
from models import ModelTrafic_64, ModelTrafic_32
#---------------类别数
num_Class = 58

model = ModelTrafic_32()
model.cuda()
# model = torch.load("save_Model_IQ")

print("#**************************损失函数*******************************")
import torch.nn as nn
certerion = nn.CrossEntropyLoss()
certerion.cuda()

print("#**************************优化函数*******************************")
import torch
#---优化函数初始化
optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)
#---管理体制
regime = {
    0: {'optimizer': 'RMSprop', 'lr': 1e-5},
    18: {'lr': 1e-5},
    25: {'lr': 1e-5},
    30: {'lr': 5e-6},
    35: {'lr': 1e-6},
}

print("#**************************训练过程******************************#")
from utils import compute_acc_IQ, EarlyStop, adjust_optimizer
import datetime
from torch.autograd import Variable
import numpy as np


print("Start_train   :", datetime.datetime.now())

#训练相关参数
#--------------------------------------------epoch总数
Epoch_Num = 1000
#--------------------------------------------每一个batch有多少个数据
batch_size_train = int(200)
batch_size_valid = int(2)
batch_size_test = int(2)
# #--------------------------------------------间隔多少个Batch输出一次loss
# Batch_Interval = Batch_Num / Batch_Num
#--------------------------------------------验证集准确度变化
valid_acc_list = []
#--------------------------------------------训练集损失值变化
train_loss_list = []
torch.backends.cudnn.enabled = False
#---导入标签
label_train = np.load("data\\train_label32.npy")
label_test = np.load("data\\test_label32.npy")
label_train = label_train
label_test = label_test
print(len(label_test))
print(len(label_train))
# label = np.load(r'data\label_AlldB.npy')
#---导入训练数据，并查看个数
Data = np.load("data\\train_data32.npy")
# ---导入训练数据，并查看个数
validData = np.load("data\\test_data32.npy")
# Data = np.expand_dims(Data, axis=1)
# validData = np.expand_dims(validData, axis=1)
Data = np.swapaxes(Data, 1,3)
Data = np.swapaxes(Data, 2,3)

validData = np.swapaxes(validData, 1,3)
validData = np.swapaxes(validData, 2,3)
for epoch in range(0, Epoch_Num):
    #---根据优化器的管理体制进行调整优化其参数
    optimizer = adjust_optimizer(optimizer, epoch, regime)
    # print(trainData.shape)
    n_examples = Data.shape[0]
    train_idx = np.random.choice(range(n_examples), size=np.int(n_examples), replace=False)
    trainData = Data[train_idx]
    train_label = label_train[train_idx]
    #---每一个训练集包含多少个Batch
    Batch_Num_train = int(n_examples / batch_size_train)
    print(Batch_Num_train)
    model.train()
    for batch_idx in range(int(Batch_Num_train)):
        # print(batch_idx)
        inputs = trainData[batch_idx *batch_size_train : (batch_idx+1) *batch_size_train]
        target = train_label[batch_idx *batch_size_train : (batch_idx+1) *batch_size_train]
        #---计算batch的输出
        inputs_var = Variable(torch.from_numpy(inputs).float().cuda())
        target_var = Variable(torch.from_numpy(target).float().cuda())
        outputs = model(inputs_var)
        outputs = torch.squeeze(outputs, 2)
        outputs = torch.squeeze(outputs, 2)
        #---利用损失函数计算损失值，并记录
        # print(target_var)
        loss = certerion(outputs, target_var.long())
        train_loss_list.append(loss)
        #---反向传播，参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #---输出训练信息
        if batch_idx % 20 == 0:
            print(f'Epoch: {epoch + 1:03d}/{Epoch_Num:03d} | '
                  f'Batch {batch_idx+ 1:03d}/{Batch_Num_train:03d} |'
                  f' Loss: {loss:.4f}')
    #---模型交叉验证集检查
    model.eval()
    #---------------验证集数据总数 与 正确个数 混淆矩阵
    num_examples = 0
    correct_examples = 0.
    Confusion = torch.zeros(num_Class, num_Class)
    n_examples = validData.shape[0]
    # ---每一个训练集包含多少个Batch
    Batch_Num_valid = int(n_examples / batch_size_valid)
    for batch_idx in range(int(Batch_Num_valid)):
        valid_inputs = validData[batch_idx * batch_size_valid: (batch_idx + 1) * batch_size_valid]
        valid_target = label_test[batch_idx * batch_size_valid: (batch_idx + 1) * batch_size_valid]
        num_example, correct_example, Confusion = compute_acc_IQ(model, valid_inputs, valid_target, Confusion, compute_confusion = False)
        num_examples += num_example
        correct_examples += correct_example
    #---计算验证集精准度并保存
    valid_acc = correct_examples / num_examples * 100
    valid_acc_list.append(valid_acc)
    # print("valid_acc",valid_acc)
    print("valid_acc",valid_acc_list[epoch])
    early = EarlyStop(model, valid_acc, patience = 50, saveModel_name= "save_Model_IQ")
    print("Eopch:%d End  ：" % epoch, datetime.datetime.now())
    if early == True:
        break
    else:
        pass

# print("#**************************测试过程******************************#")
# model = torch.load("save_Model_IQ")
# num_examples = 0
# correct_examples = 0.
# Confusion = torch.zeros(num_Class, num_Class)
# #---导入训练数据，并查看个数
# testData = np.load(************)
# n_examples = testData.shape[0]
# # ---每一个训练集包含多少个Batch
# Batch_Num_valid = int(n_examples / batch_size_valid)
#
# for batch_idx in range(int(Batch_Num_valid)):
#     test_inputs = testData[batch_idx * batch_size_valid: (batch_idx + 1) * batch_size_valid]
#     test_target = label[0, batch_idx * batch_size_valid: (batch_idx + 1) * batch_size_valid]
#     num_example, correct_example, Confusion = compute_acc_IQ(model, test_inputs, test_target, Confusion, compute_confusion = True)
#     num_examples += num_example
#     correct_examples += correct_example
# #---测试集精准度并保存
# test_acc = correct_examples / num_examples * 100
# print("test_acc",test_acc)
# print(Confusion)
