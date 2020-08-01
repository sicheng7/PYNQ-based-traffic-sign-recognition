
import torch

from torch.autograd import Variable

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer




def compute_acc(model, data_loader, num_class = 0, compute_confusion = False):
    correct_pred, num_examples = 0, 0
    model.eval()
    confusion_matrix = torch.zeros(num_class, num_class)
    for i, (features, targets) in enumerate(data_loader, 0):
        with torch.no_grad():
            features = Variable(features.cuda())
            targets = Variable(targets.cuda())

            probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        if compute_confusion:
            for p, t in zip(targets, predicted_labels):
                confusion_matrix[p, t] += 1
        else:
            pass
        num_examples += targets.size(0)
        assert predicted_labels.size() == targets.size()
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100 , confusion_matrix

def compute_acc_squence(model, x, y, confusion, compute_confusion = False):
    correct_pred, num_examples = 0, 0
    model.eval()
    with torch.no_grad():
        features = Variable(torch.from_numpy(x).float().cuda())
        targets = Variable(torch.from_numpy(y).float().cuda())

        features = features.transpose(1, 2).contiguous()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        _, target_label = torch.max(targets, 1)
    if compute_confusion:
        for p, t in zip(target_label, predicted_labels):
            confusion[p, t] += 1
    else:
        pass
    num_examples += targets.size(0)
    assert predicted_labels.size() == target_label.size()
    correct_pred += (predicted_labels == target_label).sum()
    return num_examples, correct_pred, confusion

def compute_acc_IQ(model, x, y, confusion, compute_confusion = False):
    correct_pred, num_examples = 0, 0
    model.eval()
    with torch.no_grad():
        inputs = Variable(torch.from_numpy(x).float().cuda())
        target = Variable(torch.from_numpy(y).float().cuda())

        output = model(inputs)
        outputs = torch.squeeze(output, 2)
        outputs = torch.squeeze(output, 2)
        _, predit_labels = torch.max(output, 1)
        target_labels = target.int()
    if compute_confusion:
        for p, t in zip(target_labels, predit_labels):
            # print(p, t)
            confusion[p, t] += 1
    else:
        pass
    num_examples += inputs.size()[0]
    predit_labels = predit_labels.squeeze(dim= 1)
    predit_labels = predit_labels.squeeze(dim= 1)
    assert predit_labels.size() == target_labels.size()
    correct_pred += (predit_labels == target_labels).sum()
    return num_examples, correct_pred, confusion

import os

check_variable_best = 0.0
check_times = 0

def EarlyStop(model, check_variable, patience, saveModel_name):
    global check_variable_best
    global check_times
    if(check_variable <= check_variable_best):
        check_times += 1
        if(check_times >= patience):
            return True
        else:
            return False
    else:
        if(os.path.exists(saveModel_name)):
            os.remove(saveModel_name)
        else:
            pass
        torch.save(model, saveModel_name)
        check_variable_best = check_variable
        check_times = 0
        return False

check_variable_best_name = "beat_model"

def EarlyStop_saveEvery(model, check_variable, patience, save_name, eopch):
    global check_variable_best
    global check_variable_best_name
    global check_times
    torch.save(model, save_name + "%d" % eopch)
    if(check_variable <= check_variable_best):
        check_times += 1
        if(check_times >= patience):
            return True , check_variable_best_name
        else:
            return False , check_variable_best_name
    else:
        check_variable_best_name = save_name + "%d" % eopch
        check_variable_best = check_variable
        check_times = 0
        return False , check_variable_best_name

from torch.nn import init
import torch.nn as nn

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print("init nn.Conv2d")
            init.normal(m.weight,mean=0, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            print("init nn.BatchNorm2d")
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            print("init nn.Linear")
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

if __name__ == "__main__":
    a = "abc" + "%d" % 4
    print(a)