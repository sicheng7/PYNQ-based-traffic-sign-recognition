import torch
import torch.nn as nn


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(
            mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class SignWeight(Function):
    @staticmethod
    def forward(ctx, input):
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)

    def forward(self, input):
        if self.training:
            input = SignSTE.apply(input)
            # print(input)
            self.weight_bin_tensor = SignWeight.apply(self.weight)
        else:
            input = input.clone()
            input.data = input.sign()
            self.weight_bin_tensor = self.weight.new_tensor(self.weight.sign())

        input = F.pad(input, (self.padding[0], self.padding[0],
                              self.padding[1], self.padding[1]), mode='constant', value=-1)
        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride,
                       0, self.dilation, self.groups)

        return out

if  __name__ == '__main__':
    tensor1 = torch.tensor([1., 2., 3.], requires_grad=True)
    clone_x = tensor1.clone()
    f = torch.nn.Linear(3, 1)
    y = f(clone_x)
    y.backward()
    print("clone_x.grad", clone_x.grad)
    print("tensor1.grad",tensor1.grad)


