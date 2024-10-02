#%%
import torch
import numpy as np


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    # print(output.max(1)[1])
    # print(labels.shape)
    preds = output.type_as(labels)    # 修改后的
    # preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

