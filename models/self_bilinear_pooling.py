import torch


def self_bilinear_pooling(x):
    x = torch.reshape(x, (x.size()[0], x.size()[1], x.size()[2] * x.size()[3]))

    bilinear = torch.bmm(x, torch.transpose(x, 1, 2)) / (x.size()[2])
    bilinear = torch.reshape(bilinear, (bilinear.shape[0], bilinear.shape[1] ** 2))
    bilinear = torch.nn.functional.normalize(bilinear)

    return bilinear
