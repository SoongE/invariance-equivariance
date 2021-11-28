from models.bp1 import bp1
from models.bp2 import bp2
from models.bp3 import bp3
from models.cbp1 import cbp1
from models.cbp2 import cbp2
from models.resnet_inv_eq import resnet12

model_pool = [
    'resnet12',
]

model_dict = {
    'resnet12': resnet12,
    'bp1': bp1,
    'bp2': bp2,
    'bp3': bp3,
    'cbp1': cbp1,
    'cbp2': cbp2,
}


def create_model(name, n_cls, dataset='miniImageNet', dropout=0.1, n_trans=16, embd_sz=64, avg_pool=True):
    model = model_dict[name](avg_pool=avg_pool, drop_rate=0.1, dropblock_size=2, num_classes=n_cls,
                             no_trans=n_trans, embd_size=embd_sz)

    return model

# def create_model(name, n_cls, dataset='miniImageNet', dropout=0.1, n_trans=16, embd_sz=64, avg_pool=True):
#     """create model by name"""
#     print("***********", name)
#     if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
#         if name.startswith('resnet50'):
#             print('use imagenet-style resnet50')
#             model = model_dict[name](num_classes=n_cls)
#         elif name.startswith('resnet') or name.startswith('seresnet'):
#
#         else:
#             raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
#     elif dataset == 'CIFAR-FS' or dataset == 'FC100' or dataset == "toy":
#         if name.startswith('resnet') or name.startswith('seresnet'):
#             model = model_dict[name](avg_pool=avg_pool, drop_rate=0.1, dropblock_size=2, num_classes=n_cls,
#                                      no_trans=n_trans, embd_size=embd_sz)
#         else:
#             raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
#     else:
#         raise NotImplementedError('dataset not supported: {}'.format(dataset))
#
#     return model
