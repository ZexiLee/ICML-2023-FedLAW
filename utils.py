
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.backends import cudnn
from torch.optim import Optimizer
from models_dict import densenet, resnet, cnn

##############################################################################
# Tools
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def model_parameter_vector(args, model):
    if 'fedlaw' in args.server_method:
        vector = model.flat_w
    else:
        param = [p.view(-1) for p in model.parameters()]
        vector = torch.cat(param, dim=0)
    return vector

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100

    if 'fedlaw' in args.server_method:
        if model_type == 'CNN':
            if args.dataset == 'cifar10':
                model = cnn.CNNCifar10_fedlaw()
            else:
                model = cnn.CNNCifar100_fedlaw()
        elif model_type == 'ResNet20':
            model = resnet.ResNet20_fedlaw(num_classes)
        elif model_type == 'ResNet56':
            model = resnet.ResNet56_fedlaw(num_classes)
        elif model_type == 'ResNet110':
            model = resnet.ResNet110_fedlaw(num_classes)
        elif model_type == 'WRN56_2':
            model = resnet.WRN56_2_fedlaw(num_classes)
        elif model_type == 'WRN56_4':
            model = resnet.WRN56_4_fedlaw(num_classes)
        elif model_type == 'WRN56_8':
            model = resnet.WRN56_8_fedlaw(num_classes)
        elif model_type == 'DenseNet121':
            model = densenet.DenseNet121_fedlaw(num_classes)
        elif model_type == 'DenseNet169':
            model = densenet.DenseNet169_fedlaw(num_classes)
        elif model_type == 'DenseNet201':
            model = densenet.DenseNet201_fedlaw(num_classes)
        elif model_type == 'MLP':
            model = cnn.MLP_fedlaw()
        elif model_type == 'LeNet5':
            model = cnn.LeNet5_fedlaw() 
    else:
        if model_type == 'CNN':
            if args.dataset == 'cifar10':
                model = cnn.CNNCifar10()
            else:
                model = cnn.CNNCifar100()
        elif model_type == 'ResNet20':
            model = resnet.ResNet20(num_classes)
        elif model_type == 'ResNet56':
            model = resnet.ResNet56(num_classes)
        elif model_type == 'ResNet110':
            model = resnet.ResNet110(num_classes)
        elif model_type == 'WRN56_2':
            model = resnet.WRN56_2(num_classes)
        elif model_type == 'WRN56_4':
            model = resnet.WRN56_4(num_classes)
        elif model_type == 'WRN56_8':
            model = resnet.WRN56_8(num_classes)
        elif model_type == 'DenseNet121':
            model = densenet.DenseNet121(num_classes)
        elif model_type == 'DenseNet169':
            model = densenet.DenseNet169(num_classes)
        elif model_type == 'DenseNet201':
            model = densenet.DenseNet201(num_classes)
        elif model_type == 'MLP':
            model = cnn.MLP()
        elif model_type == 'LeNet5':
            model = cnn.LeNet5() 

    return model

def init_optimizer(num_id, model, args):
    optimizer = []
    if num_id > -1 and args.client_method == 'fedprox':
        optimizer = PerturbedGradientDescent(model.parameters(), lr=args.lr, mu=args.mu)
    else:
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)
    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_selectlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return select_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    # print('Learning rate={:.4f}'.format(args.lr))

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # g = g.cuda()
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

##############################################################################
# Validation function
##############################################################################

def validate(args, node, which_dataset = 'validate'):
    node.model.cuda().eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc

def testloss(args, node, which_dataset = 'validate'):
    node.model.cuda().eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model(data)
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value

# Functions for FedLAW with param as an input
def validate_with_param(args, node, param, which_dataset = 'validate'):
    node.model.cuda().eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc

def testloss_with_param(args, node, param, which_dataset = 'validate'):
    node.model.cuda().eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = node.model.forward_with_param(data, param)
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value