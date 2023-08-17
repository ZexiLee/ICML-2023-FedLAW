import numpy as np
import torch
import torch.nn.functional as F
from utils import validate, model_parameter_vector
import copy
from nodes import Node

##############################################################################
# General client function 
##############################################################################

def receive_server_model(args, client_nodes, central_node):

    for idx in range(len(client_nodes)):
        if 'fedlaw' in args.server_method:
            client_nodes[idx].model.load_param(copy.deepcopy(central_node.model.get_param(clone = True)))
        else:
            client_nodes[idx].model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))

    return client_nodes

def Client_update(args, client_nodes, central_node):
    '''
    client update functions
    '''
    # clients receive the server model 
    client_nodes = receive_server_model(args, client_nodes, central_node)

    # update the global model
    if args.client_method == 'local_train':
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_localTrain(args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'fedprox':
        global_model_param = copy.deepcopy(list(central_node.model.parameters()))
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_fedprox(global_model_param, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

    elif args.client_method == 'feddyn':
        global_model_vector = copy.deepcopy(model_parameter_vector(args, central_node.model).detach().clone())
        client_losses = []
        for i in range(len(client_nodes)):
            epoch_losses = []
            for epoch in range(args.E):
                loss = client_feddyn(global_model_vector, args, client_nodes[i])
                epoch_losses.append(loss)
            client_losses.append(sum(epoch_losses)/len(epoch_losses))
            train_loss = sum(client_losses)/len(client_losses)

            # update old grad
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad = client_nodes[i].old_grad - args.mu * (v1 - global_model_vector)

    else:
        raise ValueError('Undefined server method...')

    return client_nodes, train_loss

def Client_validate(args, client_nodes):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in range(len(client_nodes)):
        acc = validate(args, client_nodes[idx])
        # print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)

    return avg_client_acc

# Vanilla local training
def client_localTrain(args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        node.optimizer.step()

    return loss/len(train_loader)

# FedProx
def client_fedprox(global_model_param, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss_local.backward()
        loss = loss + loss_local.item()
        # fedprox update
        node.optimizer.step(global_model_param)

    return loss/len(train_loader)

#FedDyn
def client_feddyn(global_model_vector, args, node, loss = 0.0):
    node.model.train()

    loss = 0.0
    train_loader = node.local_data  # iid
    for idx, (data, target) in enumerate(train_loader):
        # zero_grad
        node.optimizer.zero_grad()
        # train model
        data, target = data.cuda(), target.cuda()
        output_local = node.model(data)

        loss_local =  F.cross_entropy(output_local, target)
        loss = loss + loss_local.item()

        # feddyn update
        v1 = model_parameter_vector(args, node.model)
        loss_local += args.mu/2 * torch.norm(v1 - global_model_vector, 2)
        loss_local -= torch.dot(v1, node.old_grad)

        loss_local.backward()
        node.optimizer.step()

    return loss/len(train_loader)