from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os

if __name__ == '__main__':

    args = args_parser()

    # Set random seeds
    setup_seed(args.random_seed)


    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.cuda.set_device('cuda:'+args.device)


    # Loading data
    data = Data(args)


    # Data-size-based aggregation weights
    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]


    # Initialize the central node
    # num_id equals to -1 stands for central node
    central_node = Node(-1, data.test_loader[0], data.test_set, args)


    # Initialize the client nodes
    client_nodes = {}
    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
    

    # Start the FL training
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []
    for rounds in range(args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_nodes, args)

        # Client update
        client_nodes, train_loss = Client_update(args, client_nodes, central_node)
        avg_client_acc = Client_validate(args, client_nodes)
        print(args.server_method + args.client_method + ', averaged clients personalization acc is ', avg_client_acc)
        
        # Partial select function
        if args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(client_nodes))]
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)
        
        # Server update
        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights)
        acc = validate(args, central_node, which_dataset = 'local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

        # Final acc recorder
        if rounds >= args.T - 10:
            final_test_acc_recorder.update(acc)

    print(args.server_method + args.client_method + ', final_testacc is ', final_test_acc_recorder.value())