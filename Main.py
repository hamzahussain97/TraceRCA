# -*- coding: utf-8 -*-

"""
Created on Mon Mar  4 13:57:41 2024

@author: Hamza
"""
import torch
from ModelTrainer import prepare_data, train, predict, RRMSE, MBE
from BaselineModel import GNN, EmbeddingGNN, EmbNodeGNNGRU, EmbEdgeGNNGRU

'''
trainer_text = """
###############################################################################
########### Train Vanilla Model with no changes to data and simple ############
########### Optimizer. Nodes encoded as one hot                    ############
###############################################################################
"""
print(trainer_text)

data, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=True)

# Initialize the model
input_dim = 2 + len(global_map)
hidden_dim = 64
hidden_dim_two = 64
output_dim = 1  # Assuming binary classification
model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)

# Loss and optimizer
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = train(model, MSE, MAE, optimizer, measures, 10, loaders)
predict(model, data[0], measures)


trainer_text= """
###############################################################################
########### Train Embedding Model with normalizing features/target##############
########### Extra node normalized features                       ##############
###############################################################################
"""
print(trainer_text)
normalize_by_node_features = ['cpu_use', 'mem_use_percent']
data, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=False,\
                                                   normalize_features=['latency', 'cpu_use', 'mem_use_percent'], \
                                                   normalize_by_node_features=normalize_by_node_features)
# Initialize the model
input_dim = 2 + len(normalize_by_node_features)
hidden_dim = 64
hidden_dim_two = 64
output_dim = 1  # Assuming binary classification
#model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)
model = EmbeddingGNN(input_dim, hidden_dim, len(global_map), 10, 1)

# Loss and optimizer
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, RRMSE, MAE, optimizer, measures, 10, loaders, recov=True)
predict(model, data[0], measures, recov=True)


trainer_text= """
###############################################################################
########### Train Vanilla Model with normalizing features/target ##############
########### Nodes encoded as one hot vector                      ##############
###############################################################################
"""
print(trainer_text)
data, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=True,\
                                                   normalize_features=['latency', 'cpu_use', 'mem_use_percent'])
# Initialize the model
input_dim = 2 + len(global_map)
hidden_dim = 64
hidden_dim_two = 64
output_dim = 1  # Assuming binary classification
model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)

# Loss and optimizer
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, MSE, MAE, optimizer, measures, 10, loaders, recov=True)
predict(model, data[0], measures, recov=True)

'''
trainer_text= """
###############################################################################
########### Train Embedding Model with normalizing target by node #############
########### Extra features normalized by node                     #############
###############################################################################
"""

print(trainer_text)

#normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
#data, graphs, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=False,\
#                                                   normalize_features=['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate'],
#                                                   normalize_by_node_features=normalize_by_node_features,\
#                                                   scale_features=['latency'])
# Initialize the model

input_dim = 4 + len(normalize_by_node_features)
hidden_dim = 128
hidden_dim_two = 128
output_dim = 1  # Assuming binary classification
#model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)
model = EmbEdgeGNNGRU(input_dim, hidden_dim, len(global_map), 10, 1)


# Loss and optimizer
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = train(model, MSE, MAE, optimizer, measures, 50, loaders, recov_scaling=True)

predict(model, graphs[0], measures, recov_scaling=True)

'''
trainer_text= """
###############################################################################
########### Train Vanilla Model with normalizing features by node #############
########### and the target also normalized by node and one hot enc#############
###############################################################################
"""
print(trainer_text)
data, measures, global_map, loaders = prepare_data(batch_size=128, one_hot_enc=True,\
                                                   normalize_by_node_features=['cpu_use', 'mem_use_percent', 'latency'])
# Initialize the model
input_dim = 2 + len(global_map)
hidden_dim = 64
hidden_dim_two = 64
output_dim = 1  # Assuming binary classification
model = GNN(input_dim, hidden_dim, hidden_dim_two, output_dim)

# Loss and optimizer
MSE = torch.nn.MSELoss(reduction='mean')
MAE = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model = train(model, MSE, MAE, optimizer, measures, 10, loaders, recov_by_node=True)
predict(model, data[0], measures, recov_by_node=True)
'''