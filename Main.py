# -*- coding: utf-8 -*-

"""
Created on Mon Mar  4 13:57:41 2024

@author: Hamza
"""
import torch
#from ModelTrainer import prepare_data, train, predict, RRMSE, MBE, MAPE, ModelTrainer
from ModelTrainer import ModelTrainer, predict
from BaselineModel import GNN, EmbGNN, EmbNodeGNNGRU, EmbEdgeGNNGRU
import warnings


warnings.filterwarnings('ignore')


trainer_text = """
###############################################################################
########### NodeGNNGRU Model using edges and validate on graph.    ############
########### Target scaled using log and truncated between 0,1      ############
###############################################################################
"""
print(trainer_text)

# Initialize Model Trainer

data_dir = './A/microservice/test/'
batch_size = 128
predict_graph = False
one_hot_enc = False
normalize_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
scale_features = ['latency']
validate_on_trace = True
model_trainer = ModelTrainer(data_dir, batch_size, predict_graph, one_hot_enc=one_hot_enc,\
                             normalize_features=normalize_features,\
                             normalize_by_node_features=normalize_by_node_features,\
                             scale_features=scale_features, validate_on_trace=validate_on_trace)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features)
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbNodeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 2
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)

trainer_text = """
###############################################################################
########### EdgeGNNGRU Model using edges and validate on graph     ############
########### Target scaled using log and truncated between 0,1      ############
###############################################################################
"""
print(trainer_text)

# Initialize Model Trainer
'''
data_dir = './A/microservice/test/'
batch_size = 128
predict_graph = True
one_hot_enc = False
normalize_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
scale_features = ['latency']
validate_on_trace = False
model_trainer = ModelTrainer(data_dir, batch_size, predict_graph, one_hot_enc=one_hot_enc,\
                             normalize_features=normalize_features,\
                             normalize_by_node_features=normalize_by_node_features,\
                             scale_features=scale_features, validate_on_trace=validate_on_trace)
'''
# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features)
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbEdgeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 2
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)

'''
trainer_text = """
###############################################################################
########### Train Baseline Model Nodes encoded as embeddings.      ############
########### Target scaled using log and truncated between 0,1      ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features)
hidden_dim = 128
hidden_dim_two = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbGNN(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 2
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)


trainer_text = """
###############################################################################
########### Train NodeGNNGRU Model Nodes encoded as embeddings.    ############
########### Target scaled using log and truncated between 0,1      ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features)
hidden_dim = 128
hidden_dim_two = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = GNN(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 2
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)


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


trainer_text= """
###############################################################################
########### Train Embedding Model with normalizing target by node #############
########### Extra features normalized by node                     #############
###############################################################################
"""

print(trainer_text)

# Initialize Model Trainer

data_dir = './A/microservice/test/'
batch_size = 128
predict_graph = True
normalize_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate', 'latency']
normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate']
scale_features = []
validate_on_trace = False
model_trainer = ModelTrainer(data_dir, batch_size, predict_graph, normalize_features=normalize_features,\
                             normalize_by_node_features=normalize_by_node_features,\
                             scale_features=scale_features, validate_on_trace=validate_on_trace)

# Initialize the model
input_dim = 4 + len(normalize_by_node_features)
hidden_dim = 128
hidden_dim_two = 128
output_dim = 1  # Assuming binary classification
vocab_size = len(model_trainer.global_map)
node_embedding_size = 3 

model = EmbNodeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size,\
                      output_dim, predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 5
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)


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