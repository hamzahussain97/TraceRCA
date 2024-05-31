# -*- coding: utf-8 -*-

"""
Created on Mon Mar  4 13:57:41 2024

@author: Hamza
"""
import torch
#from ModelTrainer import prepare_data, train, predict, RRMSE, MBE, MAPE, ModelTrainer
from ModelTrainer import ModelTrainer, MAPE, quantile_loss, multi_quantile_loss
from BaselineModel import GNN, EmbGNN, EmbNodeGNNGRU, EmbEdgeGNNGRU
import warnings


warnings.filterwarnings('ignore')


trainer_text = """
###############################################################################
########### NodeGNNGRU Model to predict graph. Loss fn MSE         ############
########### Target logged and scaled                               ############
###############################################################################
"""
print(trainer_text)

# Initialize Model Trainer

data_dir = './A/microservice/test/'
batch_size = 5
predict_graph = True
one_hot_enc = False
normalize_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
scale_features = ['latency']
validate_on_trace = False

#quantiles = [0.0013, 0.0228, 0.05, 0.1587, 0.5000, 0.8413, 0.95, 0.9772, 0.9987]
quantiles = [0.0014, 0.0028, 0.0401, 0.1057, 0.1587, 0.2266, 0.4013, 0.4602, 0.5000, 0.5398, 0.5987, 0.7733, 0.8413, 0.8944, 0.9600, 0.9772, 0.9987]
model_trainer = ModelTrainer(data_dir, batch_size, quantiles, predict_graph, one_hot_enc=one_hot_enc,\
                             normalize_features=normalize_features,\
                             normalize_by_node_features=normalize_by_node_features,\
                             scale_features=scale_features, validate_on_trace=validate_on_trace)

#measures = model_trainer.measures['norm_by_trace']
#total_traces = measures.index.get_level_values('trace_integer').max() + 1
total_traces = 5
# Initialize the model
input_dim = 159
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 30
output_dim = len(quantiles)  # Assuming binary classification

model = EmbGNN(input_dim, hidden_dim, vocab_size, total_traces, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
epochs = 20
loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
#loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, multi_quantile_loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)
'''
trainer_text = """
###############################################################################
########### EdgeGNNGRU Model to predict graph. Loss fn MSE         ############
########### Target logged and scaled                               ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = 6 + len(normalize_by_node_features)
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 10
output_dim = 1  # Assuming binary classification

model = EmbGNN(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)


trainer_text = """
###############################################################################
########### NodeGNNGRU Model to predict graph. Loss fn MAPE        ############
########### Target normalized by node                              ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features) - 1
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbNodeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, MAPE, criterion, optimizer)

model_trainer.predict(graph_idx=0)

trainer_text = """
###############################################################################
########### EdgeGNNGRU Model to predict graph. Loss fn MSE         ############
########### Target normalized by node                              ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features) - 1
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbEdgeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, loss, criterion, optimizer)

model_trainer.predict(graph_idx=0)

trainer_text = """
###############################################################################
########### EdgeGNNGRU Model to predict graph. Loss fn MAPE        ############
########### Target normalized by node                              ############
###############################################################################
"""
print(trainer_text)

# Initialize the model
input_dim = len(normalize_features) + len(normalize_by_node_features) - 1
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 5
output_dim = 1  # Assuming binary classification

model = EmbEdgeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
            predict_graph=model_trainer.predict_graph)
model_trainer.set_model(model)

# Define Loss functions and optimizer
loss = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model_trainer.train(epochs, MAPE, criterion, optimizer)

model_trainer.predict(graph_idx=0)
'''