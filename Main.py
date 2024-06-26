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
########### GNN/GRU Trained on MicroSS Dataset                     ############
########### Validation on Edge Latencies                           ############
###############################################################################
"""
print(trainer_text)

# Initialize Model Trainer

data_dir = './TrainTicket/'
batch_size = 5
predict_graph = False
one_hot_enc = False
normalize_features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
normalize_by_node_features = ['cpu_use', 'mem_use_percent', 'mem_use_amount', 'net_send_rate', 'net_receive_rate', 'file_read_rate']
scale_features = ['latency']
validate_on_trace = False

quantiles = [0.0013, 0.0062, 0.0228, 0.0668, 0.1587, 0.2266, 0.3085, 0.3539, 0.4013, 0.4503, 0.5000, 0.5498, 0.5987, 0.6462, 0.6915, 0.7734, 0.8413, 0.9332, 0.9772, 0.9938, 0.9987]
model_trainer = ModelTrainer(data_dir, batch_size, quantiles, predict_graph, one_hot_enc=one_hot_enc,\
                             normalize_features=normalize_features,\
                             normalize_by_node_features=normalize_by_node_features,\
                             scale_features=scale_features, validate_on_trace=validate_on_trace)

#measures = model_trainer.measures['norm_by_trace']
#total_traces = measures.index.get_level_values('trace_integer').max() + 1
total_traces = 5
# Initialize the model
input_dim = model_trainer.graphs[0].x.size()[1] - 1
hidden_dim = 128
vocab_size = len(model_trainer.global_map)
node_embedding_size = 30
output_dim = len(quantiles)  # Assuming binary classification

model = EmbEdgeGNNGRU(input_dim, hidden_dim, vocab_size, node_embedding_size, output_dim,\
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
