

def Model_config():
    config = {}
    config['batch_size'] = 32
    config['num_workers'] = 8                 # (32)number of data loading workers
    config['epochs'] = 50                    # number of total epochs to run
    config['lr'] = 1e-4                       # initial learning rate
    config['num_classes'] = 86
    
    
    config['num_layers'] = 3                  
    config['num_heads'] = 8

    config['hidden_dim'] = 256
    config['fnn_dim'] = 256

    config['input_dropout_rate'] = 0
    config['encoder_dropout_rate'] = 0
    config['attention_dropout_rate'] = 0
    
    config['flatten_dim'] = 2048              
    config['inter_dim'] = 128                 # Added inter_dim

    return config
