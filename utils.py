import wandb

def set_wandb_config(learning_rate, num_epochs, hidden_projection_dim, projection_dim,
                     hidden_dim, output_dim, batch_size):
    config = {
    'learning_rate' : learning_rate,
    'epochs' : num_epochs,
    'hidden_projection_dim' : hidden_projection_dim,
    'projection_dim' : projection_dim,
    'hidden_dim' : hidden_dim,
    'output_dim' : output_dim,
    'batch_size': batch_size
    }
    return config