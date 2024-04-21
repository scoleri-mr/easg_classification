import wandb

def set_wandb_config(num_epochs, hidden_projection_dim, projection_dim,
                     hidden_dim, output_dim, batch_size,
                     lr_start, lr_step_size, lr_gamma):
    config = {
    'epochs' : num_epochs,
    'hidden_projection_dim' : hidden_projection_dim,
    'projection_dim' : projection_dim,
    'hidden_dim' : hidden_dim,
    'output_dim' : output_dim,
    'batch_size': batch_size,
    'lr_start': lr_start,
    'lr_step_size': lr_step_size,
    'lr_gamma': lr_gamma
    }
    return config