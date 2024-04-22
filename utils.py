import wandb

def set_wandb_config(num_epochs, hidden_projection_dim, projection_dim,
                     hidden_dim, output_dim, batch_size, scheduler_type,
                     lr_start, lr_step_size, lr_gamma, cosine_annealing_param):
    config = {
    'epochs' : num_epochs,
    'hidden_projection_dim' : hidden_projection_dim,
    'projection_dim' : projection_dim,
    'hidden_dim' : hidden_dim,
    'output_dim' : output_dim,
    'batch_size': batch_size,
    'scheduler': scheduler_type,
    }
    if scheduler_type == 'cosine_annealing':
        config['cosine_annealing_param'] == cosine_annealing_param
    elif scheduler_type == 'step':
        config['lr_start'] = lr_start
        config['lr_gamma'] = lr_gamma
        config['lr_step_size'] = lr_step_size

    return config