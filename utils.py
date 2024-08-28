import torch

def set_wandb_config(num_epochs, hidden_projection_dim, projection_dim,
                     hidden_dim, output_dim, batch_size, scheduler_type,
                     lr_start, lr_step_size, lr_gamma, cosine_annealing_param, 
                     edge_criterion, droput_prob):
    config = {
    'epochs' : num_epochs,
    'hidden_projection_dim' : hidden_projection_dim,
    'projection_dim' : projection_dim,
    'hidden_dim' : hidden_dim,
    'output_dim' : output_dim,
    'batch_size': batch_size,
    'dropout_prob': droput_prob,
    'edge_criterion': edge_criterion,
    'scheduler': scheduler_type
    }
    if scheduler_type == 'cosine_annealing':
        config['cosine_annealing_param'] = cosine_annealing_param
        config['lr_start'] = lr_start
    elif scheduler_type == 'step':
        config['lr_start'] = lr_start
        config['lr_gamma'] = lr_gamma
        config['lr_step_size'] = lr_step_size

    return config

def load_model(model_name, model_path, separate, output_dim=256, device='cuda'):
    import torch
    verb_dim = 2304
    obj_dim = 1024
    hidden_projection_dim = 1024
    projection_dim = 512
    hidden_dim = 512
    num_rels=14
    num_verbs=198 
    num_objs=391
    graph_type ='gcn'
    dropout_prob = 0.2
    use_focal_loss=True

    if model_name=='vae':
        from autoencoder import EASGvae
        model = EASGvae(obj_dim, verb_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, 'original', dropout_prob=dropout_prob, graph_type=graph_type, use_focal_loss=use_focal_loss, separate=separate)
        model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=False)
        model = model.to(device)
    elif model_name=='ae':
        from autoencoder import EASGAutoEncoder
        model = EASGAutoEncoder(obj_dim, verb_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, dropout_prob=dropout_prob, graph_type=graph_type, use_focal_loss=use_focal_loss, separate=separate)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model = model.to(device)
    else:
        print("wrong model name: choose 'vae' or 'ae'")
    return model

def get_pred_and_gt(model_name, model, data_loader, device='cuda'):
    verbs_output = []
    verbs_gt = []
    rels_output = []
    rels_gt = []
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            batch, v_gt, rel_gt = data
            batch = batch.to(device)
            verbs_gt.append(v_gt)
            rels_gt.append(rel_gt)
            if model_name=='vae':
                v_out, rel_out, _, _ = model(batch)
            elif model_name=='ae':
                v_out, rel_out = model(batch)
            else:
                print("wrong model name: choose 'vae' or 'ae'")
            verbs_output.append(v_out)
            rels_output.append(rel_out)
    return verbs_output, verbs_gt, rels_output, rels_gt

def verb_accuracy(list1, list2):
    if len(list1) != len(list2):
        return 0.0
    matches = sum(1 for a, b in zip(list1, list2) if a == b)
    accuracy = matches / len(list1)
    return accuracy*100

def handle_verbs_out(verbs_output):
    # verbs_predictions = [el.item() for el in torch.topk(torch.cat(verbs_output, dim=0), 1).indices]
    verbs_predictions = [torch.argmax(el).item() for el in verbs_output]
    return verbs_predictions

def get_pred_triplets(verbs_out, rels_out):
    ''' build only predicted triplets'''
    triplets_pred = []
    
    for i in range(len(verbs_out)):
        # BUILD THE PREDICTED TRIPLETS
        # apply softmax to the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        rel_binary = (rel_probs > 0.5).float()
        obj_rels_pred = torch.nonzero(rel_binary[:,:13])
        verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(len(obj_rels_pred),1)
        triplets_pred.append(torch.cat((verb_pred,obj_rels_pred), dim=1))
    return triplets_pred

def to_triplets(verbs_gt, rels_gt, verbs_out, rels_out):
    ''' function to build the triplets from models output'''
    triplets_gt = []
    triplets_pred = []
    
    for i in range(len(verbs_gt)):
        # BUILD THE GT TRIPLETS
        rel_gt = rels_gt[i].squeeze()
        obj_rels_gt = torch.nonzero(rel_gt[:,:13])

        verb = verbs_gt[i].repeat(len(obj_rels_gt),1)
        triplets_gt.append(torch.cat((verb,obj_rels_gt), dim=1))

        # BUILD THE PREDICTED TRIPLETS
        # apply softmax to the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        rel_binary = (rel_probs > 0.5).float()
        obj_rels_pred = torch.nonzero(rel_binary[:,:13])
        verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(len(obj_rels_pred),1)
        triplets_pred.append(torch.cat((verb_pred,obj_rels_pred), dim=1))
    return triplets_gt, triplets_pred

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves a checkpoint of the model and optimizer states, along with training metadata.

    Args:
    model (torch.nn.Module): The model whose parameters you want to save.
    optimizer (torch.optim.Optimizer): The optimizer with current state.
    epoch (int): Current epoch number.
    path (str): Path to save the checkpoint file.

    Returns:
    None
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')