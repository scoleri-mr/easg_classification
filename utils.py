import torch
from autoencoder import EASGvae
from autoencoder import EASGAutoEncoder

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

def load_model(model_name, model_path, separate, output_dim=256, device='cuda', graph_type ='gcn', eval=True):
    import torch
    verb_dim = 2304
    obj_dim = 1024
    hidden_projection_dim = 1024
    projection_dim = 1024
    hidden_dim = 512
    num_rels=14
    num_verbs=198 
    num_objs=391
    dropout_prob = 0.2
    use_focal_loss=True

    if model_name=='vae':
        model = EASGvae(obj_dim, verb_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, 'original', dropout_prob=dropout_prob, graph_type=graph_type, use_focal_loss=use_focal_loss, separate=separate)
        model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=False)
        model = model.to(device)
    elif model_name=='ae':
        model = EASGAutoEncoder(obj_dim, verb_dim, num_rels, num_verbs, num_objs, hidden_projection_dim, projection_dim, hidden_dim, output_dim, dropout_prob=dropout_prob, graph_type=graph_type, use_focal_loss=use_focal_loss, separate=separate)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model = model.to(device)
    else:
        print("wrong model name: choose 'vae' or 'ae'")
        model.eval()
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

def get_pred_triplets_strict(verbs_out, rels_out):
    ''' build only predicted triplets'''
    triplets_pred = []
    
    for i in range(len(verbs_out)):
        # BUILD THE PREDICTED TRIPLETS
        # apply softmax tco the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        rel_binary = (rel_probs > 0.5).float()
        obj_rels_pred = torch.nonzero(rel_binary[:,:13])
        verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(len(obj_rels_pred),1)
        triplets_pred.append(torch.cat((verb_pred,obj_rels_pred), dim=1))
    return triplets_pred

def topk_verb_accuracy(true_labels, preds_tensors, k):
    if len(true_labels) != len(preds_tensors):
        print("error")
        return 0.0
    
    # Extract top-k predictions for all tensors in one go
    topk_preds = [torch.topk(tensor, k).indices.tolist() for tensor in preds_tensors]    
    matches = 0
    
    # Loop over each true label and its corresponding top-k predictions
    for true_label, top_k in zip(true_labels, topk_preds):
        # Check if the true label is in the top-k predictions
        if true_label.squeeze() in top_k[0]:
            matches += 1

    # Calculate the accuracy
    accuracy = matches / len(true_labels)
    return accuracy * 100


def get_pred_triplets(verbs_out, rels_out, device='cuda'):
    ''' build predicted triplets with fallback for empty predictions '''
    triplets_pred = []
    
    for i in range(len(verbs_out)):
        # BUILD THE PREDICTED TRIPLETS
        # Apply sigmoid to the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        rel_binary = (rel_probs > 0.5).float()
        obj_rels_pred = torch.nonzero(rel_binary[:, :13])

        if len(obj_rels_pred) > 0:
            # If there are valid predictions based on the threshold
            verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(len(obj_rels_pred), 1)
            triplets_pred.append(torch.cat((verb_pred, obj_rels_pred), dim=1))
        else:
            # If no valid predictions, fallback to the highest probability triplet
            # Get the index of the highest probable relationship-object pair
            max_rel_idx = torch.argmax(rel_probs[:, :13])
            obj_idx, rel_idx = divmod(max_rel_idx.item(), 13)  # Get object and relationship indices
            obj_rels_pred = torch.tensor([[obj_idx, rel_idx]]).to(device)  # Create the highest probable object pair

            # Find the highest probable verb
            verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(1, 1).to(device)

            # Append the highest probable triplet
            triplets_pred.append(torch.cat((verb_pred, obj_rels_pred), dim=1))
    
    return triplets_pred

def get_pred_triplets_topk(verbs_out, rels_out, topk=5, device='cuda'):
    ''' build predicted top-k triplets '''
    triplets_pred = []
    
    for i in range(len(verbs_out)):
        # BUILD THE PREDICTED TRIPLETS
        # Apply sigmoid to the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        
        # Flatten the relationship-object pairs and get the top-k indices
        rel_probs_flat = rel_probs[:, :13].flatten()  # Consider only the first 13 relationships
        topk_rel_indices = torch.topk(rel_probs_flat, topk).indices  # Get the indices of the top-k probabilities
        
        # Convert flat indices back to object and relationship indices
        obj_rels_pred = torch.stack([topk_rel_indices // 13, topk_rel_indices % 13], dim=1).to(device)
        
        # Find the top-k predicted verbs
        topk_verb_preds = torch.topk(verbs_out[i], topk).indices.unsqueeze(1).to(device)  # Get top-k verbs
        
        # Repeat each verb prediction for corresponding object-relationship pairs
        verb_pred_repeated = topk_verb_preds.unsqueeze(1).repeat(1, obj_rels_pred.shape[0], 1).squeeze(1).to(device)
        
        # Combine verb, object, and relationship into triplets
        triplets_pred.append(torch.cat((verb_pred_repeated, obj_rels_pred), dim=1))
    
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

def to_triplets_diffusion(verbs_gt, rels_gt, verbs_out, rels_out):
    ''' function to build the triplets from models output'''
    triplets_gt = []
    triplets_pred = []
    
    for i in range(len(verbs_gt)):
        # BUILD THE GT TRIPLETS
        rel_gt = rels_gt[i].squeeze()
        obj_rels_gt = torch.nonzero(rel_gt[:,:13])

        verb = verbs_gt[i].repeat(len(obj_rels_gt), 1)
        triplets_gt.append(torch.cat((verb, obj_rels_gt), dim=1))

        # BUILD THE PREDICTED TRIPLETS
        # apply sigmoid to the matrix to get the predicted objects and relationships
        rel_probs = torch.sigmoid(rels_out[i].squeeze())
        rel_binary = (rel_probs > 0.5).float()
        obj_rels_pred = torch.nonzero(rel_binary[:, :13])

        if len(obj_rels_pred) > 0:
            # If non-empty triplet, predict based on threshold
            verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(len(obj_rels_pred), 1)
            triplets_pred.append(torch.cat((verb_pred, obj_rels_pred), dim=1))
        else:
            # If empty, pick the highest probable triplet
            # Find the highest relationship probabilities and corresponding object pair
            max_rel_idx = torch.argmax(rel_probs[:, :13])
            obj_idx, rel_idx = divmod(max_rel_idx.item(), 13)  # Get object and relationship indices
            obj_rels_pred = torch.tensor([[obj_idx, rel_idx]])  # Create a single-object pair

            # Find the highest probable verb
            verb_pred = torch.argmax(verbs_out[i]).unsqueeze(0).unsqueeze(0).repeat(1, 1)
            triplets_pred.append(torch.cat((verb_pred, obj_rels_pred), dim=1))
    
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

def triplets2words(triplets_lists, annts_path='annts_in_new_format/'):
    # Load verbs, objects, and relationships from respective files
    with open(annts_path + 'verbs.txt') as f:
        verbs = [l.strip() for l in f.readlines()]

    with open(annts_path + 'objects.txt') as f:
        objs = [l.strip() for l in f.readlines()]

    with open(annts_path + 'relationships.txt') as f:
        rels = [l.strip() for l in f.readlines()]

    # Prepare the triplets in word format while preserving dimensions
    triplets_words = []
    for l in triplets_lists:
        tr_word = []  # List to hold the word triplets for each list element
        for tr in l:
            if len(tr) == 0:
                tr_word.append([])  # Preserve empty tensor structure
            else: 
                word_triplets = []  # Hold multiple triplets for this element
                for t in tr:  # For each triplet, convert it to words
                    word_triplets.append([verbs[t[0]], objs[t[1]], rels[t[2]]])
                tr_word.append(word_triplets)  # Append all triplets for the current element
        triplets_words.append(tr_word)  # Append the processed list to the main list
    return triplets_words

def tripletsGT2anticipationGT(video_triplets, evaluation_frame:int):
    """ 
        function to turn the gt triplets into the gt used for anticipation.
        original_words: ground truth triplets for the video
        evaluation_frame: frame we are evaluating for which we need the ground truth
    """
    dobj = None
    for triplet in video_triplets[evaluation_frame]:
        verb = triplet[0]
        if triplet[2] == 'dobj':
            dobj = triplet[1]

    if dobj is None:
        print("No direct object found. Adding object with rel 'in' or 'with'.")
        for triplet in video_triplets[evaluation_frame]:
            verb = triplet[0]
            if triplet[2] == 'with' or triplet[2] == 'in':
                dobj = triplet[1]
    return verb, dobj

def topk_verb_predictions(verb_logits, top_k=5):
    """
    Given the logits for verb predictions (PyTorch tensor), return the top_k verb predictions.
    
    Parameters:
    - verb_logits: A torch tensor of size [198] (logits for each verb)
    - top_k: The number of top verb predictions to return (default is 5)
    
    Returns:
    - A torch tensor containing the indices of the top_k verb predictions
    """
    # Apply softmax to get probabilities
    verb_probs = torch.softmax(verb_logits, dim=0)
    
    # Get indices of the top_k predictions
    top_verb_probs, top_verb_indices = torch.topk(verb_probs, top_k)
    
    return [int(verb_index) for verb_index in top_verb_indices]

def topk_objrels_predictions(obj_rel_matrix, top_k=5, relationship_index=1):
    """
    Given the object-relationship matrix (PyTorch tensor) and a specified relationship index,
    return the top_k object predictions based on the logits in the specified column (relationship).
    
    Parameters:
    - obj_rel_matrix: A torch tensor of size [391, 14] (logits for objects and relationships)
    - top_k: The number of top object predictions to return (default is 5)
    - relationship_index: The index of the relationship to consider (default is 1)
    
    Returns:
    - A list of tuples (object_index, relationship_index) representing the top_k object predictions
    """
    # Extract the logits for the specified relationship column
    rel_logits = obj_rel_matrix[:, relationship_index]
    
    # Get indices of the top_k predictions based on the logits in this column
    top_obj_probs, top_object_indices = torch.topk(rel_logits, top_k)
    
    # Return the top_k object-relationship pairs
    top_object_rel_predictions = [int(obj_idx) for obj_idx in top_object_indices]
    
    return top_object_rel_predictions

def get_predictions_anticipation(decoded_videos, evaluation_frame, verbs, objs, topk=5):
    """ 
        Returns the predictions from the anticipation task as a list of list of tuples
        [[(verb_pred_1, obj_pred_1), ..., (verb_pred_topk, obj_pred_topk)], ... ]
        Each internal list represent the predictions for one subvideo
    """
    predictions = []
    for video in decoded_videos:
        v_pred = topk_verb_predictions(video[0][evaluation_frame], top_k=topk)
        o_pred = topk_objrels_predictions(video[1][evaluation_frame], top_k=topk)
        predictions.append([(verbs[verb_idx], objs[obj_idx]) for verb_idx, obj_idx in zip(v_pred, o_pred)])
    return predictions

def accuracy_anticipation(gt_actions, predicted_actions):
    """
    Calculates top-1 and top-5 accuracy for verb, noun, and action predictions.

    :param gt_actions: List of ground truth tuples (verb, noun).
    :param predicted_actions: List of lists of predicted tuples [(verb, noun), ...], up to 5 per GT.
    :return: Dictionary with accuracies for verb, noun, and action.
    """
    verb_top1, noun_top1, action_top1 = 0, 0, 0
    verb_top5, noun_top5, action_top5 = 0, 0, 0
    assert len(gt_actions) == len(predicted_actions)
    for gt, preds in zip(gt_actions, predicted_actions):
        gt_verb, gt_noun = gt

        # Check top-1 accuracy
        pred_verb, pred_noun = preds[0]
        if gt_verb == pred_verb:
            verb_top1 += 1
        if gt_noun == pred_noun:
            noun_top1 += 1
        if gt == preds[0]:
            action_top1 += 1

        # Check top-5 accuracy
        verbs, nouns = zip(*preds[:5])
        if gt_verb in verbs:
            verb_top5 += 1
        if gt_noun in nouns:
            noun_top5 += 1
        if gt in preds[:5]:
            action_top5 += 1

    total = len(gt_actions)
    accuracies = {
        'verb_top1': verb_top1*100 / total,
        'noun_top1': noun_top1*100 / total,
        'action_top1': action_top1*100 / total,
        'verb_top5': verb_top5*100 / total,
        'noun_top5': noun_top5*100 / total,
        'action_top5': action_top5*100 / total
    }

    return accuracies