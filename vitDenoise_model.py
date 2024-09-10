import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from simple_vit import Transformer
from dataset_video.dataset_video import EASGvideo

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def condition_projection(x_noisy, x_start, num_fixed_frames=5):
    """
    Ensures that the first `num_noise_free` frames of the video are noise-free.
    
    :param x: Input tensor of shape (batch_size, sequence_length+1, feature_dim)
    :param num_noise_free: Number of initial elements to keep noise-free
    :return: Modified tensor with the first `num_noise_free` elements unchanged
    """
    x_noisy[:, :num_fixed_frames] = x_start[:, :num_fixed_frames].clone()
    return x_noisy    # out ->    [batch_size, frames, code_dim]

def positional_encoding(d_model, length, batch_size, device='cuda'):
    """
    :param d_model: dimension of the codes
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.unsqueeze(0)
    return pe.expand(batch_size, -1, -1).to(device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, num_fixed_frames=5):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    x_noisy = condition_projection(x_noisy, x_start, num_fixed_frames)  # Apply condition projection
    return x_noisy

# Loss function for denoising
def p_losses(denoise_model, x_start, t, pe, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1", mode='noise', num_fixed_frames=5, device="cuda"):
    if noise is None:
        noise = torch.randn(x_start.size(0), x_start.size(1), x_start.size(2)).to(device)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, num_fixed_frames)
    out_pred = denoise_model(x_noisy, t, pe)  # if reconstruct=True out_pred will cointain the reconstructed x, otherwise the predicted noise

    if mode=='reconstruct':
        # Reconstuction loss: predict the denoised sample
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, out_pred)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, out_pred)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, out_pred)
        else:
            raise NotImplementedError()
    elif mode=='noise':
        # Noise prediction loss: predict the noise
        if loss_type == 'l1':
            loss = F.l1_loss(noise, out_pred)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, out_pred)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, out_pred)
        else:
            raise NotImplementedError()
    else:
        raise ValueError(f"Unknown mode {mode}")

    return loss

# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Transformer
class SimpleViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, time_dim, dim_head=64):
        super().__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, dim-time_dim)

    def forward(self, x):
        # x is a sequence with shape (batch_size, sequence_length, code_dim + time_dim)
        x = self.transformer(x) 
        x = self.to_latent(x)
        return self.linear_head(x)  # Project back to the original dimension (batch_size, sequence_length, code_dim)
     
# Denoise model
class DenoiseViT(nn.Module):
    def __init__(self, depth, heads, d_model, hidden_dim, time_dim=64):
        super(DenoiseViT, self).__init__()

        # time encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, time_dim),
        )

        # positional encoding
        self.pos_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
        )

        self.ViT = SimpleViT(dim=d_model+time_dim, time_dim=time_dim, depth=depth, heads=heads, mlp_dim=2048)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, x, t, pe):
        t = self.time_mlp(t).unsqueeze(1)
        t_extended = t.repeat(1,x.size(1),1)
        pe = self.pos_mlp(pe)

        x_final = torch.cat(((x+pe), t_extended), dim=2)
        x_final = self.ViT(x_final)
        return x_final

@torch.no_grad()
def p_sample(model, x, t, pe, t_index, betas, mode, num_fixed_frames=5):
    if mode=='reconstruct':
        # Direct reconstruction
        return model(x,t)
    
    else: 
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, pe) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            x_final = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            x_final = model_mean + torch.sqrt(posterior_variance_t) * noise
        x_final = condition_projection(x_final, x, num_fixed_frames)
        return x_final


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, timesteps, pe, betas, shape, start_noise, mode, num_fixed_frames=5):
    device = next(model.parameters()).device

    b = shape[0]
    imgs = [] 
    if start_noise == None:
        img = torch.randn(shape, device=device)   
    else:
        img = start_noise

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), pe, i, betas, mode, num_fixed_frames)
        imgs.append(img)
    return imgs

@torch.no_grad()
def sample(model, latent_dim, sequence_length, timesteps, pe, betas, batch_size, start_noise = None, mode = 'noise'):
    return p_sample_loop(model, timesteps, pe, betas, shape=(batch_size, sequence_length, latent_dim), start_noise = start_noise, mode=mode)

import torch

def main():
    # Define hyperparameters
    batch_size = 4
    sequence_length = 20
    feature_dim = 256
    window_size = 20
    depth = 4
    heads = 6
    hidden_dim = feature_dim
    d_model = feature_dim
    timesteps = 10
    betas = torch.linspace(0.1, 0.2, timesteps)
    num_fixed_frames = 5
    mode = 'noise'  # Can be 'noise' or 'reconstruct'

    # Initialize model
    denoise_model = DenoiseViT(window_size=window_size, depth=depth, heads=heads, d_model=d_model,
                              hidden_dim=hidden_dim, norm_type='layer')
    denoise_model.to('cuda')

    # load the dataset and create random noise
    x_start = torch.randn((batch_size, sequence_length, feature_dim))
    train_path = "easg_classification/dataset_video/encoded_videos_train_256.pth"
    train_dataset = EASGvideo(train_path)
    trainloader = DataLoader(train_dataset, batch_size=batch_size)
    t = torch.randint(0, timesteps, (batch_size,), device='cuda').long()

    # Define alpha and noise parameters
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Run the denoising model
    noise = None
    for batch in trainloader:
        batch.to('cuda')
        # Create positional encoding
        pe = positional_encoding(feature_dim, sequence_length, batch.size(0))
        loss = p_losses(denoise_model, batch, t, pe, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise, mode=mode, num_fixed_frames=num_fixed_frames)
        print(f"Loss: {loss.item()}")
        break

    # Test sampling process
    samples = sample(denoise_model, feature_dim, sequence_length, timesteps, pe, betas, batch_size, start_noise=None, mode=mode)
    print(f"Sampled output shape: {samples[-1].shape}")

if __name__ == "__main__":
    main()