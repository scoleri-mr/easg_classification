import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def condition_projection(x, num_fixed_frames=5):
    """
    Ensures that the first `num_noise_free` frames of the video are noise-free.
    
    :param x: Input tensor of shape (batch_size, sequence_length+1, feature_dim)
    :param num_noise_free: Number of initial elements to keep noise-free
    :return: Modified tensor with the first `num_noise_free` elements unchanged
    """
    x[:, 1:num_fixed_frames] = x[:, 1:num_fixed_frames].clone()
    return x

def positional_encoding(d_model, length):
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
    return pe.view(1,20,512)

class positional_mlp(nn.Module):
    def __init__(self, d_model):
        super(positional_mlp, self).__init__()
        self.d_model = d_model
        self.pos_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        
    def forward(self, positional_encoding):
        return self.pos_mlp(positional_encoding)

# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, num_fixed_frames=5):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    x_noisy = condition_projection(x_noisy, num_fixed_frames)  # Apply condition projection
    return 

# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1", mode='noise', num_fixed_frames=5):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, num_fixed_frames)
    out_pred = denoise_model(x_noisy, t, cond)  # if reconstruct=True out_pred will cointain the reconstructed x, otherwise the predicted noise

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

# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, n_layers, norm_type):
        super(DenoiseNN, self).__init__()
        self.norm_type = norm_type

        # time encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # positional encoding
        self.pos_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
        )

        # TODO: implement visual transformer
        self.ViT = ...

        if self.norm_type == 'batch':
            n_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
            self.bn = nn.ModuleList(n_layers)
        elif self.norm_type == 'layer':
            n_layers = [nn.LayerNorm(hidden_dim) for i in range(n_layers-1)]
            self.ln = nn.ModuleList(n_layers)
        else:
            raise Exception("Wrong normalization layer. Choose between 'batch' and 'layer'.")

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, x, t, pe):
        t = self.time_mlp(t).unsqueeze(1)
        pe = self.pos_enc(pe)

        x_final = torch.cat((t, (x+pe)), dim=1)
        x_final = self.ViT(x_final)
        return x_final

@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas, mode, num_fixed_frames=5):
    if mode=='reconstruct':
        # Direct reconstruction
        return model(x,t,cond)
    
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
            x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            x_final = model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            x_final = model_mean + torch.sqrt(posterior_variance_t) * noise
        x_final = condition_projection(x_final, num_fixed_frames)
        return x_final


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape, start_noise, mode, num_fixed_frames=5):
    device = next(model.parameters()).device

    b = shape[0]
    imgs = [] 
    if start_noise == None:
        img = torch.randn(shape, device=device)   
    else:
        img = start_noise

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas, mode, num_fixed_frames)
        imgs.append(img)
    return imgs

@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size, start_noise = None, mode = 'noise'):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim), start_noise = start_noise, mode=mode)