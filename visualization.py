import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sample_inspection(samples, title, bounds=(-15, 15), ax=None, xlabel='x', ylabel='y'):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    samples = samples.detach().cpu().numpy()
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def density_inspection(samples, title, bounds=(-15, 15), ax=None,xlabel='x', ylabel='y'):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    samples = samples.detach().cpu().numpy()
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], ax=ax, cmap="Blues", fill=True)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_forward_process(sde, x_0, timesteps=[0.0, 0.3, 0.6, 1.0], bounds=(-15, 15)):
   
    fig, axes = plt.subplots(1, len(timesteps), figsize=(4 * len(timesteps), 4))
    if len(timesteps) == 1:
        axes = [axes]
    for i, t in enumerate(timesteps):
        t_tensor=torch.tensor([t]).view(1, 1).expand(x_0.shape[0], 1).to(x_0.device)
        x_t = sde.sample_xt(x_0,t_tensor)
        sample_inspection(x_t, title=f"Forward Process (t={t:.1f})", bounds=bounds, ax=axes[i])
    plt.tight_layout()
    return fig


def plot_backward_process(score_model, num_samples=1000, timesteps=[1.0, 0.6, 0.3, 0.0], num_timesteps=300, sigma=1.0, dim=2, bounds=(-15, 15)):
    # define figure properties and data sturcture
    fig, axes = plt.subplots(1, len(timesteps), figsize=(4 * len(timesteps), 4))
    if len(timesteps) == 1:
        axes = [axes]
    x_t = torch.randn(num_samples, dim).to(score_model.net[0].weight.device)  # Start from noise
    ts = torch.linspace(1.0, 0.0, num_timesteps + 1).to(x_t.device)
    dt = -1.0 / num_timesteps
    timestep_indices = [int((1.0 - t) * num_timesteps) for t in timesteps]
    saved_samples = []
    for i, t in enumerate(ts):
        t = t.view(1, 1).expand(num_samples, 1)
        score = score_model(x_t, t)

        drift = (x_t / (1 - t + 1e-5)) + score  # Add small epsilon to avoid division by zero at t=1
        #noise = torch.randn_like(x_t) * (-dt)**0.5
        #drift = -0.5 * sigma**2 * score
        noise = sigma * torch.randn_like(x_t) * (-dt)**0.5
        x_t = x_t + drift * dt + noise
        if i in timestep_indices:
            saved_samples.append(x_t.clone())
    for i, (t, samples) in enumerate(zip(timesteps, saved_samples)):
        sample_inspection(samples, title=f"Backward Process (t={t:.1f})", bounds=bounds, ax=axes[i])
    plt.tight_layout()
    return fig