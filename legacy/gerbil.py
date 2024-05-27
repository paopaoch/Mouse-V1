"""THIS FILE CONTAINS THE FUNCTION TO CALCULATE THE JENSEN SHANNON DIVERGENCE.
THIS WAS SEPARATED OUT FROM rat.py AS THE CODE IS WRITTEN IN A PROCEDURAL WAY AND USES NUMPY.
"""
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import torch

def estimate_pdf(samples, bandwidth=0.5):
    # Estimate probability density function (PDF) using kernel density estimation (KDE)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(samples)
    return kde

def jensen_shannon_divergence_from_samples(samples_p, samples_q, bandwidth=0.5, num_points=100):
    # Estimate PDFs from samples_p and samples_q using kernel density estimation (KDE)
    kde_p = estimate_pdf(samples_p, bandwidth=bandwidth)
    kde_q = estimate_pdf(samples_q, bandwidth=bandwidth)
    
    # Determine the dimensions of the samples
    num_dims = samples_p.shape[1]
    
    # Generate grid of points for PDF evaluation
    ranges = []
    for d in range(num_dims):
        min_val = min(np.min(samples_p[:, d]), np.min(samples_q[:, d]))
        max_val = max(np.max(samples_p[:, d]), np.max(samples_q[:, d]))
        ranges.append((min_val, max_val))
    
    # Create meshgrid for PDF evaluation
    grid_points = [np.linspace(r[0], r[1], num_points) for r in ranges]
    grid_points_mesh = np.meshgrid(*grid_points)
    grid_points_stack = np.column_stack([mesh.ravel() for mesh in grid_points_mesh])
    
    # Evaluate PDFs at the grid points
    pdf_p = np.exp(kde_p.score_samples(grid_points_stack)).reshape(grid_points_mesh[0].shape)
    pdf_q = np.exp(kde_q.score_samples(grid_points_stack)).reshape(grid_points_mesh[0].shape)
    
    # Compute the average distribution M
    m_distribution = 0.5 * (pdf_p + pdf_q)
    
    # Compute KL divergences
    kl_p_m = entropy(pdf_p.ravel(), m_distribution.ravel())
    kl_q_m = entropy(pdf_q.ravel(), m_distribution.ravel())
    
    # Compute Jensen-Shannon Divergence (JSD)
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd


def jsd(samples_p, samples_q):
    result_jsd = 0
    for i, j  in zip(samples_p, samples_q):
        result_jsd += jensen_shannon_divergence_from_samples(i, j)
    return torch.tensor(result_jsd)


def get_jsd_loss(
        x_E: torch.Tensor, 
        y_E: torch.Tensor, 
        x_I: torch.Tensor, 
        y_I: torch.Tensor, 
        avg_step: torch.Tensor, 
        avg_step_weighting=0.002,
        bessel_val=torch.tensor(0), 
        bessel_val_weighting=torch.tensor(1), 
        x_centralised=False, y_centralised=False):
    if not x_centralised:
        x_E = centralise_all_curves(x_E)
        x_I = centralise_all_curves(x_I)
    
    if not y_centralised:
        y_E = centralise_all_curves(y_E)
        y_I = centralise_all_curves(y_I)
    
    x, y, z = y_E.shape
    y_E = y_E.reshape(x, y * z).unsqueeze(2)
    x, y, z = y_I.shape
    y_I = y_I.reshape(x, y * z).unsqueeze(2)
    x, y, z = x_E.shape
    x_E = x_E.reshape(x, y * z).unsqueeze(2)
    x, y, z = x_I.shape
    x_I = x_I.reshape(x, y * z).unsqueeze(2)

    y_E = y_E.detach().cpu().numpy()
    y_I = y_I.detach().cpu().numpy()
    x_E = x_E.detach().cpu().numpy()
    x_I = x_I.detach().cpu().numpy()

    E = jsd(x_E, y_E)
    I = jsd(x_I, y_I)

    return E + I + (torch.maximum(torch.tensor(1.), avg_step) - 1) * avg_step_weighting + bessel_val * bessel_val_weighting, E + I


def get_max_index(tuning_curve):
    max_index = torch.argmax(tuning_curve[7])
    return max_index


def centralise_curve(tuning_curve):
    max_index = get_max_index(tuning_curve)  # instead of max index, taking the mean might be better?
    shift_index = 6 - max_index  # 6 is used here as there are 13 orientations
    new_tuning_curve = torch.roll(tuning_curve, int(shift_index), dims=1)
    return new_tuning_curve


def centralise_all_curves(responses):
    tuning_curves = []
    for tuning_curve in responses:
        tuning_curves.append(centralise_curve(tuning_curve))
    return torch.stack(tuning_curves)