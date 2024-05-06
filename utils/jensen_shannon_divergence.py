import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

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
    return result_jsd

# Example usage with arbitrary-dimensional datasets
samples_p = np.random.randn(96, 1000, 1)  # Sample from distribution P (3 dimensions)
samples_q = np.random.randn(96, 1000, 1) * 5 + 1  # Sample from distribution Q (3 dimensions)
# samples_q = np.random.randn(96, 1000, 1)
# samples_q = np.random.randn(96, 1000, 1) * 10 + 10  # Sample from distribution Q (3 dimensions)

# print("Jensen-Shannon Divergence (JSD) between samples_p and samples_q:", jsd(samples_p, samples_q))



import torch
from ignore_rat import WeightsGenerator, NetworkExecuterParallel, MouseLossFunctionOptimised

executer = NetworkExecuterParallel(1000, scaling_g=0.15)
loss_function = MouseLossFunctionOptimised()

# Create dataset
J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

J_array = torch.tensor(J_array)
P_array = torch.tensor(P_array)
w_array = torch.tensor(w_array)
wg = WeightsGenerator(J_array, P_array, w_array, 1000, forward_mode=True)
W = wg.generate_weight_matrix()
tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
y_E1, y_I1 = tuning_curves[:800], tuning_curves[800:]
x, y, z = y_E1.shape
y_E1 = y_E1.reshape(x, y * z).unsqueeze(2).numpy()
x, y, z = y_I1.shape
y_I1 = y_I1.reshape(x, y * z).unsqueeze(2).numpy()

# Create second dataset
J_array = [0.587113, -1.006514, -1.494353, -0.07066000000000026]  # Should be low
P_array = [-3.455629, -0.830209, 0.48249700000000006, 1.6340260000000002]
w_array = [0.6527120000000001, 1.989928, -0.7805890000000001, 0.11133499999999995]

J_array = [-1.7346010553881064, -2.586689344097943, -1.3862943611198906, -3.1780538303479458]  # Should be high
P_array = [-1.265666373331276, -0.6190392084062235, -1.265666373331276, -0.6190392084062235]
w_array = [-1.0986122886681098, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098] 

J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886]   # Should be lowest

J_array = [-1.233466, 0.362151, -1.84669, 0.275236]
P_array = [-0.6024259999999999, -1.489425, 0.3773059999999996, 0.7860370000000003]
w_array = [-2.260707, 0.19369199999999995, 1.2847419999999998, 1.3065529999999999] 

J_array = torch.tensor(J_array)
P_array = torch.tensor(P_array)
w_array = torch.tensor(w_array)
wg = WeightsGenerator(J_array, P_array, w_array, 1000, forward_mode=True)
W = wg.generate_weight_matrix()
tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
y_E2, y_I2 = tuning_curves[:800], tuning_curves[800:]
x, y, z = y_E2.shape
y_E2 = y_E2.reshape(x, y * z).unsqueeze(2).numpy()
x, y, z = y_I2.shape
y_I2 = y_I2.reshape(x, y * z).unsqueeze(2).numpy()


print(jsd(y_E1, y_E2))
print(jsd(y_I1, y_E2))