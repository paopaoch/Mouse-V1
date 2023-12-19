import torch
from tqdm import tqdm
from time import time, sleep


def kernel(x: torch.Tensor, y: torch.Tensor, w=1):
    return torch.exp(-torch.sum((x - y) ** 2, dim=(-2, -1)) / (2 * w**2))


def individual_terms_vanilla(x: torch.Tensor, y: torch.Tensor, device="cpu") -> torch.Tensor:
    if torch.equal(x, y):  # This speeds up computation by not doing the cross term twice
        N = x.shape[0]
        accum_output1 = torch.tensor(0, device=device)
        for i in range(int(N / 2), N):
            for j in range(int(N / 2)):
                    accum_output1 = accum_output1 + kernel(x[i], x[j])
        
        accum_output1 = accum_output1 * 2

        for i in range(int(N / 2)):
            for j in range(int(N / 2)):
                    accum_output1 = accum_output1 + kernel(x[i], x[j])

        for i in range(int(N / 2), N):
            for j in range(int(N / 2), N):
                    accum_output1 = accum_output1 + kernel(x[i], x[j])
    

        return accum_output1 / (N * N)

    else:  # this is each term in the MMD2 function
        N = x.shape[0]
        M = y.shape[0]
        accum_output = torch.tensor(0, device=device)
        for i in range(N):
            for j in range(M):
                    accum_output = accum_output + kernel(x[i], y[j])

        return accum_output / (N * M)
    

def individual_terms_single_loop(x: torch.Tensor, y: torch.Tensor, device="cpu"):
        N = x.shape[0]
        M = y.shape[0]
        accum_output = torch.tensor(0, device=device)
        for i in range(N):
            x_repeated = x[i, :, :].unsqueeze(0).expand(M, -1, -1)
            accum_output = accum_output + torch.mean(kernel(y, x_repeated))
        return accum_output / N


def MMD2(x: torch.Tensor, y: torch.Tensor, device="cpu"):
    XX  = individual_terms_single_loop(x, x, device=device)
    XY  = individual_terms_single_loop(x, y, device=device)
    YY  = individual_terms_single_loop(y, y, device=device)
    return XX + YY - 2 * XY


if torch.cuda.is_available():
    device = "cuda"
    print("Model moved to GPU.")
else:
    device = "cpu"
    print("GPU not available. Keeping the model on CPU.")

tensor1 = (torch.rand(10000, 8, 12, device=device) * 2 + 100) * 10

tensor2 = torch.rand(10000, 8, 12, device=device)

tensor3 = (torch.rand(10000, 8, 12, device=device) * 2 + 100) * 10


print(individual_terms_single_loop(tensor1, tensor1, device=device))

start = time()
print(individual_terms_single_loop(tensor2, tensor2, device=device))
print("end: ", time() - start)

start = time()
print(individual_terms_single_loop(tensor1, tensor2, device=device))
print("end: ", time() - start)

start = time()
print(individual_terms_single_loop(tensor2, tensor1, device=device))
print("end: ", time() - start)


start = time()
print(MMD2(tensor1, tensor2, device=device))
print("end: ", time() - start)


start = time()
print(MMD2(tensor2, tensor3, device=device))
print("end: ", time() - start)


start = time()
print(MMD2(tensor1, tensor3, device=device))
print("end: ", time() - start)
