import torch
from tqdm import tqdm
from time import time

def kernel(x: torch.Tensor, y: torch.Tensor, w=1):
    return torch.exp(-torch.sum((x - y) ** 2) / (2 * w**2))


def individual_terms(x: torch.Tensor, y: torch.Tensor, device="cpu") -> torch.Tensor:
    if torch.equal(x, y):
        N = x.shape[0]
        accum_output1 = torch.zeros(*x.shape[1:], device=device)
        print(accum_output1.device)
        print(x[0].device)
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

        return torch.sum(accum_output1) / (N * N)
    else:
        N = x.shape[0]
        M = y.shape[0]
        accum_output = torch.zeros(*x.shape[1:], device=device)
        for i in tqdm(range(N)):
            for j in range(M):
                    accum_output = accum_output + kernel(x[i], y[j])

        return torch.sum(accum_output) / (N * M)

if torch.cuda.is_available():
    device = "cuda"
    print("Model moved to GPU.")
else:
    device = "cpu"
    print("GPU not available. Keeping the model on CPU.")

tensor1 = torch.rand(81, 8, 12, device=device)

tensor2 = torch.rand(10000, 8, 12, device=device)


print(individual_terms(tensor1, tensor1, device=device))

start = time()
print(individual_terms(tensor2, tensor2))
print("end: ", time() - start)