import torch
from time import time


def kernel(x: torch.Tensor, y: torch.Tensor, w=1):
    return torch.exp(-torch.sum((x - y) ** 2, dim=(-2, -1)) / (2 * w**2))


def broadcasted_MMD2(X, Y, device):
    XX = torch.mean(kernel(X[None, :, :, :], X[:, None, :, :]))
    XY = torch.mean(kernel(X[None, :, :, :], Y[:, None, :, :]))
    YY = torch.mean(kernel(Y[None, :, :, :], Y[:, None, :, :]))

    return XX - 2 * XY + YY


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
    print("GPU available! Using GPU")
else:
    device = "cpu"
    print("GPU not available. Using CPU.")

def time_it(func, desc, x, y):
    print(desc)
    print("shape of X: ",  x.shape)
    print("shape of Y: ",  y.shape)
    start = time()
    print("MMD value: ", func(x, y, device=device))
    print("Time taken: ", time() - start, 'seconds\n')

if __name__ == "__main__":
    tensor1 = (torch.rand(10000, 8, 12, device=device) * 4 + 1000) * 5

    tensor2 = torch.rand(10000, 8, 12, device=device)

    tensor3 = (torch.rand(10000, 8, 12, device=device) * 4 + 1000) * 5

    tensor4 = torch.rand(81, 8, 12, device=device)

    tensor5 = torch.rand(4000, 8, 12, device=device)

    print("\n###### Running MMD functions ######\n")
    time_it(individual_terms_vanilla, "Individual MMD term vanilla: ", tensor4, tensor5)
    time_it(individual_terms_single_loop, "Individual MMD term single loop: ", tensor4, tensor5)
    time_it(MMD2, "MMD between two different distributions: ", tensor1, tensor2)
    time_it(MMD2, "MMD between two different distributions: ", tensor2, tensor3)
    time_it(MMD2, "MMD between two same distributions with different samples: ", tensor1, tensor3)
    time_it(MMD2, "MMD between two exact same distributions : ", tensor1, tensor1)
    time_it(MMD2, "compare MMD2 and broadcasting, MMD2: ", tensor4, tensor5)
    
    # this might not run on some machine due to high memory, reduce the size of the tensor then re run
    time_it(broadcasted_MMD2, "compare MMD2 and broadcasting, broadcasting: ", tensor4, tensor5)

    time_it(MMD2, "MMD of actual V1 project shape: ", tensor3, tensor4)
    time_it(MMD2, "MMD of actual V1 project shape swaped: ", tensor4, tensor3)