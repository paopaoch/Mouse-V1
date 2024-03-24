import torch
import torch.autograd.forward_ad as fwAD
from tqdm import tqdm


def forward_diff(func, parameters: list[torch.Tensor], hyperparameters=None, device="cpu"):
    """A forward differentiation wrapper.
    Func: Must be a function that takes in list of tensor parameters and a hyperparameters.

    these parameters and hyperparameters should be extracted within the function to be differentiated"""
    tangent = torch.tensor(1., device=device)
    gradients = []
    func_output = torch.tensor([0.], device=device)
    for i, parameter in tqdm(enumerate(parameters)):
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(parameter, tangent)
            parameters_with_dual = parameters.copy()
            parameters_with_dual[i] =  dual_input
            
            dual_output = func(parameters_with_dual, hyperparameters, device=device)
            dual_tensor = fwAD.unpack_dual(dual_output)
            jvp = dual_tensor.tangent
            gradients.append(jvp)
            func_output += dual_tensor.primal
    return gradients,  func_output / len(parameters)



if __name__ == "__main__":
    def fn(var, hyperparams):
        x, y = var
        a, b = hyperparams
        p = (a * x ** 2 + b * y ** 2 + x * y) / 100
        return p + torch.randn(1)
    
    x = torch.tensor(4.)
    y = torch.tensor(2.)

    a = torch.tensor(3.)
    b = torch.tensor(5.)

    print(fn([x, y], [a, b]))

    print(forward_diff(fn, [x, y], [a, b]))