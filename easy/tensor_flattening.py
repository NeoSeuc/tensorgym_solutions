import torch


# https://tensorgym.com/exercises/1
def flatten(x: torch.Tensor) -> torch.Tensor:
    result = torch.Tensor.view(x, -1)
    return result


t = torch.Tensor([[1, 3],
                  [4, 6]])

print(flatten(t))  # should return tensor([1., 3., 4., 6.])