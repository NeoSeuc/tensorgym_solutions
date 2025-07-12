import torch


# https://tensorgym.com/exercises/0
def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = None
    if x.shape == y.shape:
        result = torch.add(x, y)

    return result


assert torch.equal(add_tensors(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])), torch.tensor([5, 7, 9]))
