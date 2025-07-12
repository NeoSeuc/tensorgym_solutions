import torch


# https://tensorgym.com/exercises/14
def fill_tensor_with_value(x: torch.Tensor, value: int) -> torch.Tensor:
    result = None
    # Try to only use +-/* operators. Don't create new tensor with torch.tensor or use any torch functions.
    #####. Solution should be 1 line
    result = x + (value - x) 
    return result


t = torch.tensor([[3, 3],
                  [4, 4]])

value = 7

print(fill_tensor_with_value(t, value))  # should return tensor([[7, 7], [7, 7]])