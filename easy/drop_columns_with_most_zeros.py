import torch


def drop_column(A: torch.tensor) -> torch.tensor:
    """
    Drops the column of A containing the most 0 values.

    Parameters:
        tensor: A 2D tensor.
    Returns:
        tensor: The input tensor with the column containing the most 0 values removed.
    """

    zero_counts = torch.sum(A == 0, dim=0)
    col_to_drop = torch.argmax(zero_counts)
    result = torch.cat((A[:, :col_to_drop], A[:, col_to_drop + 1:]), dim=1)

    return result


t = torch.tensor([[1, 0, 3],
                  [4, 5, 6]])

print(drop_column(t))  # should return tensor([[1, 3], [4, 6]])