import torch
from typing import Tuple, Optional

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def add_support_grid(queries, local_grid_size, local_grid_extent): 
    B = 1  # Assuming batch size is always 1 in this context
    queries = queries.unsqueeze(0)
    if local_grid_size > 0:
        aug_queries = []

        for i in range(queries.shape[1]):
            frame = queries[0, i, 0].item()
            grid_pts = get_points_on_a_grid(
                local_grid_size, 
                (local_grid_extent, local_grid_extent),
                (queries[0, i, 2].item(), queries[0, i, 1].item()),  # ensure x, y order
                device=queries.device,
            )
            grid_pts = torch.cat([torch.ones_like(grid_pts[:, :, :1]) * frame, grid_pts], dim=2)
            grid_pts = grid_pts.repeat(B, 1, 1)

            # Append query first, then its support grid
            aug_queries.append(queries[:, i:i+1, :])  # Extract and keep the original shape
            aug_queries.append(grid_pts)

        # Concatenate along the second dimension to maintain the interleaved pattern
        queries = torch.cat(aug_queries, dim=1)

    return queries.squeeze(0)
    
def augment_removed_indices(removed_indices, local_grid_size):
    if local_grid_size > 0:
        aug_removed_indices = []
        for i in range(len(removed_indices)):
            aug_index = removed_indices[i]*(1 + local_grid_size**2)
            aug_removed_indices.append(aug_index)
            removed_grid_indices = list(range(aug_index + 1, aug_index + 1 + local_grid_size**2))
            aug_removed_indices += removed_grid_indices

        removed_indices = aug_removed_indices
    return removed_indices