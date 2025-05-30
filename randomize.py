import numpy as np
import torch

def generate_random_lifetimes(T, queries, gt_trajs, gt_vis, t, seed=None):
    N = queries.shape[1]

    assert 2 * t < T, "t must be less than T/2 to allow full coverage region."
    
    rng = np.random.default_rng(seed)

    limit_mask = queries[0, :, 0] <= t
    
    # Random start in [0, t)
    start_frames = torch.tensor(rng.integers(low=0, high=t, size=int(limit_mask.sum().item())), dtype=torch.float, device=queries.device)
    # start_frames = queries[0,limit_mask, 0].clone()  # Copy existing start frames for limited queries
    # start_frames = np.sort(start_frames)
    
    # Random stop in (T-t, T)
    stop_frames = torch.tensor(rng.integers(low=T - t, high=T, size=int(limit_mask.sum().item())), dtype=torch.float, device=queries.device)
    # stop_frames = torch.ones_like(start_frames, device=queries.device) * (T - 1)  # Copy existing stop frames for limited queries

    query_frames = queries[0, :, 0]
    query_frames[limit_mask] = start_frames
    query_coords = queries[0, :, 1:3]
    for i in range(queries.shape[1]):
        if limit_mask[i]:
            query_coords[i] = gt_trajs[0, int(query_frames[i]), i]

    queries[0, :, 0] = query_frames
    queries[0, :, 1:3] = query_coords

    end_frames = torch.ones(queries.shape[1], device=queries.device) * (T-1)  # Tracked till this frame but not after
    end_frames[limit_mask] = stop_frames

    queries_squeezed = queries.squeeze(0)  # shape: (N, 3)
    sorted_t, sort_indices = queries_squeezed[:, 0].sort()
    queries_sorted = queries_squeezed[sort_indices]
    start_frames_sorted = queries_sorted[:, 0]
    end_frames_sorted = end_frames[sort_indices]
    gt_traj_sorted = gt_trajs[:,:,sort_indices]
    gt_vis_sorted = gt_vis[:,:,sort_indices]
    
    ids_list = []
    removed_indices_list = []
    new_queries_list = []

    for t in range(T):
        active_indices = [
            i for i in range(N) if start_frames_sorted[i] < t <= end_frames_sorted[i]
        ]
        active_indices = torch.tensor(active_indices, dtype=torch.long)

        if len(ids_list) == 0:
            removed_indices_list.append([])
            new_queries_list.append(0)
        else:
            removed_indices = torch.arange(ids_list[-1].shape[0])[torch.isin(ids_list[-1], active_indices, invert=True)]
            removed_indices_list.append(removed_indices.tolist())
            new_queries_list.append(torch.isin(active_indices, ids_list[-1], invert=True).sum().item())

        ids_list.append(active_indices)

    print("start_frames_sorted:", start_frames_sorted)
    print("end_frames_sorted:", end_frames_sorted)
    print("ids_list:\n" + "\n".join(
        [f"{i}: ids={ids.tolist()}, queries={torch.round(queries_sorted[ids]).tolist()}"
         for i, ids in enumerate(ids_list) 
         if ids_list[i].tolist() != ids_list[i-1].tolist()]
    ))
    print("removed_indices_list:", [[i, rm] for i, rm in enumerate(removed_indices_list) if len(rm) > 0])
    print("new_queries_list:", [[i, nq] for i, nq in enumerate(new_queries_list) if nq > 0])

    return ids_list, removed_indices_list, new_queries_list, gt_traj_sorted, gt_vis_sorted, queries_sorted.unsqueeze(0), end_frames_sorted