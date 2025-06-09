import torch

class PatchSplitter:
    def __init__(self, interp_shape, original_image_dims):
        self.interp_shape = interp_shape
        self.original_image_dims = original_image_dims
        self.tracks_mask = None
        self.max_patch_queries = None

    def split_video(self, frame):
        _, C, H, W = frame.shape
        patches = []
        h, w = self.interp_shape
        patches.append(frame[:,:,:h, :w])
        patches.append(frame[:,:,:h, W-w:])
        patches.append(frame[:,:,H-h:, :w])
        patches.append(frame[:,:,H-h:, W-w:])

        return torch.cat(patches, dim=0)

    def split_removed_indices(self, removed_indices):
        if self.max_patch_queries is None:
            return torch.ones(1, device='cuda', dtype=torch.bool)
        orig_mask = torch.ones(self.tracks_mask.shape[1], device=self.tracks_mask.device, dtype=torch.bool)
        orig_mask[removed_indices] = False

        split_mask = torch.zeros(4, self.max_patch_queries, device=self.tracks_mask.device, dtype=torch.bool)  # Removing the extra padded queries every step
        split_mask[0, :(self.tracks_mask[0]).sum()] = orig_mask[self.tracks_mask[0]]
        split_mask[1, :(self.tracks_mask[1]).sum()] = orig_mask[self.tracks_mask[1]]
        split_mask[2, :(self.tracks_mask[2]).sum()] = orig_mask[self.tracks_mask[2]]
        split_mask[3, :(self.tracks_mask[3]).sum()] = orig_mask[self.tracks_mask[3]]

        return split_mask

    def split_queries(self, queries, t):
        H, W = self.original_image_dims
        h, w = self.interp_shape
        pad_query = torch.tensor([t, 0.0, 0.0], device=queries.device)
        mask0 = (queries[0,:,1] >= 0) & (queries[0,:,1] < w) & (queries[0,:,2] >= 0) & (queries[0,:,2] < h)
        mask1 = (queries[0,:,1] >= W-w) & (queries[0,:,1] < W) & (queries[0,:,2] >= 0) & (queries[0,:,2] < h)
        mask2 = (queries[0,:,1] >= 0) & (queries[0,:,1] < w) & (queries[0,:,2] >= H-h) & (queries[0,:,2] < H)
        mask3 = (queries[0,:,1] >= W-w) & (queries[0,:,1] < W) & (queries[0,:,2] >= H-h) & (queries[0,:,2] < H)
        self.tracks_mask = torch.stack([mask0, mask1, mask2, mask3], dim=0)  # 4,N (N is num queries in this timestep - changes every step)

        self.max_patch_queries = max(mask0.sum(), mask1.sum(), mask2.sum(), mask3.sum())
        patch_queries = pad_query.repeat(4, self.max_patch_queries, 1)
        patch_queries[0, :mask0.sum(), :] = queries[0, mask0, :]
        patch_queries[1, :mask1.sum(), :] = queries[0, mask1, :] - torch.tensor([0, W-w, 0], device=queries.device)
        patch_queries[2, :mask2.sum(), :] = queries[0, mask2, :] - torch.tensor([0, 0, H-h], device=queries.device)
        patch_queries[3, :mask3.sum(), :] = queries[0, mask3, :] - torch.tensor([0, W-w, H-h], device=queries.device)

        return patch_queries

    def combine_tracks(self, tracks, status):
        N = self.tracks_mask.shape[1]
        H, W = self.original_image_dims
        h, w = self.interp_shape

        combined_tracks = torch.zeros(4, N, 2, device=tracks.device)  # 4,N,2
        combined_tracks[0, self.tracks_mask[0], :] = tracks[0, :self.tracks_mask[0].sum(), :]
        combined_tracks[1, self.tracks_mask[1], :] = tracks[1, :self.tracks_mask[1].sum(), :] + torch.tensor([W-w, 0.0], device=tracks.device)
        combined_tracks[2, self.tracks_mask[2], :] = tracks[2, :self.tracks_mask[2].sum(), :] + torch.tensor([0.0, H-h], device=tracks.device)
        combined_tracks[3, self.tracks_mask[3], :] = tracks[3, :self.tracks_mask[3].sum(), :] + torch.tensor([W-w, H-h], device=tracks.device)

        combined_status = torch.zeros(4, N, device=status.device)  # 4,N
        combined_status[0, self.tracks_mask[0]] = status[0, :self.tracks_mask[0].sum()]
        combined_status[1, self.tracks_mask[1]] = status[1, :self.tracks_mask[1].sum()]
        combined_status[2, self.tracks_mask[2]] = status[2, :self.tracks_mask[2].sum()]
        combined_status[3, self.tracks_mask[3]] = status[3, :self.tracks_mask[3].sum()]

        row_idx = combined_status.argmax(dim=0)
        final_status = combined_status.max(dim=0, keepdim=True).values
        tracks_perm = combined_tracks.permute(1, 0, 2)  # (N, B, 2)
        index = row_idx[:, None].expand(N, 1).unsqueeze(-1).expand(N, 1, 2)
        final_tracks = tracks_perm.gather(1, index)  # (N, 1, 2)
        final_tracks = final_tracks.squeeze(1).unsqueeze(0)

        return final_tracks, final_status