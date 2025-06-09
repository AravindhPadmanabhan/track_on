import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from trackon.track_on import TrackOn

from utils.coord_utils import get_points_on_a_grid
from utils.coord_utils import indices_to_coords

from trackon.split_patches import PatchSplitter


class TrackOnFF(TrackOn):
    def __init__(self, args):
        super().__init__(args=args)

        self.query_times = None
        # self.set_memory_size(new_memory_size, new_memory_size)

        self.t = 0

        self.q_init = None

        self.spatial_memory = None
        self.context_memory = None
        self.past_occ = None
        self.past_mask = None

        self.prev_ft = None
        self.prev_spatial_memory = None
        self.prev_context_memory = None
        self.prev_past_mask = None
        self.prev_past_occ = None
        self.prev_p = None
        self.prev_v = None

        self.splitter = None

    def init_queries_and_memory(self, queries, frame):
        # :args queries: (N, 3)         (t, x, y) in given frame
        # :args frame: (1, C, H, W)     frame to extract features from

        H, W = frame.shape[-2], frame.shape[-1]
        self.splitter = PatchSplitter(self.size, (H, W))
        frame = self.splitter.split_video(frame)  # (B, C, H, W)
        queries = self.splitter.split_queries(queries.unsqueeze(0), self.t)  # (B, N_max, 3)

        self.t = 0
        self.query_times = queries[:, :, 0]     # (B,N)
        queries = queries[:,:,1:]  # (B, N, 2)
        self.N = queries.shape[1]
        B = queries.shape[0]

        self.device = queries.device
        self.queries = queries

        # === === ===

        # === Sample Queries ===
        pos_query = queries                          # (B, N_prime, 2)
        B, N_prime = pos_query.shape[:2]

        # Normalize positions from [0, H or W] -> [-1, 1] range
        x, y = pos_query[:, :, 0], pos_query[:, :, 1]   # (B, N')
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1
        grid = torch.stack([x_grid, y_grid], dim=-1)    # (B, N_prime, 2)
        grid = grid.view(B, N_prime, 1, 2)              # for grid_sample: (B, N, 1, 2)

        video_frame = frame / 255.0                     # (B, C, H, W)
        video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=False)
        video_frame = self.backbone.normalization(video_frame)  # (B, C, H, W), in [-1, 1]

        f4, f8, _, _ = self.backbone.vit_encoder(video_frame)  # (B, 384, H4, W4)
        f = f4 if self.stride == 4 else f8                     # Choose feature map

        H_prime, W_prime = f.shape[-2:]
        assert H_prime == self.H_prime and W_prime == self.W_prime, \
            f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"

        f = self.backbone.token_projection(f)                 # (B, C, H4, W4)
        f = f + self.backbone.frame_pos_embedding             # (B, C, H4, W4)

        # Sample from f using per-batch grids
        sampled = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False)  # (B, C, N_prime, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N_prime, C)

        q_init = sampled  # (B, N_prime, C)
        self.q_init = q_init
        C = f.shape[1]

        # === === ===
        # === Init memories ===
        max_memory_size = max([self.query_decoder.memory_size, self.sm_query_updater.memory_size])
        query_num = q_init.shape[1]

        # Spatial Memory
        self.spatial_memory = torch.zeros(B, query_num, max_memory_size, C, device=self.device)             # (B, N, max_memory_size, C)

        # Context Memory
        self.context_memory = torch.zeros(B, query_num, max_memory_size, C, device=self.device)             # (B, N, max_memory_size, C)

        # Masking
        self.past_occ = torch.ones(B, query_num, max_memory_size, device=self.device, dtype=torch.bool)     # (B, N, max_memory_size)
        self.past_mask = torch.ones(B, query_num, max_memory_size, device=self.device, dtype=torch.bool)
        # === === ===

    def sample_queries(self, queries, removed_mask):
        B, N_max = queries.shape[:2]
        C = self.prev_ft.shape[1]

        for i in range(B):
            N_new = N_max - removed_mask[i].sum().item()  # Number of new queries to be added
            if N_new == 0:
                continue

            pos_query = queries[i,-N_new:].unsqueeze(0)           # (1, N_new, 2) 
            x, y = pos_query[:, :, 0], pos_query[:, :, 1]  # (1, N_new)
            x_grid = (x / self.size[1]) * 2 - 1
            y_grid = (y / self.size[0]) * 2 - 1
            grid = torch.stack([x_grid, y_grid], dim=-1).view(N_new, 1, 1, 2).to(self.device)                   # (N_new, 1, 1, 2)

            f_t = self.prev_ft[i].expand(N_new, -1, -1, -1).reshape(N_new, C, self.H_prime, self.W_prime)       # (N_new, C, H4, W4)
            sampled = F.grid_sample(f_t, grid, mode='bilinear', padding_mode='border', align_corners=False)     # (N_new, C, 1, 1)
            q_init_new = sampled.reshape(N_new, C)
            assert (self.q_init[i, -N_new:] == 0).all().item(), "Query embedding is being overwritten"          # (N_new, C)
            self.q_init[i, -N_new:] = q_init_new

        

    def update_queries_and_memory(self, queries, removed_indices):
        """Function called at all timesteps t>0 to remove rejected queries and add new queries
        Args:
            queries (Tensor): Contains old queries that are still being tracked and new queries
                added to the previous frame. Shape (N,3) - Each query is of the format [t, x, y]
            removed_indices (List(int)): Indices of the queries that were removed, so that the
                removal can be reflected on the memory
        """
        removed_mask = self.splitter.split_removed_indices(removed_indices)    # (B, N_max_prev)
        queries = self.splitter.split_queries(queries.unsqueeze(0), self.t-1)  # (B, N_max, 3)

        B = queries.shape[0]
        N_max = queries.shape[1]
        self.query_times = queries[:, :, 0]     # (B,N_max)
        if self.t == 1:
            return
        elif N_max == removed_mask.shape[1] and removed_mask.all().item() == True:  # On the offchance that no new queries are added, when all patches have same no. of queries and none removed.
            return

        self.queries = queries[:,:,1:]          # (B, N_max, 2)

        # Remove rejected queries and memory
        M, C = self.prev_spatial_memory.shape[-2:]
        spatial_memory = torch.zeros(B, N_max, M, C, device=self.device)                              # (B, N_max, M, C)
        context_memory = torch.zeros(B, N_max, M, C, device=self.device)                              # (B, N_max, M, C)
        past_occ = torch.ones(B, N_max, M, device=self.device, dtype=torch.bool)                      # (B, N_max, M)
        past_mask = torch.ones(B, N_max, M, device=self.device, dtype=torch.bool)                     # (B, N_max, M)
        p_head_t = torch.zeros(B, N_max, 2, device=self.device)                                       # (B, N_max, 2)
        prev_v = torch.ones(B, N_max, device=self.device, dtype=torch.bool)                          # (B, N_max)
        q_init = torch.zeros(B, N_max, C, device=self.device)                                         # (B, N_max, C)
        for i in range(B):
            spatial_memory[i, :removed_mask[i].sum()] = self.prev_spatial_memory[i, removed_mask[i]]
            context_memory[i, :removed_mask[i].sum()] = self.prev_context_memory[i, removed_mask[i]]
            past_occ[i, :removed_mask[i].sum()] = self.prev_past_occ[i, removed_mask[i]]  
            past_mask[i, :removed_mask[i].sum()] = self.prev_past_mask[i, removed_mask[i]]
            p_head_t[i, :removed_mask[i].sum()] = self.prev_p[i, removed_mask[i]]         
            prev_v[i, :removed_mask[i].sum()] = self.prev_v[i, removed_mask[i]]        
            q_init[i, :removed_mask[i].sum()] = self.q_init[i, removed_mask[i]]           

        self.q_init = q_init
        self.sample_queries(queries, removed_mask)

        # Update memory for new queries
        for i in range(B):
            N_new = N_max - removed_mask[i].sum().item()
            p_head_t[i, -N_new:] = self.queries[i,-N_new:]

        # fake ff_forward to recompute memory with replaced queries
        f_t = self.prev_ft.permute(0, 2, 3, 1)                                                        # (B, H4, W4, C)
        f_t = f_t.reshape(f_t.shape[0], -1, C)                                                        # (B, P, C)
        h_t = self.feature_decoder(f_t)

        q_init = self.q_init
        t= self.t - 1  # Under the assumption that the below calculations are being redone with the new queries that were sampled in the PREVIOUS frame
        query_times = self.query_times                                                                # (B, N_max)
        queried_now_or_before = (query_times <= t)
        assert queried_now_or_before.all().item() == True, "Future points have been queried!"
        q_init_t = self.sm_query_updater(q_init,
                                        spatial_memory,
                                        past_mask,
                                        past_occ,
                                        query_times,
                                        t)   

        q_t = self.query_decoder(q_init_t, 
                                 f_t, 
                                 context_memory.clone(), 
                                 past_mask, 
                                 queried_now_or_before)                                               # (B, N_max, C)
        q_t = self.projection1(q_t)                                                                   # (B, N_max, C)

        c1_t = self.correlation(q_t, h_t)
        q_t, _, _ = self.rerank_module(q_t, h_t, c1_t) 

        q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t, query_times, t)   # (B, N_max, C)

        self.spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)           # (B, N_max, M, C)
        self.context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)             # (B, N_max, M, C)
        self.past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)   # (B, N_max, M)
        self.past_occ = torch.cat([past_occ[:, :, 1:], prev_v.unsqueeze(-1)], dim=2)                     # (B, N_max, M)

    def ff_forward(self, frame):
        # :args frame: (1, C, H, W)     frame to extract features from
        frame = self.splitter.split_video(frame)  # (B, C, H, W)

        # Retrieve Variables 
        q_init = self.q_init
        self.prev_spatial_memory = self.spatial_memory.clone()
        self.prev_context_memory = self.context_memory.clone()
        self.prev_past_occ = self.past_occ.clone()
        self.prev_past_mask = self.past_mask.clone()
        spatial_memory = self.spatial_memory
        context_memory = self.context_memory
        past_occ = self.past_occ
        past_mask = self.past_mask
        t = self.t
        query_times = self.query_times           # (B, N_max)
        queried_now_or_before = (query_times <= t)

        # ##### Spatial Memory - Query Update #####
        q_init_t = self.sm_query_updater(q_init,
                                        spatial_memory,
                                        past_mask,
                                        past_occ,
                                        query_times,
                                        t)
        # ##### ##### #####

        # ##### Visual Encoder #####
        f_t = self.backbone.encode_frames_online(frame)           # (B, P, C)
        self.prev_ft = f_t
        C = f_t.shape[1]
        f_t = f_t.permute(0, 2, 3, 1)                             # (B, H4, W4, C)
        f_t = f_t.reshape(f_t.shape[0], -1, C)                    # (B, P, C)
        h_t = self.feature_decoder(f_t)                           # (B, P, C)
        # ##### ##### #####

        # ##### Query Decoder #####
        q_t = self.query_decoder(q_init_t, f_t, context_memory.clone(), past_mask, queried_now_or_before)       # (B, N, C)
        q_t = self.projection1(q_t)                                                                             # (B, N, C)
        # ##### ##### #####

        # ##### Correlation - 1 #####
        c1_t = self.correlation(q_t, h_t)                                                         # (B, N, P)
        # ##### ##### #####

        # ##### Reranking #####
        q_t, top_k_u_logit, top_k_p = self.rerank_module(q_t, h_t, c1_t)                           # (B, N, C), (B, N, K), (B, N, K, 2) 
        q_t_corr = self.projection2(q_t)                                                           # (B, N, C)
        # ##### ##### #####

        # ##### Correlation - 2 #####
        c2_t = self.correlation(q_t_corr, h_t)                                                                     # (B, N, P)
        p_head_patch_t = indices_to_coords(torch.argmax(c2_t, dim=-1).unsqueeze(1), self.size, self.stride)[:, 0]  # (B, N, 2)
        # ##### ##### #####

        # ##### Offset Prediction #####
        o_t = self.offset_head(q_t_corr, h_t, p_head_patch_t)           # (B, #offset_layers, N, 2)
        p_head_t = p_head_patch_t + o_t[:, -1]       # (B, N, 2)
        # ##### ##### #####

        # ##### Visibility and Uncertainty Prediction #####
        v_t_logit, u_t_logit = self.visibility_head(q_t, h_t, p_head_t)  # (B, N), (B, N)
        # ##### ##### #####

        # ##### Memory Update #####
        # Spatial Memory Update
        q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t, query_times, t)   # (B, N, C)
        spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)                # (B, N, max_memory_size, C)

        # Context Memory Update
        context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)                                     # (B, N, max_memory_size, C)

        # Masking Update
        past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)                           # (B, N, memory_size)
        past_occ = torch.cat([past_occ[:, :, 1:], (torch.sigmoid(v_t_logit) < self.visibility_treshold).unsqueeze(-1)], dim=2)  # (B, N, memory_size)
        # ##### ##### #####

        # Update Variables
        self.spatial_memory = spatial_memory
        self.context_memory = context_memory
        self.past_occ = past_occ
        self.past_mask = past_mask
        self.t += 1

        # Return Pred and Visü
        vis_pred = torch.sigmoid(v_t_logit) > self.visibility_treshold
        conf_pred = 1 - torch.sigmoid(u_t_logit)

        self.prev_p = p_head_t.clone()
        self.prev_v = vis_pred.clone()

        # coord_pred[:, 1] = (p_head_t[:, 1] / self.size[0]) * H
        # coord_pred[:, 0] = (p_head_t[:, 0] / self.size[1]) * W

        p_head_t, conf_pred = self.splitter.combine_tracks(p_head_t, conf_pred)  # (1, N, 2), (1, N)

        return p_head_t[0], conf_pred[0] > self.confidence_treshold



