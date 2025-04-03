import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from trackon.track_on import TrackOn

from utils.coord_utils import get_points_on_a_grid
from utils.coord_utils import indices_to_coords


class TrackOnFF(TrackOn):
    def __init__(self, args):
        super().__init__(args=args)

        self.extend_queries = False
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

    def init_queries_and_memory(self, queries, frame):
        # :args queries: (N, 2)         (x, y) in given frame
        # :args frame: (1, C, H, W)     frame to extract features from

        self.query_times = queries[:, 0]     # (N)
        queries = queries[:,1:]  # (N, 2)
        self.N = queries.size(0)
        H, W = frame.shape[-2], frame.shape[-1]

        self.device = queries.device


        # === Set Queries ===
        queries[:, 1] = (queries[:, 1] / H) * self.size[0]
        queries[:, 0] = (queries[:, 0] / W) * self.size[1]



        if self.extend_queries:
            K = 20
            extra_queries = get_points_on_a_grid(K, self.size, self.device)           # (1, K ** 2, 2)
            queries = torch.cat([queries, extra_queries[0]], dim=0)              # (N + K ** 2, 2)
            self.query_times = torch.cat([self.query_times, torch.zeros(extra_queries.shape[1], self.device)], dim=0)   # (N + K ** 2)
        self.queries = queries

        # === === ===

        # === Sample Queries ===
        pos_query = queries.unsqueeze(0)  # (1, N_prime, 2) = (1, N + K ** 2, 2)
        N_prime = pos_query.size(1)

        x, y = pos_query[:, :, 0], pos_query[:, :, 1]   # (B, N')
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1
        grid = torch.stack([x_grid, y_grid], dim=-1).view(N_prime, 1, 1, 2).to(self.device)  # (N_prime, 1, 1, 2)

        video_frame = frame / 255.0          # (1, C, H, W), to [0, 1]
        video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=False)
        video_frame = self.backbone.normalization(video_frame)         # to [-1, 1]

        f4, f8, _, _ = self.backbone.vit_encoder(video_frame)            # (1, 384, H4, W4)
        if self.stride == 4:
            f = f4
        elif self.stride == 8:
            f = f8

        H_prime, W_prime = f.shape[-2], f.shape[-1]
        assert H_prime == self.H_prime and W_prime == self.W_prime, f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"

        f = self.backbone.token_projection(f)                                                                    # (1, C, H4, W4)
        f = f + self.backbone.frame_pos_embedding                                                                # (1, C, H4, W4)
        C = f.shape[1]

        f = f.expand(N_prime, -1, -1, -1).reshape(N_prime, C, H_prime, W_prime)                         # (N_prime, C, H4, W4)
        sampled = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False)   # (N_prime, C, 1, 1)
        q_init = sampled.reshape(1, N_prime, C)                                                         # (1, N_prime, C)

        self.q_init = q_init

        # === === ===
        # === Init memories ===
        max_memory_size = max([self.query_decoder.memory_size, self.sm_query_updater.memory_size])
        query_num = q_init.shape[1]

        # Spatial Memory
        self.spatial_memory = torch.zeros(1, query_num, max_memory_size, C, device=self.device)             # (1, N, max_memory_size, C)

        # Context Memory
        self.context_memory = torch.zeros(1, query_num, max_memory_size, C, device=self.device)             # (1, N, max_memory_size, C)

        # Masking
        self.past_occ = torch.ones(1, query_num, max_memory_size, device=self.device, dtype=torch.bool)     # (1, N, max_memory_size)
        self.past_mask = torch.ones(1, query_num, max_memory_size, device=self.device, dtype=torch.bool)
        # === === ===

    def sample_queries(self, new_queries):
        N_new = new_queries.shape[0]
        C = self.prev_ft.shape[1]
        pos_query = new_queries.unsqueeze(0)           # (1, N_new, 2) 

        x, y = pos_query[:, :, 0], pos_query[:, :, 1]  # (1, N_new)
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1
        grid = torch.stack([x_grid, y_grid], dim=-1).view(N_new, 1, 1, 2).to(self.device)                         # (N_new, 1, 1, 2)

        f_t = self.prev_ft.expand(N_new, -1, -1, -1).reshape(N_new, C, self.H_prime, self.W_prime)       # (N_new, C, H4, W4)
        sampled = F.grid_sample(f_t, grid, mode='bilinear', padding_mode='border', align_corners=False)  # (N_new, C, 1, 1)
        q_init_new = sampled.reshape(1, N_new, C)                                                                 # (1, N_new, C)
        self.q_init = torch.cat([self.q_init, q_init_new], dim=1)

    def update_queries_and_memory(self, queries, frame, removed_indices):
        """Function called at all timesteps t>0 to remove rejected queries and add new queries
        Args:
            queries (Tensor): Contains old queries that are still being tracked and new queries
                added to the previous frame. Shape (N,3) - Each query is of the format [t, x, y]
            frame (Tensor):   The latest image frame. Shape (1, C, H, W)
            removed_indices (List(int)): Indices of the queries that were removed, so that the
                removal can be reflected on the memory
        """
        self.query_times = queries[:, 0]     # (N)
        if self.t == 1:
            N_new = 0
        else:
            N_new = (self.query_times == self.t - 1).sum().item()
        if N_new == 0 and len(removed_indices) == 0:
            return

        # Rescale queries
        queries = queries[:,1:]              # (N, 2)
        self.N = queries.size(0)
        H, W = frame.shape[-2], frame.shape[-1]
        queries[:, 1] = (queries[:, 1] / H) * self.size[0]
        queries[:, 0] = (queries[:, 0] / W) * self.size[1]
        self.queries = queries

        # Remove rejected queries and memory
        N_prev = self.q_init.shape[1]
        removed_mask = torch.ones(N_prev, device=queries.device, dtype=torch.bool)
        removed_mask[removed_indices] = False
        spatial_memory = self.prev_spatial_memory[:, removed_mask, :, :]  # (1, N", max_memory_size, C)
        context_memory = self.prev_context_memory[:, removed_mask, :, :]  # (1, N", max_memory_size, C)
        past_occ = self.prev_past_occ[:, removed_mask, :]              # (1, N", max_memory_size)
        past_mask = self.prev_past_mask[:, removed_mask, :]            # (1, N", max_memory_size)
        p_head_t = self.prev_p[removed_mask]                              # (N", 2)
        prev_v = self.prev_v[removed_mask]                                # (N")
        self.q_init = self.q_init[:, removed_mask, :]                     # (1, N", C)

        # Sample new query tokens from current frame
        # new_queries = queries[queries[:, 0] == self.t - 1]
        # N_new = new_queries.shape[0]
        assert N_new == self.N - self.q_init.shape[1]
        max_memory_size = self.spatial_memory.shape[2]
        C = self.prev_ft.shape[1]

        # Update memory for new queries
        if N_new > 0:
            new_queries = queries[-N_new:]
            self.sample_queries(new_queries)
            spatial_memory = torch.cat([spatial_memory, torch.zeros(1, N_new, max_memory_size, C, device=self.device)], dim=1)
            context_memory = torch.cat([context_memory, torch.zeros(1, N_new, max_memory_size, C, device=self.device)], dim=1)
            past_mask = torch.cat([past_mask, torch.ones(1, N_new, max_memory_size, device=self.device, dtype=torch.bool)], dim=1)
            past_occ = torch.cat([past_occ, torch.ones(1, N_new, max_memory_size, device=self.device, dtype=torch.bool)], dim=1)
            p_head_t = torch.cat([p_head_t, self.queries[-N_new:]], dim=0)  
            prev_v = torch.cat([prev_v, torch.ones(N_new, device=queries.device, dtype=torch.bool)], dim=0)

        # Update latest memory
        # self.spatial_memory[:, -N_new:, -1, :] = q_init_new   # Under the assumption that I can just put q_init in the memory and dopnt have to make it q_t
        # self.context_memory[:, -N_new:, -1, :] = q_init_new
        # self.past_mask[:, -N_new:, -1] = False
        # self.past_occ[:, -N_new:, -1] = False

        # fake ff_forward to recompute memory with replaced queries
        f_t = self.prev_ft.permute(0, 2, 3, 1)                           # (B, H4, W4, C)
        f_t = f_t.reshape(f_t.shape[0], -1, C)                       # (B, P, C)
        h_t = self.feature_decoder(f_t)

        q_init = self.q_init
        t= self.t - 1  # Under the assumption that the below calculations are being redone with the new queries that were sampled in the PREVIOUS frame
        query_times = self.query_times.unsqueeze(0)  # (1, N_prime)
        queried_now_or_before = (query_times <= t)
        assert queried_now_or_before.all().item() == True, "Future points have been queried!"
        q_init_t = self.sm_query_updater(q_init,
                                        spatial_memory,
                                        past_mask,
                                        past_occ,
                                        query_times,
                                        t)   

        q_t = self.query_decoder(q_init_t, f_t, context_memory.clone(), past_mask, queried_now_or_before)       # (1, N, C)
        q_t = self.projection1(q_t)                                                                             # (1, N, C)

        c1_t = self.correlation(q_t, h_t)
        q_t, _, _ = self.rerank_module(q_t, h_t, c1_t) 

        q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t.unsqueeze(0), query_times, t)   # (1, N, C)
        self.spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)           # (1, N, max_memory_size, C)
        self.context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)             # (1, N, max_memory_size, C)
        self.past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)  # (1, N, memory_size)
        self.past_occ = torch.cat([past_occ[:, :, 1:], prev_v.reshape(1, -1, 1)], dim=2)  # (1, N, memory_size)

    def ff_forward(self, frame):
        # :args frame: (1, C, H, W)     frame to extract features from
        H, W = frame.shape[-2], frame.shape[-1]

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
        query_times = self.query_times.unsqueeze(0)  # (1, N_prime)
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
        f_t = self.backbone.encode_frames_online(frame)           # (1, P, C)
        self.prev_ft = f_t
        C = f_t.shape[1]
        f_t = f_t.permute(0, 2, 3, 1)                           # (B, H4, W4, C)
        f_t = f_t.reshape(f_t.shape[0], -1, C)                       # (B, P, C)
        h_t = self.feature_decoder(f_t)                           # (1, P, C)
        # ##### ##### #####

        # ##### Query Decoder #####
        q_t = self.query_decoder(q_init_t, f_t, context_memory.clone(), past_mask, queried_now_or_before)       # (1, N, C)
        q_t = self.projection1(q_t)                                                                             # (1, N, C)
        # ##### ##### #####

        # ##### Correlation - 1 #####
        c1_t = self.correlation(q_t, h_t)                                                         # (1, N, P)
        # ##### ##### #####

        # ##### Reranking #####
        q_t, top_k_u_logit, top_k_p = self.rerank_module(q_t, h_t, c1_t)                           # (1, N, C), (1, N, K), (1, N, K, 2) 
        q_t_corr = self.projection2(q_t)                                                           # (1, N, C)
        # ##### ##### #####

        # ##### Correlation - 2 #####
        c2_t = self.correlation(q_t_corr, h_t)                                                                     # (1, N, P)
        p_head_patch_t = indices_to_coords(torch.argmax(c2_t, dim=-1).unsqueeze(1), self.size, self.stride)[:, 0]  # (1, N, 2)
        # ##### ##### #####

        # ##### Offset Prediction #####
        o_t = self.offset_head(q_t_corr, h_t, p_head_patch_t)           # (1, #offset_layers, N, 2)
        p_head_t = p_head_patch_t + o_t[:, -1]       # (1, N, 2)
        # ##### ##### #####

        # ##### Visibility and Uncertainty Prediction #####
        v_t_logit, u_t_logit = self.visibility_head(q_t, h_t, p_head_t)  # (1, N), (1, N)
        # ##### ##### #####

        # ##### Memory Update #####
        # Spatial Memory Update
        q_aug = self.sm_query_updater.get_augmented_memory(q_init, q_t, f_t, p_head_t, query_times, t)   # (1, N, C)
        spatial_memory = torch.cat([spatial_memory[:, :, 1:], q_aug.unsqueeze(2)], dim=2)                # (1, N, max_memory_size, C)

        # Context Memory Update
        context_memory = torch.cat([context_memory[:, :, 1:], q_t.unsqueeze(2)], dim=2)                                     # (1, N, max_memory_size, C)

        # Masking Update
        past_mask = torch.cat([past_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2)                           # (1, N, memory_size)
        past_occ = torch.cat([past_occ[:, :, 1:], (F.sigmoid(v_t_logit) < self.visibility_treshold).unsqueeze(-1)], dim=2)  # (1, N, memory_size)
        # ##### ##### #####

        # Update Variables
        self.spatial_memory = spatial_memory
        self.context_memory = context_memory
        self.past_occ = past_occ
        self.past_mask = past_mask
        self.t += 1

        # Return Pred and Visü
        if self.extend_queries:
            coord_pred = p_head_t[0, :self.N]               # (N, 2)
            vis_pred = F.sigmoid(v_t_logit)[0, :self.N] > self.visibility_treshold     # (N)
            conf_pred = 1 - F.sigmoid(u_t_logit)[0, :self.N] > self.confidence_treshold     # (N)
        else:
            coord_pred = p_head_t[0]
            vis_pred = F.sigmoid(v_t_logit)[0] > self.visibility_treshold
            conf_pred = 1 - F.sigmoid(u_t_logit)[0] > self.confidence_treshold

        self.prev_p = coord_pred.clone()
        self.prev_v = vis_pred.clone()

        coord_pred[:, 1] = (coord_pred[:, 1] / self.size[0]) * H
        coord_pred[:, 0] = (coord_pred[:, 0] / self.size[1]) * W

        return coord_pred, vis_pred, conf_pred



