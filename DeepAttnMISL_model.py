"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""

import torch.nn as nn
import torch

class DeepAttnMIL_Surv(nn.Module):
    """
    Deep AttnMISL Model with clinical feature cross-attention (single direction)
    """

    def __init__(self, cluster_num, clinical_dim=27, clinical_emb_dim=64, cross_attn_heads=1, attn_emb_dim=64):
        '''
        cluster_num: int, number of clusters
        clinical_dim: int, dimension of input clinical_param vector
        clinical_emb_dim: int, dimension of clinical embedding
        cross_attn_heads: int, number of heads for cross-attention
        attn_emb_dim: int, WSI feature embedding dim (should match clinical_emb_dim for cross-attn)
        '''
        super(DeepAttnMIL_Surv, self).__init__()
        self.cluster_num = cluster_num
        self.attn_emb_dim = attn_emb_dim

        # WSI patch feature embedding
        self.embedding_net = nn.Sequential(
            nn.Conv2d(2048, attn_emb_dim, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # Attention pooling to get bag-level feature
        self.attention = nn.Sequential(
            nn.Linear(attn_emb_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Clinical参数MLP embedding
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, clinical_emb_dim),
            nn.ReLU()
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=attn_emb_dim, num_heads=cross_attn_heads, batch_first=True)

        # Survival prediction head
        self.fc6 = nn.Sequential(
            nn.Linear(attn_emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )
    
    def masked_softmax(self, x, mask=None, dim=-1):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate. 
        If mask is cluster-level, will be expanded to patch-level inside forward().
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            x = x.masked_fill(~mask.bool(), -1e9)
        return torch.softmax(x, dim=dim)

    def forward(self, x, mask, clinical_param):
        """
        x: list of cluster_num tensors, each is (N_patch_in_cluster, 2048, 1, 1)
        mask: (cluster_num,) tensor or (num_total_patches,) tensor
        clinical_param: (batch, clinical_dim) tensor, batch=1
        """
        # 1. Patch embedding
        res = []
        patches_per_cluster = []
        for i in range(self.cluster_num):
            hh = x[i]  # (N_patch, 2048, 1, 1)
            output = self.embedding_net(hh)  # (N_patch, emb_dim, 1, 1)
            output = output.view(output.size()[0], -1)  # (N_patch, emb_dim)
            res.append(output)
            patches_per_cluster.append(output.size(0))
        h = torch.cat(res, dim=0)  # (num_total_patches, emb_dim)

        # 2. Attention pooling
        b = h.size(0)
        c = h.size(1)
        h = h.view(b, c)
        A = self.attention(h)  # (num_total_patches, 1)
        A = torch.transpose(A, 1, 0)  # 1 x num_total_patches

        # ---- mask扩展逻辑 ----
        # 如果mask只按cluster_num给出，则扩展到patch级别
        if mask.dim() == 1 and mask.shape[0] == self.cluster_num:
            device = mask.device
            mask_patch = []
            for i in range(self.cluster_num):
                mask_patch.append(mask[i].repeat(patches_per_cluster[i]).to(device))
            mask_patch = torch.cat(mask_patch).unsqueeze(0)  # shape: (1, num_total_patches)
        else:
            mask_patch = mask  # 假定已是patch级别

        A = self.masked_softmax(A, mask_patch)

        M = torch.mm(A, h)  # 1 x emb_dim, 即bag-level特征

        # 3. 临床参数MLP
        clinical_emb = self.clinical_mlp(clinical_param)  # (1, emb_dim)

        # 4. 单向 Cross Attention
        # Query: M (1, emb_dim) -> (batch, seq_len=1, emb_dim)
        # Key/Value: clinical_emb (1, emb_dim) -> (batch, seq_len=1, emb_dim)
        query = M.unsqueeze(1)  # (batch=1, seq=1, emb_dim)
        key = clinical_emb.unsqueeze(1)  # (batch=1, seq=1, emb_dim)
        value = clinical_emb.unsqueeze(1)  # (batch=1, seq=1, emb_dim)
        cross_attn_out, _ = self.cross_attn(query, key, value)  # (1, 1, emb_dim)
        fused = cross_attn_out.squeeze(1)  # (1, emb_dim)

        # 5. Survival prediction
        Y_pred = self.fc6(fused)  # (1, 1)
        return Y_pred
