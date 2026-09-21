import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv,GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index):
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算节点度数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 执行消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 消息传递（邻居信息聚合）
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 直接返回聚合后的输出
        return aggr_out




# ============================================================
# CMDL / NCE-CLUB critics for MI-based disentanglement
# ============================================================
def _cmdl_mlp(dim, hidden_dim, output_dim, layers, activation):
    activation_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    seq = [nn.Linear(dim, hidden_dim), activation_cls()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation_cls()]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


class InfoNCECritic(nn.Module):
    """Negative InfoNCE lower-bound objective used by CMDL."""
    def __init__(self, A_dim, B_dim, hidden_dim=1, layers=1, activation="relu"):
        super().__init__()
        self._f = _cmdl_mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        if sample_size < 2:
            return x_samples.new_zeros(())
        x_tile = x_samples.unsqueeze(0).repeat(sample_size, 1, 1)
        y_tile = y_samples.unsqueeze(1).repeat(1, sample_size, 1)
        T0 = self._f(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim=-1))
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - math.log(sample_size))
        return -lower_bound


class CLUBInfoNCECritic(nn.Module):
    """NCE-CLUB upper-bound critic used by CMDL."""
    def __init__(self, A_dim, B_dim, hidden_dim=1, layers=1, activation="relu"):
        super().__init__()
        self._f = _cmdl_mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        if sample_size < 2:
            return x_samples.new_zeros(())
        x_tile = x_samples.unsqueeze(0).repeat(sample_size, 1, 1)
        y_tile = y_samples.unsqueeze(1).repeat(1, sample_size, 1)
        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))
        return T0.mean() - T1.mean()

    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        if sample_size < 2:
            return x_samples.new_zeros(())
        x_tile = x_samples.unsqueeze(0).repeat(sample_size, 1, 1)
        y_tile = y_samples.unsqueeze(1).repeat(1, sample_size, 1)
        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - math.log(sample_size))
        return -lower_bound


class CMDLSharedPrivate(nn.Module):
    """CMDL-style text/image disentanglement without MI regularization.

    This module implements the representation split used by CMDL:
      1) learn one modality score for image and text with a shared query MLP;
      2) softmax the two scores to obtain modality weights;
      3) shared = w_image * image + w_text * text;
      4) private_image = image - shared;
         private_text  = text  - shared.

    SP-v1 deliberately does NOT add mutual-information regularization yet.
    The three factors are projected back to the original semantic width by
    one local linear layer; the downstream KG/collaborative fusion code is
    unchanged.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.query_common = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        # Minimal projection back to the original semantic width so the
        # downstream KG/collaborative fusion remains unchanged.
        self.factor_fuse = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, text_embeds, image_embeds):
        # Same ordering as the official CMDL implementation: image, text.
        att_common = torch.cat(
            [self.query_common(image_embeds), self.query_common(text_embeds)],
            dim=-1,
        )
        weights = F.softmax(att_common, dim=-1)

        shared = (
            weights[:, 0].unsqueeze(-1) * image_embeds
            + weights[:, 1].unsqueeze(-1) * text_embeds
        )
        image_private = image_embeds - shared
        text_private = text_embeds - shared

        # Only this local Text/Image block changes. The output keeps the
        # exact dimensionality expected by the original MSCRS fusion.
        semantic = self.factor_fuse(
            torch.cat([shared, text_private, image_private], dim=-1)
        )

        return semantic, shared, text_private, image_private, weights


class MMPrompt_inspired(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None, use_sp=False, use_mi=False, mi_sample_size=128,
    ):
        super(MMPrompt_inspired, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.use_sp = bool(use_sp)
        self.use_mi = bool(use_mi and use_sp)
        self.mi_sample_size = int(mi_sample_size)
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        entity_hidden_size = hidden_size // 2
        self.sp_disentangler = CMDLSharedPrivate(entity_hidden_size) if self.use_sp else None
        if self.use_mi:
            critic_hidden_dim = 1
            critic_layers = 1
            self.club_zx_zs = CLUBInfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_zy_zs = CLUBInfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.infonce_x_zs = InfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.infonce_y_zs = InfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_x_ys_cond = CLUBInfoNCECritic(entity_hidden_size * 2, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_y_xs_cond = CLUBInfoNCECritic(entity_hidden_size * 2, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def _cmdl_mi_sample(self):
        required = ["sp_text_input", "sp_image_input", "sp_shared", "sp_text_private", "sp_image_private"]
        for name in required:
            if not hasattr(self, name):
                raise RuntimeError("MI representations are not ready; call get_entity_embeds() first.")
        n = self.sp_shared.shape[0]
        sample_size = min(self.mi_sample_size, n)
        if sample_size < 2:
            idx = torch.arange(n, device=self.sp_shared.device)
        else:
            idx = torch.randperm(n, device=self.sp_shared.device)[:sample_size]
        return (
            self.sp_text_input[idx], self.sp_image_input[idx], self.sp_shared[idx],
            self.sp_text_private[idx], self.sp_image_private[idx],
        )

    def compute_mi_loss(self):
        if not self.use_mi:
            return self.node_embeds.new_zeros(())
        text_input, image_input, shared, text_private, image_private = self._cmdl_mi_sample()
        with torch.autocast(device_type=shared.device.type, enabled=False):
            text_input = text_input.float(); image_input = image_input.float(); shared = shared.float()
            text_private = text_private.float(); image_private = image_private.float()
            high_zx_zs = self.club_zx_zs(text_private, shared)
            high_zy_zs = self.club_zy_zs(image_private, shared)
            low_x_zs = self.infonce_x_zs(text_input, shared)
            low_y_zs = self.infonce_y_zs(image_input, shared)
            cond_high_x_ys = self.club_x_ys_cond(torch.cat([shared, image_input], dim=1), text_input)
            cond_high_y_xs = self.club_y_xs_cond(torch.cat([shared, text_input], dim=1), image_input)
            total = high_zx_zs + high_zy_zs + low_x_zs + low_y_zs + cond_high_x_ys + cond_high_y_xs
        return total

    def compute_mi_critic_learning_loss(self):
        if not self.use_mi:
            return self.node_embeds.new_zeros(())
        text_input, image_input, shared, text_private, image_private = self._cmdl_mi_sample()
        text_input = text_input.detach(); image_input = image_input.detach(); shared = shared.detach()
        text_private = text_private.detach(); image_private = image_private.detach()
        with torch.autocast(device_type=shared.device.type, enabled=False):
            text_input = text_input.float(); image_input = image_input.float(); shared = shared.float()
            text_private = text_private.float(); image_private = image_private.float()
            low_zx_zs = self.club_zx_zs.learning_loss(text_private, shared)
            low_zy_zs = self.club_zy_zs.learning_loss(image_private, shared)
            cond_low_x_ys = self.club_x_ys_cond.learning_loss(torch.cat([shared, image_input], dim=1), text_input)
            cond_low_y_xs = self.club_y_xs_cond.learning_loss(torch.cat([shared, text_input], dim=1), image_input)
            return low_zx_zs + low_zy_zs + cond_low_x_ys + cond_low_y_xs

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 +movie_embeds_ts2)/2   

        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1+movie_embeds_is2)/2  
        if self.use_sp:
            self.sp_text_input = movie_embeds_mean_t
            self.sp_image_input = movie_embeds_mean_i
            (
                movie_embeds_mean_t,
                self.sp_shared,
                self.sp_text_private,
                self.sp_image_private,
                self.sp_weights,
            ) = self.sp_disentangler(movie_embeds_mean_t, movie_embeds_mean_i)
        else:
            movie_embeds_mean_t = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        entity_embeds_c1 =self.conv_c1(entity_embeds, self.edge_index_c)
        entity_embeds_c2 =self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 =self.conv_c3(entity_embeds_c2, self.edge_index_c)

        entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3+ entity_embeds) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean_t.to(dtype=entity_embeds.dtype))
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids] 
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)




class MMPrompt(nn.Module):
    def __init__(
    self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
    n_entity, num_relations, num_bases, edge_index, edge_type,
    edge_index_c, edge_index_t_s, edge_index_i_s, idx_to_id,
    n_prefix_rec=None, n_prefix_conv=None, semantic_gate=0.0, use_sp=False, use_mi=False, mi_sample_size=128,
    ):
        super(MMPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.semantic_gate = float(semantic_gate)
        self.use_sp = bool(use_sp)
        self.use_mi = bool(use_mi and use_sp)
        self.mi_sample_size = int(mi_sample_size)
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)


        entity_hidden_size = hidden_size // 2
        self.sp_disentangler = CMDLSharedPrivate(entity_hidden_size) if self.use_sp else None

        if self.use_mi:
            critic_hidden_dim = 1
            critic_layers = 1
            self.club_zx_zs = CLUBInfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_zy_zs = CLUBInfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.infonce_x_zs = InfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.infonce_y_zs = InfoNCECritic(entity_hidden_size, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_x_ys_cond = CLUBInfoNCECritic(entity_hidden_size * 2, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")
            self.club_y_xs_cond = CLUBInfoNCECritic(entity_hidden_size * 2, entity_hidden_size, critic_hidden_dim, critic_layers, activation="relu")

        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN


        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)


    def _cmdl_mi_sample(self):
        required = ["sp_text_input", "sp_image_input", "sp_shared", "sp_text_private", "sp_image_private"]
        for name in required:
            if not hasattr(self, name):
                raise RuntimeError("MI representations are not ready; call get_entity_embeds() first.")
        n = self.sp_shared.shape[0]
        sample_size = min(self.mi_sample_size, n)
        if sample_size < 2:
            idx = torch.arange(n, device=self.sp_shared.device)
        else:
            idx = torch.randperm(n, device=self.sp_shared.device)[:sample_size]
        return (
            self.sp_text_input[idx], self.sp_image_input[idx], self.sp_shared[idx],
            self.sp_text_private[idx], self.sp_image_private[idx],
        )

    def compute_mi_loss(self):
        if not self.use_mi:
            return self.node_embeds.new_zeros(())
        text_input, image_input, shared, text_private, image_private = self._cmdl_mi_sample()
        with torch.autocast(device_type=shared.device.type, enabled=False):
            text_input = text_input.float()
            image_input = image_input.float()
            shared = shared.float()
            text_private = text_private.float()
            image_private = image_private.float()
            high_zx_zs = self.club_zx_zs(text_private, shared)
            high_zy_zs = self.club_zy_zs(image_private, shared)
            low_x_zs = self.infonce_x_zs(text_input, shared)
            low_y_zs = self.infonce_y_zs(image_input, shared)
            cond_high_x_ys = self.club_x_ys_cond(torch.cat([shared, image_input], dim=1), text_input)
            cond_high_y_xs = self.club_y_xs_cond(torch.cat([shared, text_input], dim=1), image_input)
            total = high_zx_zs + high_zy_zs + low_x_zs + low_y_zs + cond_high_x_ys + cond_high_y_xs
        return total

    def compute_mi_critic_learning_loss(self):
        if not self.use_mi:
            return self.node_embeds.new_zeros(())
        text_input, image_input, shared, text_private, image_private = self._cmdl_mi_sample()
        text_input = text_input.detach(); image_input = image_input.detach(); shared = shared.detach()
        text_private = text_private.detach(); image_private = image_private.detach()
        with torch.autocast(device_type=shared.device.type, enabled=False):
            text_input = text_input.float(); image_input = image_input.float(); shared = shared.float()
            text_private = text_private.float(); image_private = image_private.float()
            low_zx_zs = self.club_zx_zs.learning_loss(text_private, shared)
            low_zy_zs = self.club_zy_zs.learning_loss(image_private, shared)
            cond_low_x_ys = self.club_x_ys_cond.learning_loss(torch.cat([shared, image_input], dim=1), text_input)
            cond_low_y_xs = self.club_y_xs_cond.learning_loss(torch.cat([shared, text_input], dim=1), image_input)
            learning_loss = low_zx_zs + low_zy_zs + cond_low_x_ys + cond_low_y_xs
        return learning_loss

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 +movie_embeds_ts2)/2   

        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1+movie_embeds_is2)/2  
        if self.use_sp:
            self.sp_text_input = movie_embeds_mean_t
            self.sp_image_input = movie_embeds_mean_i
            (
                movie_embeds_mean_t,
                self.sp_shared,
                self.sp_text_private,
                self.sp_image_private,
                self.sp_weights,
            ) = self.sp_disentangler(movie_embeds_mean_t, movie_embeds_mean_i)
        else:
            movie_embeds_mean_t = (movie_embeds_mean_t + movie_embeds_mean_i) / 2

        entity_embeds_c1 =self.conv_c1(node_embeds, self.edge_index_c)
        entity_embeds_c2 =self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 =self.conv_c3(entity_embeds_c2, self.edge_index_c)

        entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3+ entity_embeds) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]
        #entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        
        if self.semantic_gate != 0.0:
            semantic_update = self.semantic_gate * movie_embeds_mean_t
            semantic_update = semantic_update.to(dtype=entity_embeds.dtype)
            entity_embeds = entity_embeds.index_add(
                0,
                indices,
                semantic_update
            )

        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids] 
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)







