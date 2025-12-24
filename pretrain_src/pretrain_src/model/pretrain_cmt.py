from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT, gelu, BertOutAttention
from .ops import pad_tensors_wgrad, gen_seq_masks, extend_neg_masks

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, config, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, config, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class ResidualTransformBlock(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = gelu
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.LayerNorm(x)
        return x
    
class NextActionPrediction(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*2),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size*2, eps=config.layer_norm_eps),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size*2, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'sap' in config.pretrain_tasks:
            self.graph_query_text = BertOutAttention(config)
            self.graph_attentioned_txt_embeds_transform = ResidualTransformBlock(self.config, self.config.hidden_size, self.config.hidden_dropout_prob)
            self.global_sap_head = NextActionPrediction(self.config, self.config.hidden_size, self.config.pred_head_dropout_prob)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['txt_task_encoding'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_task_embeddings'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['txt_task_encoding'], batch['traj_view_img_fts'], batch['traj_view_dep_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_task_embeddings'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        txt_labels, compute_loss
    ):
        txt_embeds, _ = self.bert(
            txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        )

        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden) 
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1)) 
        return hidden_masked

    def forward_sap(
        self, txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        gmap_visited_masks, global_act_labels, local_act_labels, compute_loss
    ):
        batch_size = txt_ids.size(0)
        txt_embeds, gmap_embeds = self.bert(
            txt_ids, txt_lens, txt_task_encoding, traj_view_img_fts, traj_view_dep_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_task_embeddings, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        )

        txt_masks = gen_seq_masks(txt_lens)
        extended_txt_masks = extend_neg_masks(txt_masks)
        graph_attentioned_txt_embeds, _ = self.graph_query_text(gmap_embeds, txt_embeds, attention_mask=extended_txt_masks) 
        graph_attentioned_txt_embeds = self.graph_attentioned_txt_embeds_transform(graph_attentioned_txt_embeds)
        fusion_input = torch.cat([gmap_embeds, graph_attentioned_txt_embeds], dim=-1) 
        global_logits = self.global_sap_head(fusion_input).squeeze(2) 

        global_logits.masked_fill_(gmap_visited_masks, -float('inf')) 
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf')) 

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            losses = global_losses
            return losses
        else:
            return global_logits,  global_act_labels