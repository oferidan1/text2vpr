import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class GenericPairLoss(BaseMetricLossFunction):
    def __init__(self, mat_based_loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels, embeds2, w):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        w_i = w[:,0].unsqueeze(1)
        w_t = w[:,1].unsqueeze(1)   
        # calc text sim in the batch
        text_sim = torch.matmul(embeds2, embeds2.T)
        img_sim = torch.matmul(embeddings, embeddings.T)
        # w_i_ij = torch.zeros_like(img_sim)
        # w_t_ij = torch.zeros_like(text_sim)
        # batch_size = embeddings.shape[0]       
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         w_i_ij[i, j] = (w_i[i] + w_i[j]) / 2.0
        #         w_t_ij[i, j] = (w_t[i] + w_t[j]) / 2.0
        w_i_ij = ((w_i.unsqueeze(1) + w_i.unsqueeze(0)) / 2.0).squeeze(-1)
        w_t_ij = ((w_t.unsqueeze(1) + w_t.unsqueeze(0)) / 2.0).squeeze(-1)

        # calculate dynamic weights
        s_ij = w_i_ij * img_sim + w_t_ij * text_sim
        return self.loss_method(mat, indices_tuple, s_ij)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, indices_tuple, s_ij):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        self._assert_either_pos_or_neg(pos_mask, neg_mask)
        return self._compute_loss(mat, pos_mask, neg_mask, s_ij)

    def pair_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)

    @staticmethod
    def _assert_either_pos_or_neg(pos_mask, neg_mask):
        assert not torch.any(
            (pos_mask != 0) & (neg_mask != 0)
        ), "Each pair should be either be positive or negative"
