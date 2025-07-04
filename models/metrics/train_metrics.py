import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanMetric

from models.topo_loss import compute_topo_loss
from models.metrics.abstract_metrics import CrossEntropyMetric


class TrainLoss(nn.Module):
    def __init__(self, lambda_train, cfg=None):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.degree_loss = MeanMetric()
        self.y_loss = CrossEntropyMetric()
        self.train_pos_mse = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)
        self.lambda_train = lambda_train

        self.use_deg_loss = cfg.model.get("deg_loss", False)
        self.use_topo = cfg.model.get("use_topo", False)

    def forward(self, masked_pred, masked_true, log, epoch):
        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_pos = masked_true.pos[node_mask]
        masked_pred_pos = masked_pred.pos[node_mask]

        true_X = masked_true.X[node_mask]
        masked_pred_X = masked_pred.X[node_mask]

        true_charges = masked_true.charges[node_mask]
        masked_pred_charges = masked_pred.charges[node_mask]

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]
        true_E = masked_true.E[edge_mask]

        # loss_topo = 0.0
        # if self.use_topo:
        #     pred_labels = masked_pred.E.argmax(dim=-1)
        #     true_labels = masked_true.E.argmax(dim=-1)
        #     pred_adj = (pred_labels != 0).float()
        #     true_adj = (true_labels != 0).float()
        #     total = 0.0
        #     for i in range(bs):
        #         loss0 = compute_topo_loss(pred_adj[i], true_adj[i], dim=0)
        #         loss1 = compute_topo_loss(pred_adj[i], true_adj[i], dim=1)
        #         total += (loss0 + loss1)
        #     loss_topo = total / bs

        assert (true_X != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()
        loss_pos = self.train_pos_mse(masked_pred_pos, true_pos)
        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E)
        loss_deg = 0
        if self.use_deg_loss:
            loss_deg = self.node_degree_loss(pred=masked_pred, true=masked_true, node_mask=node_mask,
                                             edge_mask=edge_mask)
        self.degree_loss.update(loss_deg)
        loss_y = self.y_loss(masked_pred.y, masked_true.y)
        batch_loss = (self.lambda_train[0] * loss_pos +
                      self.lambda_train[1] * loss_X +
                      self.lambda_train[2] * loss_charges +
                      self.lambda_train[5] * loss_deg +
                      self.lambda_train[3] * loss_E +
                      self.lambda_train[4] * loss_y
                      # self.lambda_train[5] * loss_topo
                      )

        to_log = {
            "train_loss/pos_mse": self.lambda_train[0] * self.train_pos_mse.compute() if true_X.numel() > 0 else -1,
            "train_loss/X_CE": self.lambda_train[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
            "train_loss/charges_CE": self.lambda_train[
                                         2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
            "train_loss/deg_kl": self.lambda_train[5] * loss_deg if true_X.numel() > 0 else -1,
            "train_loss/E_CE": self.lambda_train[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
            "train_loss/y_CE": self.lambda_train[4] * self.y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
            # "train_loss/topo_loss": self.lambda_train[5] * loss_topo if self.use_topo else -1.0,
            "train_loss/batch_loss": batch_loss.item()} if log else None

        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.charges_loss, self.edge_loss, self.y_loss,
                       self.train_pos_mse, self.degree_loss]:
            metric.reset()

    def node_degree_loss(self, pred, true, node_mask, edge_mask, max_node_deg=15):
        gt_adj = true.E
        pred_adj = self.get_adjacency(placeholder=pred)  # B, N, N, C
        gt_adj = gt_adj[..., 1:].sum(dim=-1)  # BxNxN
        pred_adj = pred_adj[..., 1:].sum(dim=-1)  # BxNxN
        gt_adj = gt_adj * edge_mask
        pred_adj = pred_adj * edge_mask
        gt_degrees = gt_adj.sum(dim=2)  # BxN
        pred_degrees = pred_adj.sum(dim=2)
        gt_degrees = gt_degrees[node_mask]  # BxN
        pred_degrees = pred_degrees[node_mask]  # BxN
        gt_degree_prob = self.convert_list_to_hist(gt_degrees, max_node_deg)
        pred_degrees = self.convert_list_to_hist(pred_degrees, max_node_deg)
        return F.kl_div(torch.log(pred_degrees), gt_degree_prob, reduction='batchmean')

    def convert_list_to_hist(self, degrees, max_node_deg):
        hist = torch.zeros(max_node_deg + 1).to(degrees.device)
        degrees = degrees + 1  # [1, max_node_deg]
        for i in range(max_node_deg + 1):
            if i in degrees:
                hist[i] = (degrees == i + 1).sum()
        hist[hist == 0] = 1e-6
        norm_hist = hist / hist.sum()
        return norm_hist

    def get_adjacency(self, placeholder):
        b, n, _, c = placeholder.E.shape
        edges = F.gumbel_softmax(placeholder.E.view(-1, c), hard=True, dim=-1, tau=0.5)  # , tau=0.1 tau=0.01)
        edges = edges.view(b, n, n, c)
        return edges

    def log_epoch_metrics(self):
        epoch_pos_loss = self.train_pos_mse.compute().item() if self.train_pos_mse.total > 0 else -1.0
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_y_loss = self.y_loss.compute().item() if self.y_loss.total_samples > 0 else -1.0
        epoch_deg_loss = self.degree_loss.compute().item()

        to_log = {
            "train_epoch/pos_mse": epoch_pos_loss,
            "train_epoch/degree_kl": epoch_deg_loss,
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/charges_CE": epoch_charges_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss}
        return to_log


class ValLoss(nn.Module):
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.val_y_loss = CrossEntropyMetric()
        self.lambda_val = lambda_train

    def forward(self, masked_pred, masked_true, log: bool):
        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_X = masked_true.X[node_mask]  # q x 4
        masked_pred_X = masked_pred.X[node_mask]  # q x 4

        true_charges = masked_true.charges[node_mask]  # q x 3
        masked_pred_charges = masked_pred.charges[node_mask]

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]  # r x num_categ
        true_E = masked_true.E[edge_mask]  # r x num_categ

        # Check that the masking is correct
        assert (true_X != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()

        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.val_y_loss(masked_pred.y, masked_true.y) if masked_true.y.numel() > 0 else 0.0

        batch_loss = (self.lambda_val[1] * loss_X +
                      self.lambda_val[2] * loss_charges +
                      self.lambda_val[3] * loss_E +
                      self.lambda_val[4] * loss_y)

        to_log = {
            "val_loss/X_CE": self.lambda_val[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
            "val_loss/charges_CE": self.lambda_val[
                                       2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
            "val_loss/E_CE": self.lambda_val[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
            "val_loss/y_CE": self.lambda_val[4] * self.val_y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
            "val_loss/batch_loss": batch_loss.item()} if log else None

        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.val_y_loss, self.charges_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss > 0 else -1.0
        epoch_y_loss = self.val_y_loss.compute().item() if self.val_y_loss.total_samples > 0 else -1.0

        to_log = {
            "val_epoch/x_CE": epoch_node_loss,
            "val_epoch/E_CE": epoch_edge_loss,
            "val_epoch/charges_CE": epoch_charges_loss,
            "val_epoch/y_CE": epoch_y_loss}
        return to_log


class TrainMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()

    def forward(self, masked_pred, masked_true, log: bool):
        return None
        self.train_atom_metrics(masked_pred.X, masked_true.X)
        self.train_bond_metrics(masked_pred.E, masked_true.E)
        if not log:
            return

        to_log = {}
        for key, val in self.train_atom_metrics.compute().items():
            to_log['train/' + key] = val.item()
        for key, val in self.train_bond_metrics.compute().items():
            to_log['train/' + key] = val.item()
        return to_log

    def reset(self):
        return
