import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceWhitening(nn.Module):

    def __init__(self, dim):    
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C = f_map.shape  # f_map: (B, C)

    if eye is None:
        eye = torch.eye(C, device=f_map.device)

    # 中心化：减去每个特征的均值（按列）
    f_map_centered = f_map - f_map.mean(dim=0, keepdim=True)  # shape: (B, C)

    # 计算协方差矩阵 (C x C)
    cov_matrix = (f_map_centered.T @ f_map_centered) / (B - 1)  # shape: (C, C)

    # 加上 epsilon * I 保持数值稳定
    cov_matrix = cov_matrix + eps * eye

    return cov_matrix, B

# Calcualate Cross Covarianc of two feature maps
# reference : https://github.com/shachoi/RobustNet
def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape
    B, C = f_map1.shape

    if eye is None:
        eye = torch.eye(C, device=f_map1.device)

    # 中心化（去均值）
    f_map1_centered = f_map1 - f_map1.mean(dim=0, keepdim=True)  # (B, C)
    f_map2_centered = f_map2 - f_map2.mean(dim=0, keepdim=True)  # (B, C)

    # 计算跨协方差矩阵：C × C
    cross_cov = (f_map1_centered.T @ f_map2_centered) / (B - 1)

    # 加上数值稳定项
    cross_cov = cross_cov + eps * eye

    return cross_cov, B

def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape
    B, C = k_feat.shape

    # 中心化
    k_centered = k_feat - k_feat.mean(dim=0, keepdim=True)
    q_centered = q_feat - q_feat.mean(dim=0, keepdim=True)

    # 计算协方差矩阵
    cross_cov = (k_centered.T @ q_centered) / (B - 1)

    diag = torch.diagonal(cross_cov)
    eye = torch.ones_like(diag)
    on_diag_loss = F.mse_loss(diag, eye)

    off_diag = cross_cov - torch.diag_embed(diag)
    off_diag_loss = (off_diag ** 2).sum() / (C * (C - 1))

    return on_diag_loss + 0.005 * off_diag_loss
