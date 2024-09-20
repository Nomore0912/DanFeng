#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


def kabsch(p, q):
    #  使用 Kabsch 算法找到两个点集P和Q之间的最佳旋转矩阵
    c = torch.mm(p.T, q)
    v, s, w = torch.svd(c)
    d = (torch.det(v) * torch.det(w)) < 0.0
    if d:
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    r = torch.mm(v, w.T)
    return r


def align_point(p, q):
    # 计算两个点集的质心
    centroid_p = torch.mean(p, dim=0)
    centroid_q = torch.mean(q, dim=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    r = kabsch(p_centered, q_centered)

    p_aligned = torch.mm(p_centered, r)

    diff = p_aligned - q_centered
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))

    return p_aligned + centroid_q, rmsd


if __name__ == '__main__':
    P = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    Q = torch.tensor([
        [2.0, 3.5, 4.5],
        [5.0, 6.1, 7.1],
        [8.0, 9.1, 10.1]
    ])

    # 对齐点集
    P_aligned, rmsd = align_point(P, Q)

    print("对齐后的P点集:")
    print(P_aligned)
    print(f"RMSD: {rmsd}")
