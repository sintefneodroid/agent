#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08/12/2019
           """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["MultiHeadAttention"]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # (I)
        query = query.refine_names(..., "T", "D")
        self_attn = key is None and value is None
        if self_attn:
            mask = mask.refine_names(..., "T")
        else:
            mask = mask.refine_names(..., "T", "T_key")  # enc attn

        dim = query.size("D")
        assert (
            dim == self.dim
        ), f"Dimensions do not match: {dim} query vs {self.dim} configured"
        assert mask is not None, "Mask is None, please specify a mask"
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        # (II)
        def prepare_head(tensor):
            tensor = tensor.refine_names(..., "T", "D")
            return tensor.unflatten(
                "D", [("H", n_heads), ("D_head", dim_per_head)]
            ).align_to(..., "H", "T", "D_head")

        assert value is None
        if self_attn:
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            key = key.refine_names(..., "T", "D")
            value = key
        dim = key.size("D")

        # Distinguish between query_len (T) and key_len (T_key) dims.
        k = prepare_head(self.k_lin(key)).rename(T="T_key")
        v = prepare_head(self.v_lin(value)).rename(T="T_key")
        q = prepare_head(self.q_lin(query))

        dot_prod = q.div_(scale).matmul(k.align_to(..., "D_head", "T_key"))
        dot_prod.refine_names(..., "H", "T", "T_key")  # just a check

        # (III)
        attn_mask = (mask == 0).align_as(dot_prod)
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = self.attn_dropout(F.softmax(dot_prod / scale, dim="T_key"))

        # (IV)
        attentioned = (
            attn_weights.matmul(v)
            .refine_names(..., "H", "T", "D_head")
            .align_to(..., "T", "H", "D_head")
            .flatten(["H", "D_head"], "D")
        )

        return self.out_lin(attentioned).refine_names(..., "T", "D")


if __name__ == "__main__":
    n, t, d, h = 7, 5, 2 * 3, 3
    query = torch.randn(n, t, d, names=("N", "T", "D"))
    mask = torch.ones(n, t, names=("N", "T"))
    attn = MultiHeadAttention(h, d)
    output = attn(query, mask=mask)
    # works as expected!
    print(output.names)

    query = torch.randn(t, d, names=("T", "D"))
    mask = torch.ones(t, names=("T",))
    output = attn(query, mask=mask)
    print(output.names)
