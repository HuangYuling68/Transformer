import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """
        位置编码，为输入序列中的每个位置添加唯一的位置表示，以引入位置信息。

        参数:
            d_model: 嵌入维度，即每个位置的编码向量的维度。
            dropout: 位置编码后应用的 Dropout 概率。
            max_len: 位置编码的最大长度，适应不同长度的输入序列。
        """
        super(PositionalEncoder, self).__init__()
        self.dropout =nn.Dropout(p=dropout)
        self.d_model =d_model

        # 创建位置编码矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 位置索引 (max_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/ d_model)) # 分母项 (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数

        pe = pe.unsqueeze(0)  # 添加批次维度 (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将pe注册为缓冲区，不会被视为模型参数

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x: 输入序列的嵌入向量，形状为 (batch_size, seq_len, d_model)。

        返回:
            加入位置编码和 Dropout 后的嵌入向量，形状为 (batch_size, seq_len, d_model)。
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)