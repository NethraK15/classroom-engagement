import math
import torch
import torch.nn as nn
import config_binary_transformer as config

class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding for Temporal Transformer.
    Injects frame sequence order into embeddings.
    """
    def __init__(self, d_model, max_len=config.WINDOW_SIZE, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BinaryEngagementTransformer(nn.Module):
    """
    Binary Temporal Transformer Classifier: Engaged vs Disengaged (2 classes).

    Architecture:
      1. Linear Frame Projection: input_dim -> d_model
      2. Sinusoidal Positional Encoding
      3. Transformer Encoder Stack (Multi-Head Self-Attention)
      4. Masked Temporal Mean Pooling (handles variable-length windows)
      5. MLP Classification Head -> 2 output logits (Binary)

    Sliding-window inference:
      During training, each sample is a fixed WINDOW_SIZE window drawn by
      the sliding window preprocessor. At inference, windows from the same
      video are averaged (majority-vote aggregation) to produce one label
      per video — ensuring the ENTIRE video is seen, not just the first frames.
    """
    def __init__(
        self,
        input_dim,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ):
        super(BinaryEngagementTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Linear Projection (Input_Dim -> d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 2. Sinusoidal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # 3. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 4. MLP Classification Head (Binary)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, num_classes)
        )

    def forward(self, x, seq_lens=None):
        """
        Args:
            x        : [Batch, Window_Size, Input_Dim]
            seq_lens : Optional actual lengths (for padded sequences)
        Returns:
            logits   : [Batch, 2]  (Engaged / Disengaged)
        """
        # Squeeze any unintended singleton dimensions (e.g. [B, T, 1, D] -> [B, T, D])
        if x.dim() == 4:
            if x.shape[2] == 1:
                x = x.squeeze(2)
            elif x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0).squeeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # [1, T, D] if batch dimension was omitted

        batch_size, max_len, _ = x.shape
        device = x.device

        # 1. Project input frames
        x_proj = self.input_proj(x)  # [B, T, d_model]

        # 2. Add Positional Encoding
        x_pos = self.pos_encoder(x_proj)

        # 3. Padding mask (True = padded/ignored position)
        src_key_padding_mask = None
        if seq_lens is not None:
            lengths_tensor = (
                seq_lens.to(device)
                if isinstance(seq_lens, torch.Tensor)
                else torch.tensor(seq_lens, device=device)
            )
            src_key_padding_mask = (
                torch.arange(max_len, device=device).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
            )

        # 4. Transformer Encoder
        enc_out = self.transformer_encoder(x_pos, src_key_padding_mask=src_key_padding_mask)

        # 5. Masked Temporal Mean Pooling
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
            pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = enc_out.mean(dim=1)

        # 6. Binary Classification Head
        logits = self.classifier(pooled)
        return logits
