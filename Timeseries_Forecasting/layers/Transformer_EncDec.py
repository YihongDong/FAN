import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FANLayer import FANLayer

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", exp_setting=0):
        super(EncoderLayer, self).__init__()
        self.exp_setting = exp_setting
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        assert exp_setting in [0, 2, 4]
        if exp_setting == 0:
            self.mlp1 = nn.Linear(d_model, d_ff)
            self.mlp2 = nn.Linear(d_ff, d_model)
        elif exp_setting == 2:
            self.mlp1 = FANLayer(input_dim=d_model, output_dim=d_ff, with_gate=True)
            self.mlp2 = FANLayer(input_dim=d_ff, output_dim=d_model, with_gate=True)
        elif exp_setting == 4:
            self.mlp1 = FANLayer(input_dim=d_model, output_dim=d_ff, with_gate=False)
            self.mlp2 = FANLayer(input_dim=d_ff, output_dim=d_model, with_gate=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        if self.exp_setting == 0:
            y = self.dropout(self.activation(self.mlp1(y)))
        else:
            y = self.dropout(self.mlp1(y))
        y = self.dropout(self.mlp2(y))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", exp_setting=0):
        super(DecoderLayer, self).__init__()
        self.exp_setting = exp_setting
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        assert exp_setting in [0, 2, 4]
        if exp_setting == 0:
            self.mlp1 = nn.Linear(d_model, d_ff)
            self.mlp2 = nn.Linear(d_ff, d_model)
        elif exp_setting == 2:
            self.mlp1 = FANLayer(input_dim=d_model, output_dim=d_ff, with_gate=True)
            self.mlp2 = FANLayer(input_dim=d_ff, output_dim=d_model, with_gate=True)
        elif exp_setting == 4:
            self.mlp1 = FANLayer(input_dim=d_model, output_dim=d_ff, with_gate=False)
            self.mlp2 = FANLayer(input_dim=d_ff, output_dim=d_model, with_gate=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        if self.exp_setting == 0:
            y = self.dropout(self.activation(self.mlp1(y)))
        else:
            y = self.dropout(self.mlp1(y))
        y = self.dropout(self.mlp2(y))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
