import torch
from torch import Tensor, nn
from transformers import BertModel
from torch.nn import functional as F
from abc import abstractmethod

class MultiFusion(nn.Module):
    def __init__(self, eps=5e-4):
        super(MultiFusion, self).__init__()
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("bld,bld->bld", k, v)
        qkv = torch.einsum("bld,bld->bld", q, kv)
        return qkv

    def forward(self, q, k, v):
        B, L, _ = q.shape

        q = q.view(B, L, -1)
        k = k.view(B, L, -1)
        v = v.view(B, L, -1)

        q = self.kernel_method(q)
        k = self.kernel_method(k)

        sink_incoming = 1.0 / (torch.einsum("bld,bd->bl", q + self.eps, k.sum(dim=1) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("bld,bd->bl", k + self.eps, q.sum(dim=1) + self.eps))

        conserved_sink = torch.einsum("bld,bd->bl", q + self.eps,
                                      (k * source_outgoing.unsqueeze(-1)).sum(dim=1) + self.eps)
        conserved_source = torch.einsum("bld,bd->bl", k + self.eps,
                                        (q * sink_incoming.unsqueeze(-1)).sum(dim=1) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)

        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[1]) / float(k.shape[1])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[1])

        x = (self.dot_product(q * sink_incoming.unsqueeze(-1),
                              k,
                              v * source_competition.unsqueeze(-1))
             * sink_allocation.unsqueeze(-1))

        return x


class MSDG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSDG, self).__init__()

        self.dilated_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_features=out_channels)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_features=out_channels)
        )

        self.fusion = MultiFusion()


    def forward(self, x):
        x_dil1 = self.dilated_conv1(x)
        x_dil2 = self.dilated_conv2(x)
        x_dil3 = self.dilated_conv3(x)

        x_dil1 = x_dil1.transpose(1, 2)
        x_dil2 = x_dil2.transpose(1, 2)
        x_dil3 = x_dil3.transpose(1, 2)

        x_fused = self.fusion(x_dil1, x_dil2, x_dil3)
        x_fused = x_fused.transpose(1, 2)

        return x_fused

def silu(x):
    return x * F.sigmoid(x)

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Mamba(nn.Module):
    def __init__(self, d_model: int,
                 n_layer: int = 24,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 chunk_size: int = 64,
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state

        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: Tensor):

        A = -torch.exp(self.A_log)

        zxbcdt = self.in_proj(u)

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )

        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2))

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)

        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3])
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3])
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3])
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4])

        return Y

class _BiMamba(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,
                 n_layer: int = 24,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 chunk_size: int = 64,
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)
        self.mamba_for = Mamba(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)
        self.mamba_back = Mamba(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)
        self.fc_out = nn.Linear(d_model, cout, bias=False)
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass

class BiMamba(_BiMamba):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)
        self.fc_in = torch.nn.Linear(cin, d_model)
        self.fc_out = torch.nn.Linear(d_model, cout)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8))
        _b, _c, _h, _w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)

        x = self.fc_in(x)
        x1 = self.mamba_for(x)
        x2 = self.mamba_back(x.flip(1)).flip(1)

        x = x1 + x2
        x = self.fc_out(x)

        x = x.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)
        x = x[:, :, :h, :w]
        return x


class MDBM_TFBS(nn.Module):

    def __init__(self):
        super(MDBM_TFBS, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = r"./DNABERT"
        self.dnabert = BertModel.from_pretrained(self.model_path)

        self.Dilated_conv = MSDG(in_channels=768, out_channels=64)
        self.convolution_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(64, 16), stride=(1, 1)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_seq = nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 2))
        self.dropout_seq = nn.Dropout(0.2)

        self.convolution_to_shape = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.bimamba = BiMamba(cin=64, cout=128, d_model=64)
        self.convolution_shape = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 16), stride=(1, 1)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_shape = nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 2))
        self.dropout_shape = nn.Dropout(0.2)

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def execute(self, seq, shape):
        with torch.no_grad():
            seq = self.dnabert(seq)[0].to(self.device)
        seq = seq.float()
        seq = seq.transpose(1, 2)
        seq = self.Dilated_conv(seq)
        seq = seq.unsqueeze(1)
        seq = self.convolution_seq(seq)
        seq = self.max_pooling_seq(seq)
        seq = self.dropout_seq(seq)

        shape = shape.float()
        shape = shape.unsqueeze(1)
        shape = self.convolution_to_shape(shape)
        shape = self.bimamba(shape)
        shape = self.convolution_shape(shape)
        shape = self.max_pooling_shape(shape)
        shape = self.dropout_shape(shape)

        seq_shape = torch.cat((seq, shape), dim=1)
        output = self.output(seq_shape)

        return output

    def forward(self, seq, shape):
        return self.execute(seq, shape)