import os
import math
from functools import partial
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
import torch.optim as optim
from ema_pytorch import EMA
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm
from sklearn.metrics import (f1_score, accuracy_score)
import pandas as pd

from data.make_dataset import process_data
from utils import *


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        """
        standard ResNet style convolutional block
        """
        self.same_channels = inc == outc
        self.ks = kernel_size
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(gn, outc),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.layer(x)
        x = self.pool(x)
        return x

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat * 1
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 3
        self.d4_out = n_feat * 4

        self.u1_out = n_feat
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=8, factor=2)

        self.up2 = UnetUp(self.d3_out, self.u2_out, 1, gn=1, factor=2)
        self.up3 = UnetUp(self.u2_out + self.d2_out, self.u3_out, 1, gn=1, factor=2)
        self.up4 = UnetUp(self.u3_out + self.d1_out, self.u4_out, 1, gn=1, factor=2)
        self.out = nn.Conv1d(self.u4_out + in_channels, in_channels, 1)

    def forward(self, x, t):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        temb = self.sin_emb(t).view(-1, self.n_feat, 1)

        up1 = self.up2(down3)
        up2 = self.up3(torch.cat([up1 + temb, down2], 1))
        up3 = self.up4(torch.cat([up2 + temb, down1], 1))
        out = self.out(torch.cat([up3, x], 1))

        down = (down1, down2, down3)
        up = (up1, up2, up3)
        return out, down, up

class Encoder(nn.Sequential):
    def __init__(self, input_shape=(22, 750), in_channels=1, emb_size=16, depth=1, **kwargs):
        super(Encoder, self).__init__(
            PatchEmbedding(input_shape, in_channels, emb_size),
            TransformerEncoder(depth, emb_size)
        )

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            input_shape,
            in_channels,
            F1=8,
            D=2,
            F2=8 * 2,
            T1=125,
            T2=33,
            P1=8,
            P2=16,
            drop_out=0.25,
            pool_mode='mean'
    ):
        super(PatchEmbedding, self).__init__()
        s, t = input_shape[0], input_shape[1]
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels, F1, (1, T1), bias=False, padding='same'),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, (1, 1), stride=(1, 1), bias=False)
        )

        # Spatial
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (s, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            pooling_layer((1, P1)),
            nn.Dropout(drop_out),
            nn.Conv2d(F1 * D, F1 * D, (1, 1), stride=(1, 1), bias=False)
        )

        # Temporal-Spatial
        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), bias=False, padding='same', groups=F1),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            pooling_layer((1, P2)),
            nn.Dropout(drop_out),
            nn.Conv2d(F2, F2, (1, 1), stride=(1, 1), bias=False),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.temporal_spatial(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.25,
                 forward_expansion=4,
                 forward_drop_p=0.25
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class gELDM(nn.Module):
    def __init__(self, encoder, fc):
        super(gELDM, self).__init__()

        self.encoder = encoder
        self.fc = fc

    def forward(self, x0):
        encoder_out = self.encoder(x0)
        fc_out = self.fc(encoder_out.flatten(start_dim=1))
        return fc_out[0], fc_out[1], encoder_out

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.lin = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        last_layer = x.flatten(start_dim=1)
        x = self.lin(last_layer)
        return x, last_layer

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def ddpm_schedules(beta1, beta2, T):
    beta_t = cosine_beta_schedule(T, s=0.008).float()
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    return {
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, encoder, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.encoder = encoder.to(device)
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device

    def forward(self, x, model_name, *args):
        x = self.encoder(x)
        # Calculate mean and variance for each channel
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        mean = torch.mean(args[0], dim=0).unsqueeze(0)
        var = torch.var(args[0], dim=0).unsqueeze(0)
        noise = torch.Tensor(np.random.normal(mean.detach().cpu(), var.detach().cpu(), x.shape)).to(self.device)
        x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * noise
        times = _ts / self.n_T
        output, down, up = self.nn_model(x_t, times)
        return output, down, up, noise, x

def train(args):
    subject = args.subject
    device = torch.device(args.device)
    dataset_name = args.dataset_name
    channel = args.channel
    num_epochs = args.num_epochs
    model_name = args.model_name
    timestamp = args.timestamp
    early_stopping = args.early_stopping
    es_patience = 10
    kfold = 1
    epoch_result = args.epoch_result

    save_path = f'./results/checkpoints/{dataset_name}_{model_name}_{timestamp}/'
    os.makedirs(save_path, exist_ok=True), os.makedirs(f'./results/{dataset_name}_{model_name}_{timestamp}/', exist_ok=True)
    es = EarlyStopping(patience=es_patience, verbose=True, path=save_path, subject=subject, fold=kfold) if early_stopping else False
    train_loader, test_loader, num_classes = process_data(dataset_name, subject, device)

    n_T = 1000
    ddpm_dim = 16
    encoder_dim = 80

    encoder = Encoder(input_shape=(channel, 750), in_channels=1).to(device)
    ddpm_model = UNet(in_channels=5, n_feat=ddpm_dim).to(device)
    ddpm = DDPM(encoder=encoder, nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(device)
    fc = LinearClassifier(feature_dim=encoder_dim, num_classes=num_classes).to(device)
    geldm = gELDM(encoder, fc).to(device)

    # Define optimizer
    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.Adam(ddpm.parameters(), lr=base_lr)
    optim2 = optim.Adam(geldm.parameters(), lr=base_lr)

    # EMAs
    fc_ema = EMA(geldm.fc, beta=0.95, update_after_step=100, update_every=10, )

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(
        optimizer=optim1,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    scheduler2 = optim.lr_scheduler.CyclicLR(
        optimizer=optim2,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )

    best_acc, best_f1 = 0, 0
    data_rest_all = train_loader.dataset.data[:, 0, :, :].to(device)

    with tqdm(total=num_epochs, desc=f"Method ALL - Processing subject {subject}") as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            geldm.train()

            _, _, encoder_out_rest = geldm(data_rest_all)
            ############################## Train ###########################################
            for x, y in train_loader:
                x = x[:, 1, :, :].to(device)
                y = y.type(torch.LongTensor).to(device)
                y_cat = F.one_hot(y, num_classes=num_classes).type(torch.FloatTensor).to(device)

                # Train DDPM
                optim1.zero_grad()
                x_hat, down, up, noise, encoder_out = ddpm(x, model_name, encoder_out_rest, False)
                loss_ddpm = F.mse_loss(x_hat, encoder_out, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()

                # Train gELDM
                optim2.zero_grad()
                y_hat, ll_features_train, _ = geldm(x)
                y_hat = F.softmax(y_hat, dim=1)

                loss_c = F.cross_entropy(y_hat, y_cat)
                loss_d = loss_ddpm.detach().mean()
                loss = loss_d + 0.5 * loss_c
                loss.backward()
                optim2.step()
                scheduler1.step()
                scheduler2.step()
                fc_ema.update()

            metrics_train, ll_features_train, ll_labels_train, ll_features_rest_train = evaluate(geldm.encoder, fc_ema,
                                                                                                 train_loader, device,
                                                                                                 num_classes)

            ############################## Test ###########################################
            with torch.no_grad():
                ddpm.eval()
                geldm.eval()

                if epoch % 50 == 0:
                    end_epoch(ll_features_train, ll_labels_train, ll_features_rest_train, num_classes, epoch, subject,
                              metrics_train["accuracy"], metrics_train["f1"], timestamp, dataset_name, model_name, True)

                metrics_test, ll_features_test, ll_labels_test, ll_features_rest_test = evaluate(geldm.encoder, fc_ema,
                                                                                                 test_loader, device,
                                                                                                 num_classes)

                acc = metrics_test["accuracy"]
                f1 = metrics_test["f1"]
                if epoch % 10 == 0:
                    epoch_result = pd.concat([epoch_result, pd.DataFrame(
                        {'subject': subject, 'epoch': epoch, 'train_acc': [round(metrics_train["accuracy"] * 100, 2)],
                         'test_acc': [round(acc * 100, 2)], 'train_loss': [loss.item()]})], axis=0, ignore_index=True)

                # Early stopping
                if (epoch >= 400) and early_stopping:
                    best_acc = max(acc, best_acc)
                    if acc == best_acc:
                        best_f1 = f1
                    es(acc, geldm, ddpm)
                    if es.early_stop:
                        end_epoch(ll_features_test, ll_labels_test, ll_features_rest_test, num_classes, epoch, subject,
                                  acc, f1, timestamp, dataset_name, model_name)
                        print("Early stopping\n")
                        print(f"\nBest epoch: {epoch}, Best accuracy: {best_acc * 100:.2f}%, F1: {best_f1 * 100:.2f}%")
                        return subject, epoch, round(best_acc * 100, 2), round(best_f1 * 100, 2), epoch_result, loss.item()

                description = f"Epoch: {epoch}, Accuracy: {acc * 100:.2f}%, F1: {f1 * 100:.2f}%"
                pbar.set_description(f"Method ALL - Processing subject {subject} - {description}")
            pbar.update(1)

def evaluate(encoder, fc, generator, device, num_classes):
    labels = np.arange(0, num_classes)
    Y = []
    Y_hat = []
    ll_features = torch.Tensor()
    ll_features_rest = torch.Tensor()
    for x, y in generator:
        x_rest = x[:, 0, :, :].to(device)
        x = x[:, 1, :, :].to(device)
        y = y.type(torch.LongTensor).to(device)
        encoder_out_rest = encoder(x_rest)
        encoder_out = encoder(x)
        y_hat = fc(encoder_out.flatten(start_dim=1))[0]
        y_hat = F.softmax(y_hat, dim=1)
        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())
        ll_features_rest = torch.cat((ll_features_rest, encoder_out_rest.flatten(start_dim=1).detach().cpu()), dim=0)
        ll_features = torch.cat((ll_features, encoder_out.flatten(start_dim=1).detach().cpu()), dim=0)

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = torch.cat(Y_hat, dim=0).numpy()

    accuracy = accuracy_score(Y, Y_hat.argmax(axis=1))
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    metrics = {"accuracy": accuracy, "f1": f1}
    return metrics, ll_features, torch.tensor(Y), ll_features_rest

