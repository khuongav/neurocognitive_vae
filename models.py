import torch
import torch.nn as nn


c_dim = 3
z_dim = 32
eeg_chan = 98


def latent_sample(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.empty_like(std).normal_()
    return eps.mul(std).add_(mu)


class NetDown(nn.Module):
    def __init__(self, d, dropout=False):
        super(NetDown, self).__init__()

        self.dropout = dropout

        self.net_down_middle_block = nn.Sequential(
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1, padding=3, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=2, padding=2, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
        )

        self.net_down_middle_block_dropout = nn.Sequential(
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=1, padding=3, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.7),
            (
                nn.Conv1d(d * 4, d * 4, kernel_size=6, stride=2, padding=2, padding_mode='reflect', bias=False)),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.7)
        )

    def forward(self, x):
        if self.dropout:
            return self.net_down_middle_block_dropout(x)
        else:
            return self.net_down_middle_block(x)


class NetUp(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=6, stride=3, output_padding=0):
        super(NetUp, self).__init__()

        self.net_up_middle_block = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels,
                               kernel_size, stride, output_padding=output_padding),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(0.1),

        )

        self.net_up_final_block = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels,
                               kernel_size, stride, output_padding=output_padding),
        )

    def forward(self, noise, final_layer=False):
        if not final_layer:
            return self.net_up_middle_block(noise)
        else:
            return self.net_up_final_block(noise)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()

        # Construct the conv layers
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * T)
            returns :
                out : self attention value + input feature 
                attention: B * T * T
        """
        x = torch.unsqueeze(x, -1)
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product

        attention = self.softmax(energy)  # B * N * N
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(
            0, 2, 1))  # batch matrix-matrix product
        out = out.view(m_batchsize, C, width, height)  # B * C * W * H

        # Add attention weights onto input
        out = self.gamma*out + x
        return torch.squeeze(out, -1), attention


class FeatureExtractor(nn.Module):
    def __init__(self, d=32, eeg_chan=eeg_chan, dropout=False):
        super(FeatureExtractor, self).__init__()

        self.netE_down0 = nn.Sequential(
            (
                nn.Conv1d(eeg_chan, d * 4, kernel_size=1, stride=1, bias=True)),
            nn.LeakyReLU(0.1),
        )

        self.netE_down1 = NetDown(d, dropout)
        self.netE_down2 = NetDown(d, dropout)
        self.netA_1 = Self_Attn(d*4)
        self.netE_down3 = NetDown(d, dropout)
        self.netE_down4 = NetDown(d, dropout)

    def forward(self, x):

        x = self.netE_down0(x)
        x = self.netE_down1(x)
        x = self.netE_down2(x)
        x, _ = self.netA_1(x)
        x = self.netE_down3(x)
        x = self.netE_down4(x)

        x = torch.flatten(x, start_dim=1)

        return x


class EncoderX(nn.Module):
    def __init__(self, z_dim=z_dim):
        super(EncoderX, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.netE = FeatureExtractor()

        self.fcE = nn.Linear(128 * 16 + c_dim, z_dim * 2)

    def forward(self, x, c):
        x = torch.squeeze(x, dim=1)
        x = self.dropout(x)

        x = self.netE(x)
        x = torch.cat([x, torch.squeeze(c, dim=1)], 1)

        return self.fcE(x)


class DecoderX(nn.Module):

    def __init__(self, z_dim=z_dim, eeg_chan=eeg_chan, hid_dim=128, cond_addition=False):
        super(DecoderX, self).__init__()
        self.cond_addition = cond_addition

        self.fc = nn.Sequential(
            (nn.Linear(3, int(hid_dim))),
            nn.LeakyReLU(0.1),
            (nn.Linear(int(hid_dim), int(hid_dim / 4))),
            nn.LeakyReLU(0.1),
        )

        self.netD_up1 = NetUp(z_dim + int(hid_dim / 4),
                              hid_dim * 4, kernel_size=8, stride=4)
        self.netD_up2 = NetUp(hid_dim * 4, hid_dim * 2,
                              output_padding=1, kernel_size=8, stride=4)
        self.netA_1 = Self_Attn(hid_dim * 2)
        self.netD_up3 = NetUp(hid_dim * 2, hid_dim, output_padding=2)
        self.netD_up4 = NetUp(hid_dim, hid_dim, stride=1, output_padding=0)
        self.netA_2 = Self_Attn(hid_dim)
        self.netD_up5 = NetUp(
            hid_dim, eeg_chan, kernel_size=10, stride=2, output_padding=0)

    def forward(self, zX, z_ddm, c):
        c = torch.argmax(c, dim=2) + 1

        z_ddm = self.fc(z_ddm)

        if self.cond_addition:
            z_ddm = z_ddm + c
        else:    
            z_ddm = z_ddm * c

        z = torch.cat([zX, z_ddm], dim=1)
        z = torch.unsqueeze(z, 2)

        z = self.netD_up1(z)
        z = self.netD_up2(z)
        z, att1 = self.netA_1(z)
        z = self.netD_up3(z)
        z = self.netD_up4(z)
        z, att2 = self.netA_2(z)

        eeg = self.netD_up5(z, final_layer=True)
        eeg = torch.unsqueeze(eeg, 1)

        return eeg, att1, att2


class EEGVAE(nn.Module):
    def __init__(self, cond_addition=False):
        super(EEGVAE,  self).__init__()
        self.encoderX = EncoderX()
        self.decoderX = DecoderX(cond_addition=cond_addition)

    def forward(self):
        pass


class PosteriorDDM(nn.Module):
    def __init__(self):
        super(PosteriorDDM, self).__init__()

        self.netE = FeatureExtractor(dropout=True)

        self.fcEv_mean = nn.Sequential(
            nn.Linear(128 * 16 + c_dim, 1))
        self.fcEv_logvar = nn.Linear(128 * 16 + c_dim, 1)
        self.fcEc = nn.Linear(128 * 16 + c_dim, 1)
        self.tanh = nn.Tanh()

        self.fcEa_mean = nn.Sequential(
            nn.Linear(128 * 16 + c_dim, 1), nn.Softplus())
        self.fcEa_logvar = nn.Linear(128 * 16 + c_dim, 1)

        self.fcEndt_mean = nn.Sequential(
            nn.Linear(128 * 16 + c_dim, 1), nn.Softplus())
        self.fcEndt_logvar = nn.Linear(128 * 16 + c_dim, 1)

    def forward(self, x, c):
        x = torch.squeeze(x, dim=1)

        x = self.netE(x)
        x = torch.cat([x, torch.squeeze(c, dim=1)], 1)

        ndt_mean = self.fcEndt_mean(x)
        ndt_logvar = self.fcEndt_logvar(x)
        ndt_stat = torch.cat([ndt_mean, ndt_logvar], dim=1)

        a_mean = self.fcEa_mean(x)
        a_logvar = self.fcEa_logvar(x)
        a_stat = torch.cat([a_mean, a_logvar], dim=1)

        v_mean = self.fcEv_mean(x)
        choice = self.tanh(self.fcEc(x))
        # v_mean = torch.mul(v_mean, torch.sign(choice.detach()))
        v_logvar = self.fcEv_logvar(x)
        v_stat = torch.cat([v_mean, v_logvar], dim=1)
        
        return v_stat, a_stat, ndt_stat, choice


class BridgerDDM(nn.Module):
    def __init__(self, y_dim=1, hid_dim=128):
        super(BridgerDDM, self).__init__()

        self.netE = nn.Sequential(
            nn.Linear(y_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1)
        )

        self.fc = nn.Linear(hid_dim + c_dim, 3 * 2)

    def forward(self, y, c):
        y = self.netE(y)
        y = torch.cat(
            [torch.squeeze(y, dim=1), torch.squeeze(c, dim=1)], dim=1)

        return self.fc(y)


class BridgerEEG(nn.Module):
    def __init__(self, z_dim=z_dim, hid_dim=128):
        super(BridgerEEG, self).__init__()

        self.netE = nn.Sequential(
            nn.Linear(3, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1)
        )

        self.fc = nn.Linear(hid_dim + c_dim, z_dim * 2)

    def forward(self, ddm, c):
        ddm = self.netE(ddm)
        ddm = torch.cat(
            [torch.squeeze(ddm, dim=1), torch.squeeze(c, dim=1)], dim=1)

        return self.fc(ddm)