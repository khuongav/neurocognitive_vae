from utilities import *
from losses import *
from models import *
from data import get_data_loader

import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# --------------------- Set Parameters ---------------------#

subject = 's59'
one_hot_cond = True

experiment_name = 'ddm-vae_%s' % subject
# experiment_name = 'ddm-vae_%s_uninformed' % subject
informative_priors = False if 'uninformed' in experiment_name else True
print('informative_priors', informative_priors)

snr_labels_train = ['high', 'med', 'low']
cond = 'all'

c_dim = 3
z_dim = 32

lr = 5e-4

beta = 20
kl_weight = beta * (z_dim + 3) / (98 * 250)

if informative_priors:
    if subject == 's59':
        klw_ndt = kl_weight / 5
    elif subject == 's109':
        klw_ndt = kl_weight * 1
    elif subject == 's100':
        klw_ndt = kl_weight / 2
    elif subject == 's110':
        klw_ndt = kl_weight * 1

    kl_ddm_weight = [4 * kl_weight, 2 * kl_weight, klw_ndt] # v, a, ndt
else:
    kl_ddm_weight = [kl_weight, kl_weight, kl_weight]

kl_y_weight = kl_weight

rec_x_avg_weight = 100
rec_x_weight = 1
rec_y_weight = 2

start_epoch = 0
n_epoch = 15001
joint_epoch = 14001
ndt_epoch = 3000
only_eeg_epoch = 5000

saved_every = 500
eval_every = 10

high_cond = torch.tensor([1, 0, 0]).cuda()
med_cond = torch.tensor([0, 1, 0]).cuda()
low_cond = torch.tensor([0, 0, 1]).cuda()

if start_epoch == 0:
    os.makedirs("saved_models/%s" % experiment_name, exist_ok=True)
    overwrite_log_dir("logs/%s" % experiment_name)
    
tb = SummaryWriter("logs/%s" % experiment_name)

# --------------------- Load Data ---------------------#

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset/'
train_dataloader, test_dataloader = get_data_loader('data_loaders/%s' %data_dir, batch_size=64, 
                                                    device=device, subject=subject,
                                                    shuffle_training=True, shuffle_testing=True,
                                                    snr_labels_train=snr_labels_train,
                                                    one_hot_cond=one_hot_cond)

fixed_ndt = get_fixed_ndt(train_dataloader, device)

# --------------------- Create Models ----------------------#

bridgerDDM = BridgerDDM().to(device)
bridgerEEG = BridgerEEG().to(device)
bridger_opt = torch.optim.Adam(list(bridgerDDM.parameters()) + list(bridgerEEG.parameters()), lr=lr)

vaeEEG = EEGVAE().to(device)
vae_eeg_opt = torch.optim.Adam(
    list(vaeEEG.encoderX.parameters()) + list(vaeEEG.decoderX.parameters()), lr=lr)

posteriorDDM = PosteriorDDM().to(device)
posterior_ddm_opt = torch.optim.Adam(posteriorDDM.parameters(), lr=lr)
posterior_ndt_opt = torch.optim.Adam(list(posteriorDDM.fcEndt_mean.parameters()) + list(posteriorDDM.fcEndt_logvar.parameters()), lr=lr)


if start_epoch > 0:
    saved_path = "saved_models/%s/ddmvae_%s.pth" % (
        experiment_name, start_epoch)
    print(saved_path)

    checkpoint = torch.load(saved_path)

    vaeEEG.load_state_dict(checkpoint['vae_state_dict'])
    posteriorDDM.load_state_dict(checkpoint['posterior_ddm_state_dict'])
    bridgerDDM.load_state_dict(checkpoint['bridger_ddm_state_dict'])
    bridgerEEG.load_state_dict(checkpoint['bridger_eeg_state_dict'])

    vae_eeg_opt.load_state_dict(checkpoint['vae_opt_state_dict'])
    posterior_ddm_opt.load_state_dict(
        checkpoint['posterior_ddm_opt_state_dict'])
    posterior_ndt_opt.load_state_dict(
        checkpoint['posterior_ndt_opt_state_dict'])
    bridger_opt.load_state_dict(checkpoint['bridger_opt_state_dict'])

# --------------------- Running ----------------------#

def run(batch, epoch):
    eeg, rts, conds, priors = torch.unsqueeze(batch[1].to(device), 1), \
        torch.unsqueeze(batch[2].to(device), 1), torch.unsqueeze(
            batch[3].to(device), 1), batch[4].to(device)
    # rt_idx = torch.squeeze(rts > 0)
    # eeg, rts, conds, priors = eeg[rt_idx], rts[rt_idx], conds[rt_idx], priors[rt_idx]

    eeg_avg = torch.mean(eeg, dim=2)

    encodingX = vaeEEG.encoderX(eeg, conds)
    q_meanX, q_logvarX = encodingX[:, :z_dim], (encodingX[:, z_dim:])
    v_stat, a_stat, ndt_stat, choice = posteriorDDM(eeg, conds)
    # if epoch > only_eeg_epoch:
    #     v_stat[:, :1] = torch.mul(v_stat[:, :1], torch.sign(choice.detach()))

    c_lbs = torch.sign(rts).squeeze()
    c_lbs[c_lbs < 0] = 0
    choice_loss = binary_choice_loss(choice.squeeze(), c_lbs) * rec_y_weight
    # rts = torch.abs(rts)

    if epoch <= ndt_epoch:
        ndt_stat = torch.zeros_like(ndt_stat)
    q_mean_ddm = torch.cat([v_stat[:, :1], a_stat[:, :1], ndt_stat[:, :1]], dim=1)
    q_logvar_ddm = torch.cat([v_stat[:, 1:], a_stat[:, 1:], ndt_stat[:, 1:]], dim=1)

    q_mean = torch.cat([q_meanX, q_mean_ddm], dim=1)
    q_logvar = torch.cat([q_logvarX, q_logvar_ddm], dim=1)
    z_sampleX = latent_sample(q_meanX, q_logvarX)
    z_sample_ddm = latent_sample(q_mean_ddm, q_logvar_ddm)

    v = torch.unsqueeze(z_sample_ddm[:, -3], 1) 
    a = torch.unsqueeze(z_sample_ddm[:, -2], 1)

    if epoch > ndt_epoch:
        ndt = torch.unsqueeze(z_sample_ddm[:, -1], 1)
        wfpt_loss = wiener_loss(rts, v, ndt, a, backprop_ndt=True) * rec_y_weight
    else:
        ndt = torch.full_like(v, fixed_ndt)
        wfpt_loss = wiener_loss(rts, v, ndt, a) * rec_y_weight

    eeg_rec, dec_att_1, dec_att_2 = vaeEEG.decoderX(z_sampleX, z_sample_ddm, conds)
    eeg_avg_rec = torch.mean(eeg_rec, dim=2)
    rec_loss_eeg_avg = reconstruction_loss(
        eeg_avg_rec, eeg_avg) * rec_x_avg_weight
    rec_loss_eeg = reconstruction_loss(eeg_rec, eeg) * rec_x_weight

    kl_loss, kl_dim = kl_divergence_loss(informative_priors, kl_weight, kl_ddm_weight, q_mean, q_logvar, priors_ddm=priors, prior_ndt=True if epoch > ndt_epoch else False)

    corr_ndt = correlation_measure(ndt, torch.abs(rts))
    corr_v = correlation_measure(v, rts)
    corr_a = correlation_measure(a, torch.abs(rts))
    
    n200_peaks_ori = softargmax(-eeg_avg[:, 0, 25 + int(150/4): 25 + int(275/4)])
    n200_peaks_rec = softargmax(-eeg_avg_rec[:, 0, 25 + int(150/4): 25 + int(275/4)])
    corr_n200 = correlation_measure(n200_peaks_ori, n200_peaks_rec)

    kl_loss_y = 0
    if epoch > joint_epoch:
        q_ddm_rt = bridgerDDM(rts, conds)
        q_mean_ddm_rt, q_logvar_ddm_rt = q_ddm_rt[:, :3], q_ddm_rt[:, 3:]
        kl_loss_ddm = kl_divergence_loss(informative_priors, kl_y_weight, kl_ddm_weight, q_mean_ddm, q_logvar_ddm, q_mean_ddm_rt, q_logvar_ddm_rt)[0]
        z_sample_ddm = latent_sample(q_mean_ddm_rt, q_logvar_ddm_rt)

        qX_rt = bridgerEEG(z_sample_ddm, conds)
        q_meanX_rt, q_logvarX_rt = qX_rt[:, :z_dim], qX_rt[:, z_dim:]
        kl_lossX = kl_divergence_loss(informative_priors, kl_y_weight, kl_ddm_weight, q_meanX, q_logvarX, q_meanX_rt, q_logvarX_rt)[0]
        
        kl_loss_y = kl_loss_ddm + kl_lossX 

    return rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, corr_n200, corr_a, corr_v, corr_ndt, kl_dim


def get_all_losses(batch, epoch):
    vae_eeg_opt.zero_grad()
    posterior_ddm_opt.zero_grad()
    bridger_opt.zero_grad()

    rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, \
        corr_n200, corr_a, corr_v, corr_ndt, kl_dim = run(batch, epoch)

    if epoch <= only_eeg_epoch:
        total_loss = kl_loss + wfpt_loss# + choice_loss
    elif epoch > only_eeg_epoch and epoch <= joint_epoch:
        total_loss = rec_loss_eeg + kl_loss
    elif epoch > joint_epoch:
        total_loss = kl_loss_y

    return total_loss, rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, corr_n200, corr_a, corr_v, corr_ndt, kl_dim


def eval(batches_done, epoch):
    vaeEEG.eval()
    posteriorDDM.eval()
    bridgerDDM.eval()
    bridgerEEG.eval()

    with torch.no_grad():
        val_total_loss, val_rec_loss_eeg_avg, val_rec_loss_eeg, val_wfpt_loss, val_choice_loss, val_kl_loss, val_kl_loss_y, \
            val_corr_n200, val_corr_a, val_corr_v, val_corr_ndt, val_kl_dim = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        for batch in test_dataloader:
            total_loss, rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, \
                corr_n200, corr_a, corr_v, corr_ndt, kl_dim = get_all_losses(batch, epoch)

            val_corr_n200 += corr_n200
            val_corr_a += corr_a
            val_corr_v += corr_v
            val_corr_ndt += corr_ndt

            val_total_loss += total_loss
            val_rec_loss_eeg_avg += rec_loss_eeg_avg
            val_rec_loss_eeg += rec_loss_eeg
            val_wfpt_loss += wfpt_loss
            val_choice_loss += choice_loss

            val_kl_loss += kl_loss
            val_kl_loss_y += kl_loss_y
            val_kl_dim += kl_dim

        len_val = len(test_dataloader)

        write_logs(tb, 'eval', val_total_loss / len_val, val_rec_loss_eeg_avg / len_val, val_rec_loss_eeg / len_val,
                   val_wfpt_loss / len_val, val_choice_loss / len_val, val_kl_loss / len_val, val_kl_loss_y / len_val,
                   val_corr_n200 / len_val, val_corr_a / len_val, val_corr_v / len_val, val_corr_ndt / len_val, val_kl_dim / len_val, 
                   batches_done)

        vaeEEG.train()
        posteriorDDM.train()
        bridgerDDM.train()
        bridgerEEG.train()


for epoch in range(start_epoch + 1, n_epoch):
    print('Epoch', epoch)

    for i, batch in enumerate(train_dataloader):

        total_loss, rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, \
            corr_n200, corr_a, corr_v, corr_ndt, kl_dim = get_all_losses(batch, epoch)

        total_loss.backward()

        if epoch <= ndt_epoch:
            posterior_ddm_opt.step()
        elif epoch > ndt_epoch and epoch <= only_eeg_epoch:
            posterior_ndt_opt.step()
        elif epoch > only_eeg_epoch and epoch <= joint_epoch:
            vae_eeg_opt.step()
        else:
            bridger_opt.step()

        batches_done = epoch * len(train_dataloader) + i

    write_logs(tb, 'train', total_loss, rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y, \
               corr_n200, corr_a, corr_v, corr_ndt, kl_dim, batches_done)

    if epoch % eval_every == 0:
        eval(batches_done, epoch)

    if epoch % saved_every == 0:
        torch.save({
            'epoch': epoch,
            'vae_state_dict': vaeEEG.state_dict(),
            'vae_opt_state_dict': vae_eeg_opt.state_dict(),

            'posterior_ddm_state_dict': posteriorDDM.state_dict(),
            'posterior_ddm_opt_state_dict': posterior_ddm_opt.state_dict(),
            'posterior_ndt_opt_state_dict': posterior_ndt_opt.state_dict(),

            'bridger_ddm_state_dict': bridgerDDM.state_dict(),
            'bridger_eeg_state_dict': bridgerEEG.state_dict(),
            'bridger_opt_state_dict': bridger_opt.state_dict(),
        }, "saved_models/%s/ddmvae_%s.pth" % (experiment_name, epoch))
