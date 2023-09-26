import os
import shutil

import random
import numpy as np
import torch

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# plt.rcParams['figure.figsize'] = 20, 15
import seaborn as sns
# sns.set_theme()
# sns.set_style('white')


def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {k_1} vs {k_2}")
            # return False


def overwrite_log_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def get_noise_lb(snr):
    if snr == 'high':
        return 'Low Noise'
    elif snr == 'low':
        return 'High Noise'
    else:
        return 'Med Noise'


def get_cond_idx(conds, cond_of_interest):
    return torch.all(torch.eq(torch.squeeze(conds), cond_of_interest), dim=1)


def avg_n_rows(a, r, c):
    a = a.cpu().detach().numpy()
    a = a.transpose().reshape(-1, r).mean(1).reshape(c, -1).transpose()
    try:
        return np.squeeze(a, axis=1)
    except ValueError:
        return a


def squeeze_(a):
    return a.squeeze(dim=1).cpu().detach().numpy()


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    # original annealing: linearly increases to n_epoch * ratio

    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def get_rts_cond(dataloader, high_cond, med_cond, low_cond, device):
    rts_high = []
    rts_med = []
    rts_low = []

    for batch in dataloader:
        v0, eeg, rts, conds = torch.unsqueeze(batch[0].to(device), 1), torch.unsqueeze(batch[1].to(device), 1), \
            batch[2].to(device), torch.unsqueeze(batch[3].to(device), 1)

        cond_idx = get_cond_idx(conds, high_cond)
        rts_high_ = rts[cond_idx]
        rts_high.append(torch.flatten(rts_high_))

        cond_idx = get_cond_idx(conds, med_cond)
        rts_med_ = rts[cond_idx]
        rts_med.append(torch.flatten(rts_med_))

        cond_idx = get_cond_idx(conds, low_cond)
        rts_low_ = rts[cond_idx]
        rts_low.append(torch.flatten(rts_low_))

    rts_high = torch.hstack(rts_high).cpu().numpy()
    rts_med = torch.hstack(rts_med).cpu().numpy()
    rts_low = torch.hstack(rts_low).cpu().numpy()

    return rts_high, rts_med, rts_low


def get_fixed_ndt(train_dataloader, device):
    rts_train = []
    for batch in train_dataloader:
        v0, eeg, rts, conds = torch.unsqueeze(batch[0].to(device), 1), torch.unsqueeze(batch[1].to(device), 1), \
            batch[2].to(device), torch.unsqueeze(batch[3].to(device), 1)
        rts_train.append(torch.flatten(rts))

    fixed_ndt = 0.93 * torch.min(torch.abs(torch.hstack(rts_train))).item()
    return fixed_ndt


def plot_rt_dist(rts_high, rts_med, rts_low, fig_name, colors=['lightskyblue', 'g', 'r']):

    fig, ax = plt.subplots(1, 1)

    sns.distplot(rts_high, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 1},
                 label='Low Noise', color=colors[0])
    sns.distplot(rts_med, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 1},
                 label='Med Noise', color=colors[1])
    sns.distplot(rts_low, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 1},
                 label='High Noise', color=colors[2], axlabel='RT (ms)')
    ax.legend()
    # plt.savefig('saved_results/%s.png' %
    #             (fig_name), dpi=400)
    #     sns.boxenplot(data = (np.abs(rts_low), np.abs(rts_med), np.abs(rts_high)), ax=ax2)


def plot_fft(sig, ylim_):
    sampling_rate = 250
    fourier_transform = np.fft.rfft(sig)
    abs_fourier_transform = np.abs(fourier_transform)
    frequency = np.linspace(0, sampling_rate/2, len(abs_fourier_transform))
    plt.plot(frequency, abs_fourier_transform)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(25, 45)
    plt.ylim(0, ylim_)
#     plt.vlines(30, -1, 1)
#     plt.vlines(40, -1, 1)
    # plt.title('Mean frequency spectra - substract mean')
    plt.title('Mean frequency spectra')
    # plt.text(30, .8, "Some text")

    return abs_fourier_transform


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def write_logs(tb, loss_type, total_loss, rec_loss_eeg_avg, rec_loss_eeg, wfpt_loss, choice_loss, kl_loss, kl_loss_y,
               corr_n200, corr_a, corr_v, corr_ndt, kl_dim, batches_done):

    tb.add_scalars(loss_type+'/panel0',
                   {'total_loss': total_loss}, batches_done)

    tb.add_scalars(loss_type+'/panel1', {'wfpt_loss': wfpt_loss}, batches_done)
    tb.add_scalars(loss_type+'/panel1', {'choice_loss': choice_loss}, batches_done)

    tb.add_scalars(
        loss_type+'/panel2', {'rec_loss_eeg': rec_loss_eeg.item()}, batches_done)
    tb.add_scalars(
        loss_type+'/panel2', {'rec_loss_eeg_avg': rec_loss_eeg_avg.item()}, batches_done)

    tb.add_scalars(loss_type+'/kl',
                   {'kl_loss': kl_loss.item()}, batches_done)
    try:
        tb.add_scalars(loss_type+'/kl',
                       {'kl_loss_y': kl_loss_y.item()}, batches_done)
    except AttributeError:
        pass

    for idx, kl in enumerate(kl_dim.detach().cpu().numpy()):
        tb.add_scalars(loss_type + '/kl2', {'d %s ' % idx: kl}, batches_done)

    tb.add_scalars(loss_type+'/corr', {'corr_a': corr_a.item()}, batches_done)
    tb.add_scalars(loss_type+'/corr', {'corr_v': corr_v.item()}, batches_done)
    tb.add_scalars(loss_type+'/corr',
                   {'corr_ndt': corr_ndt.item()}, batches_done)
    tb.add_scalars(loss_type+'/corr', {'corr_n200': corr_n200}, batches_done)


def plot_erp(eeg_arr, dim=None, svded=False, ylim=None, n200_loc=None):
    if not svded:
        erp = np.mean(eeg_arr, axis=0).T
    else:
        erp = eeg_arr.T

    fig, axs = plt.subplots(figsize=(10, 4))
    axs.plot(np.arange(-100, 900, 4), erp)
    axs.xaxis.set_major_locator(
        ticker.FixedLocator(np.arange(-100, 900, 100)))
    axs.set_ylabel('Normalized Amplitude')
    axs.set_xlabel('Time (ms)')
    axs.set_title('ERP')
    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])
        axs.axvline(x = np.linspace(-100, 900,250)[int((n200_loc + 100)/ 4)], color = 'black', linewidth = 4, alpha=0.5, label = 'N200 Peak')

    axs.margins(x=0, y=0)
    if dim is not None:
        axs.text(-100, -1, 'Feature dim: %s' % dim)


def plot_fft_channels(eeg_arr, avg_trials=True, xlim=(25, 45), ylim=30, power_only=False):
    sampling_rate = 250

    if avg_trials:
        eeg_arr = np.mean(eeg_arr, axis=0)

    fourier_transform = np.fft.rfft(eeg_arr)
    # fourier_transform = np.mean(fourier_transform, axis=0)
    abs_fourier_transform = np.abs(fourier_transform).T
    frequency = np.linspace(0, sampling_rate/2, len(abs_fourier_transform))

    if not power_only:
        plt.figure(figsize=(5, 4))
        plt.plot(frequency, abs_fourier_transform)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(0, ylim)

        if avg_trials:
            plt.title('Mean Frequency Spectra')      
        else:
            plt.title('Frequency Spectra')
    
    hz_30 = np.max(abs_fourier_transform[25:35, :])
    hz_40 = np.max(abs_fourier_transform[35:45, :])
    print('30 Hz', hz_30)
    print('40 Hz', hz_40)
    return hz_30, hz_40


def plot_30_40_Hz_energies(eeg_arr_rec):

    def get_energies(eeg_arr):
        fourier_transform = np.fft.rfft(eeg_arr)
        abs_fourier_transform = np.abs(fourier_transform)

        energies_30Hz = np.max(abs_fourier_transform[:, :, 28:32], axis=(1, 2))
        energies_40Hz = np.max(abs_fourier_transform[:, :, 38:42], axis=(1, 2))

        return energies_30Hz, energies_40Hz
    
    # energies_30Hz_ori, energies_40Hz_ori = get_energies(eeg_arr_ori)
    energies_30Hz_rec, energies_40Hz_rec = get_energies(eeg_arr_rec)

    # sns.distplot(energies_30Hz_ori, hist = False, kde = True,
    #                     kde_kws = {'shade': True, 'linewidth': 2, 'color':'r'}, 
    #                     label = 'Original - 30 Hz')
    # sns.distplot(energies_40Hz_ori, hist = False, kde = True,
    #                     kde_kws = {'shade': True, 'linewidth': 2, 'color':'b'}, 
    #                     label = 'Original - 40 Hz')
    sns.distplot(energies_30Hz_rec, hist = False, kde = True,
                        kde_kws = {'shade': False, 'linewidth': 2, 'linestyle':'--', 'color':'r'}, 
                        label = '30 Hz' + ' (%.2f $\pm$ %.2f)' %(np.median(energies_30Hz_rec), np.std(energies_30Hz_rec)))
    sns.distplot(energies_40Hz_rec, hist = False, kde = True,
                        kde_kws = {'shade': False, 'linewidth': 2, 'linestyle':'--', 'color':'b'}, 
                        label = '40 Hz' + ' (%.2f $\pm$ %.2f)' %(np.median(energies_40Hz_rec), np.std(energies_40Hz_rec)))
    plt.title('Single-Trial Energy')
    plt.legend()


def set_seed(subject):
    if subject == 's59':
        seed = 42
    elif subject == 's109':
        seed = 1234
    elif subject == 's100':
        seed = 234
    elif subject == 's110':
        seed = 2859

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")