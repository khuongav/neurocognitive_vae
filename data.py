import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(v0_path, eeg_path, rt_path):
    print(v0_path)
    with open(v0_path, 'rb') as f:
        v0 = np.load(f)
        print('v0', v0.shape)

    with open(eeg_path, 'rb') as f:
        eeg = np.load(f)
        print('eeg', eeg.shape)

    with open(rt_path, 'rb') as f:
        rt = np.load(f)
        print('rt', rt.shape)

    return v0, eeg, rt


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def make_data_loader(v0, eeg, rt, conds, priors, batch_size, device, shuffle):
    v0 = torch.from_numpy(v0.copy()).float().to(device)
    print('v0_sub.shape', v0.shape)
    eeg = torch.from_numpy(eeg.copy()).float().to(device)
    print('eeg_sub.shape', eeg.shape)

    rt = torch.from_numpy(rt.copy()).float().to(device)
    print('rt_sub.shape', rt.shape)
    conds = torch.from_numpy(conds.copy()).float().to(device)
    print('conds_sub.shape', conds.shape)
    priors = torch.from_numpy(priors.copy()).float().to(device)
    print('priors_sub.shape', priors.shape)
    v0_eeg_rt_conds_priors = TensorDataset(v0, eeg, rt, conds, priors)

    data_loader = DataLoader(
        v0_eeg_rt_conds_priors, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def get_data_loader(dataset_prefix, batch_size, device, subject, shuffle_training=True, shuffle_testing=True, snr_labels_train=['high', 'med', 'low'], one_hot_cond=True):

    cond_values = {'high': 1, 'med': 2, 'low': 3}
    if subject == 's59':
        prior_values = {'high': [1.5324653364172738,
                                 1.4435200413438745,
                                 0.280558571562465,
                                 0.004012675745488551,
                                 0.000714615266116153,
                                 3.08513150026066e-06],
                        'med': [1.488954766582121,
                                1.5467805978346614,
                                0.2666104306269267,
                                0.003684116834538521,
                                0.0008685321690840548,
                                3.4577067873152984e-06],
                        'low': [0.7859354734985293,
                                1.4255523712417282,
                                0.3348746158941173,
                                0.002855133408089654,
                                0.0005196431616208804,
                                3.731812878893718e-06]}
    elif subject == 's109':
        prior_values = {'high': [2.053575974492313,
                                 1.3008459324003314,
                                 0.36074313880195746,
                                 0.005598426820587994,
                                 0.0007487388762021242,
                                 5.644763962973401e-06],
                        'med': [1.7719287702803694,
                                1.2821880192529722,
                                0.3676846445621556,
                                0.004826471053787847,
                                0.0006571554487690494,
                                7.028899674226639e-06],
                        'low': [1.1986047280821215,
                                1.2604699715906729,
                                0.39393988203441577,
                                0.0038751748121676984,
                                0.0004768521093585591,
                                5.896607173795102e-06]}
    elif subject == 's100':
        prior_values = {'high': [1.4374165023062204,
                                 1.764450204375769,
                                 0.3912952198961386,
                                 0.0029373657627004184,
                                 0.0012435831058223218,
                                 1.2135542613155777e-05],
                        'med': [1.4222748440601698,
                                1.7315536254185158,
                                0.43965122954688274,
                                0.002897802492100718,
                                0.001155695605308332,
                                1.1701509844106998e-05],
                        'low': [1.147249385366183,
                                1.7363872866995504,
                                0.5005097745718387,
                                0.002526571039651541,
                                0.0010979466658905146,
                                1.4992378855519454e-05]}
    elif subject == 's110':
        prior_values = {'high': [2.145639323731114,
                                1.5887535578268066,
                                0.4862547719174235,
                                0.005299640435349637,
                                0.0011985162330547076,
                                4.357423282727389e-06],
                        'med': [1.8874035424826068,
                                1.4794304253294146,
                                0.48919718090626263,
                                0.00468808871084913,
                                0.0009613425656859002,
                                5.6124978546892605e-06],
                        'low': [1.3870239123130688,
                                1.4687445861735604,
                                0.5140392532410207,
                                0.003576952977682242,
                                0.0007245680593937209,
                                5.544730209959142e-06]}

    snr_labels = ['high', 'med', 'low']
    v0_train_subject = []
    eeg_train_subject = []
    rt_train_subject = []
    conds_train_subject = []
    priors_train_subject = []

    v0_val_subject = []
    eeg_val_subject = []
    rt_val_subject = []
    conds_val_subject = []
    priors_val_subject = []

    for idx, snr in enumerate(snr_labels):
        v0_val, eeg_val, rt_val = load_data(dataset_prefix + 'v0_val_%s_%s.npy' % (subject, snr), dataset_prefix +
                                            'eeg_val_%s_%s.npy' % (subject, snr), dataset_prefix + 'rt_val_%s_%s.npy' % (subject, snr))

        prior_val = [prior_values[snr]] * len(rt_val)
        priors_val_subject.append(prior_val)
        if one_hot_cond:
            conds_val = [idx] * len(rt_val)
            conds_val_subject.extend(conds_val)
        else:
            conds_val = np.array([[cond_values[snr]]*3] * len(rt_val))
            conds_val_subject.append(conds_val)

        v0_val_subject.append(v0_val)
        eeg_val_subject.append(eeg_val)
        rt_val_subject.append(rt_val)

        if snr in snr_labels_train:
            v0_train, eeg_train, rt_train = load_data(dataset_prefix + 'v0_train_%s_%s.npy' % (
                subject, snr), dataset_prefix + 'eeg_train_%s_%s.npy' % (subject, snr), dataset_prefix + 'rt_train_%s_%s.npy' % (subject, snr))

            if not one_hot_cond:
                v0_train = np.concatenate([v0_train, v0_val])
                eeg_train = np.concatenate([eeg_train, eeg_val])
                rt_train = np.concatenate([rt_train, rt_val])

            prior_train = [prior_values[snr]] * len(rt_train)
            priors_train_subject.append(prior_train)
            if one_hot_cond:
                conds_train = [idx] * len(rt_train)
                conds_train_subject.extend(conds_train)
            else:
                conds_train = np.array([[cond_values[snr]]*3] * len(rt_train))
                conds_train_subject.append(conds_train)

            v0_train_subject.append(v0_train)
            eeg_train_subject.append(eeg_train)
            rt_train_subject.append(rt_train)

    v0_train_subject = np.vstack(v0_train_subject)
    eeg_train_subject = np.concatenate(eeg_train_subject, axis=0)
    rt_train_subject = np.concatenate(rt_train_subject)
    priors_train_subject = np.concatenate(priors_train_subject)
    if one_hot_cond:
        conds_train_subject = get_one_hot(
            np.array(conds_train_subject), nb_classes=3)
        conds_val_subject = get_one_hot(
            np.array(conds_val_subject), nb_classes=3)
    else:
        conds_train_subject = np.concatenate(conds_train_subject)
        conds_val_subject = np.concatenate(conds_val_subject)

    v0_val_subject = np.vstack(v0_val_subject)
    eeg_val_subject = np.concatenate(eeg_val_subject, axis=0)
    rt_val_subject = np.concatenate(rt_val_subject)
    priors_val_subject = np.concatenate(priors_val_subject)

    train_data_loader = make_data_loader(
        v0_train_subject, eeg_train_subject, rt_train_subject, conds_train_subject, priors_train_subject, batch_size, device, shuffle_training)
    val_data_loader = make_data_loader(
        v0_val_subject, eeg_val_subject, rt_val_subject, conds_val_subject, priors_val_subject, 64, device, shuffle_testing)

    return train_data_loader, val_data_loader
