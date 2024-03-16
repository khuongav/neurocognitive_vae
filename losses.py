import math
import torch
import torch.nn as nn

reconstruction_loss = nn.MSELoss(reduction='mean')
binary_choice_loss = nn.BCEWithLogitsLoss()

def wiener_loss(t, v, t0, a, backprop_ndt=False):
    # small-time version

    w = torch.tensor(0.5).cuda()     # convert to relative start point

    # we set K to be 10 regardless of the datasets
    kk = torch.arange(-4, 6)
    try:
        k = torch.tile(kk, (t.shape[0], 1)).cuda()
    except IndexError:
        k = kk.cuda()

    err = torch.tensor(0.02).cuda()
    if backprop_ndt:
        tt = torch.max(torch.abs(t) - t0, err) / torch.max(err, a) ** 2
    else:
        tt = torch.max(torch.abs(t) - torch.tensor(t0).cuda(),
                       err) / torch.max(err, a) ** 2

    tt_vec = torch.tile(tt, (1, 10))

    pp = torch.cumsum(
        (w + 2 * k) * torch.exp(-(((w + 2 * k) ** 2) / 2) / tt_vec), axis=1)
    pp = pp[:, -1] / \
        torch.sqrt(2 * torch.tensor(math.pi).cuda() * (torch.squeeze(tt) ** 3))
    pp = pp[:, None]

    v = torch.where(t > 0, -torch.abs(v), v)   # if time is negative, flip the sign of v
    v = torch.clamp(v, -6, 6)
    t = torch.where(t > 0, t, -t)

    p = pp * (torch.exp(-v * torch.max(err, a) * w -
              (v ** 2) * t / 2) / (torch.max(err, a) ** 2))
    p = torch.log(p)
    # p = torch.where(torch.tensor(v).cuda()>0, p, -p)
    return -(p.mean())


def softargmax(alpha, theda=100.0):
    time_step = alpha.shape[1]
    alpha = theda * alpha
    m = torch.nn.Softmax()
    alpha = m(alpha)
    indices = torch.range(start=0, end=time_step-1)
    indices_x = torch.reshape(indices, [-1, 1])  # 28 * 1
    outputs = torch.matmul(alpha.float().cuda(), indices_x.float().cuda())
    return outputs


def correlation_measure(y_pred, y_true):
    x = y_pred.clone()
    y = y_true.clone()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cov = torch.sum(vx * vy)
    corr = cov / (torch.sqrt(torch.sum(vx ** 2)) *
                  torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
    return corr
#     corr = torch.maximum(torch.minimum(corr,torch.tensor(1)), torch.tensor(-1))
#     return torch.sub(torch.tensor(1), corr ** 2)


def kl_divergence_loss(informative_priors, kl_weight, kl_ddm_weight, mu1, logvar1, mu2=None, logvar2=None, priors_ddm=None, prior_ndt=False):

    if mu2 is None:
        mu2 = torch.zeros_like(mu1)

        if informative_priors:
            mu2[:, -3] = torch.mul(priors_ddm[:, 0], torch.sign(mu1[:, -3]))
            mu2[:, -2] = priors_ddm[:, 1]
            if prior_ndt:
                mu2[:, -1] = priors_ddm[:, 2]
        else:
            mu2[:, -3] = torch.mul(1., torch.sign(mu1[:, -3]))
            mu2[:, -2] = 1.
            mu2[:, -1] = .5

    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

        if informative_priors:
            logvar2[:, -3] = torch.log(priors_ddm[:, 3])
            logvar2[:, -2] = torch.log(priors_ddm[:, 4])
            if prior_ndt:
                logvar2[:, -1] = torch.log(priors_ddm[:, 5])
        else:
            logvar2[:, -3] = torch.log(torch.full_like(priors_ddm[:, 3], 1e-3))
            logvar2[:, -2] = torch.log(torch.full_like(priors_ddm[:, 4], 1e-4))
            logvar2[:, -1] = torch.log(torch.full_like(priors_ddm[:, 5], 1e-5))


    kl_div = 0.5 * (logvar2 - logvar1 + (torch.exp(logvar1) +
                    (mu1 - mu2).pow(2)) / torch.exp(logvar2) - 1)
    kl_div_weight = torch.full_like(kl_div, kl_weight)
    kl_div_weight[:, -3] = kl_ddm_weight[-3]
    kl_div_weight[:, -2] = kl_ddm_weight[-2]
    kl_div_weight[:, -1] = kl_ddm_weight[-1]
    kl_div *= kl_div_weight

    kl_div_dim = kl_div.mean(dim=0)
    kl_div = kl_div.mean(dim=-1)
    kl_div = kl_div.view(-1).mean()

    return kl_div, kl_div_dim
