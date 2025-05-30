import torch


def mae(output, target, reduction="mean"):
    return torch.nn.functional.l1_loss(output, target, reduction=reduction)


def mse(output, target, reduction="mean"):
    return torch.nn.functional.mse_loss(output, target, reduction=reduction)


def huber(output, target, reduction="mean", delta=1):
    return torch.nn.functional.huber_loss(
        output, target, reduction=reduction, delta=delta
    )


def angle_loss(output, target, target_minus1, avg_diff, reduction="mean"):
    dPR = torch.abs(target - output)
    dPRM = torch.sqrt(avg_diff**2 + (output - target_minus1) ** 2)
    dRRM = torch.sqrt(avg_diff**2 + (target - target_minus1) ** 2)
    cosR = (dPRM**2 + dRRM**2 - dPR**2) / (2 * dPRM * dRRM)
    cosR = torch.clamp(cosR, -0.999999, 0.999999)
    if reduction == "mean":
        loss = torch.mean(torch.arccos(cosR))
        return loss
    if reduction == "none":
        loss = torch.arccos(cosR)
        return loss


def rmse(output, target, reduction="mean"):
    err = (output - target) ** 2
    if torch.mean(err) == 0:
        err += 1e-10
    if reduction == "mean":
        loss = torch.sqrt(torch.mean(err))
        return loss
    if reduction == "none":
        loss = torch.sqrt(err)
        return loss


def nrmse(output, target, norm_factor, reduction="mean"):
    err = (output - target) ** 2
    if torch.mean(err) == 0:
        err += 1e-10
    if norm_factor == 0:
        norm_factor += 1
    if reduction == "mean":
        loss = torch.sqrt(torch.mean(err)) / norm_factor
        return loss
    if reduction == "none":
        loss = torch.sqrt(err) / norm_factor
        return loss

def rrmse(output, target, reduction="mean"):
    err = (output - target) ** 2
    if torch.mean(err) == 0:
        err += 1e-10
    denominarot = output**2
    if torch.mean(denominarot) == 0:
        denominarot += 1e-5
    if reduction == "mean":
        loss = torch.sqrt(torch.mean((err) / torch.sum(denominarot)))
        return loss
    if reduction == "none":
        loss = torch.sqrt((err) / denominarot)
        return loss


def msle(output, target, reduction="mean"):
    output = torch.clamp(output, min=0)
    target = torch.clamp(target, min=0)
    if reduction == "mean":
        loss = torch.mean((torch.log(1 + target) - torch.log(1 + output)) ** 2)
        return loss
    if reduction == "none":
        loss = (torch.log(1 + target) - torch.log(1 + output)) ** 2
        return loss


def rmsle(output, target, reduction="mean"):
    output = torch.clamp(output, min=0)
    target = torch.clamp(target, min=0)
    inside_val = (torch.log(1 + target) - torch.log(1 + output)) ** 2
    if torch.mean(inside_val) == 0:
        inside_val += 1e-10
    if reduction == "mean":
        loss = torch.sqrt(torch.mean(inside_val))
        return loss
    if reduction == "none":
        loss = torch.sqrt(inside_val)
        return loss


def mase(output, target, train_avg_diff, reduction="mean"):
    if train_avg_diff == 0:
        train_avg_diff += 1e-5
    if reduction == "mean":
        loss = torch.mean(torch.abs(output - target) / train_avg_diff)
        return loss
    if reduction == "none":
        loss = torch.abs(output - target) / train_avg_diff
        return loss


def rmsse(output, target, avg_sq_train_diff, reduction="mean"):
    if avg_sq_train_diff == 0:
        avg_sq_train_diff += 1e-5
    sq_err = (output - target) ** 2
    if torch.mean(sq_err) == 0:
        sq_err += 1e-10
    if reduction == "mean":
        loss = torch.sqrt(torch.mean((sq_err) / avg_sq_train_diff))
        return loss
    if reduction == "none":
        loss = torch.sqrt((sq_err) / avg_sq_train_diff)
        return loss


def poisson(output, target, reduction="mean"):
    output = torch.clamp(output, 1e-10)
    l = output - target * torch.log(output)
    if reduction == "mean":
        loss = torch.mean(l)
        return loss
    if reduction == "none":
        loss = l
        return loss


def logCosh(output, target, reduction="mean"):
    if reduction == "mean":
        loss = torch.mean(torch.log(torch.cosh(output - target)))
        return loss
    if reduction == "none":
        loss = torch.log(torch.cosh(output - target))
        return loss


def mape(output, target, reduction="mean"):
    target = torch.clamp(target, 1e-10)
    main_f = (target - output) / target
    if reduction == "mean":
        loss = torch.mean(torch.abs(main_f))
        return loss
    if reduction == "none":
        loss = torch.abs(main_f)
        return loss


def mbe(output, target, reduction="mean"):
    if reduction == "mean":
        loss = torch.mean(output - target)
        return loss
    if reduction == "none":
        loss = output - target
        return loss


def rae(output, target, dataset_mean, reduction="sum"):
    numerator = torch.abs(target - output)
    denominator = torch.abs(target - dataset_mean)
    if torch.mean(denominator) == 0:
        denominator += 1
    if reduction == "sum":
        loss = torch.sum(numerator) / torch.sum(denominator)
        return loss
    if reduction == "none":
        loss = numerator / denominator
        return loss


def rse(output, target, dataset_mean, reduction="sum"):
    numerator = torch.square(target - output)
    denominator = torch.square(target - dataset_mean)
    if torch.mean(denominator) == 0:
        denominator += 1
    if reduction == "sum":
        loss = torch.sum(numerator) / torch.sum(denominator)
        return loss
    if reduction == "none":
        loss = numerator - denominator
        return loss


def kernelMSE(output, target, reduction="sum"):
    sigma = torch.sqrt(torch.tensor(2)) / 2
    up = -(torch.square(target - output)) / (2 * sigma**2)
    l = 1 - torch.exp(up)
    if reduction == "sum":
        loss = torch.sum(l)
        return loss
    if reduction == "none":
        loss = l
        return loss


def quantile(output, target, q=0.5, reduction="sum"):
    errors = target - output
    loss = torch.max((q - 1) * errors, q * errors)

    if reduction == "sum":
        loss = torch.sum(loss)
        return loss
    if reduction == "none":
        return loss

def quantile25(output, target, reduction="sum"):
    return quantile(output, target, q=0.25, reduction=reduction)

def quantile75(output, target, reduction="sum"):
    return quantile(output, target, q=0.75, reduction=reduction)
