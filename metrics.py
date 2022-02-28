
__all__ = ["compute_correct"]


def compute_correct(output, target):
    _, preds = output.max(1) # [1] get the index of the max log-probability?
    correct = preds.eq(target).sum()

    return correct