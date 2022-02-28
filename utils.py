class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Tradeoff(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.train_time = 0
        self.test_acc = 0

    def update(self, add_time, test_acc):
        self.train_time += add_time
        self.test_acc = test_acc
