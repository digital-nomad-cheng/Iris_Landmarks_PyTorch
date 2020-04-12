import torch

class NLL_OHEM(torch.nn.NLLLoss):
    """Online hard sample mining, Needs input from nn.LogSoftmax()"""
    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True):
            self.ration = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio*num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            insta_losses[idx] = -x.data[idx, label]
        _, idxs = inst_losses.topk(num_hns)
        y_hn = y.index_select(0, idxs)
        x_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)

