import torch
from torch import nn

class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):   # recommended values
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha   # adjustable, e.g. 0.3 ??
        self.epsilon = epsilon   # adjustable

    def forward(self, input, target):

        multi_hot_key = target
        logits = input
        # logits = torch.sigmoid(logits)

        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()

        return loss.mean()
    

if __name__ == "__main__":

    m = nn.Sigmoid()
    loss = BinaryFocalLoss()

    input = torch.rand(3, requires_grad=True)
    target = torch.empty(3).random_(2)

    output = loss(m(input), target)

    print("loss:", output)
    output.backward()
    