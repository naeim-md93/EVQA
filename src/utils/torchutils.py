import torch


class Accuracy:
    def __call__(self, y_pred, y):
        res = torch.argmax(input=y_pred, dim=1)
        res = (res == y).type(torch.FloatTensor)

        accuracy = torch.mean(input=res)

        return accuracy.item()