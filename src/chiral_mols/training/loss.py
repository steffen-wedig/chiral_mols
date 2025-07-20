from torch import nn




class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma):
        pass





class CrossEntropyLoss(nn.Module):

    def __init__(self, class_weights):

        pass

    def forward(self, predicted_class_probabilities, reference_labels):
        pass


