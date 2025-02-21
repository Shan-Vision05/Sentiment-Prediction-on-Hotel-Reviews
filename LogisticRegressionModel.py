import torch

#### Taken as is from the Sample Code ####
class SentimentClassifier(torch.nn.Module):

    def __init__(self, input_dim: int = 6, output_size: int = 1):
        super(SentimentClassifier, self).__init__()

        # Define the parameters that we will need.
        # Torch defines nn.Linear(), which gives the linear function z = Xw + b.
        self.linear = torch.nn.Linear(input_dim, output_size)

    def forward(self, feature_vec):
        # Pass the input through the linear layer,
        # then pass that through sigmoid to get a probability.
        z = self.linear(feature_vec)
        return torch.sigmoid(z)