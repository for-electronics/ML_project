import torch
import torch.nn as nn
import torchvision.models as models


class CNN_LSTM_Model(nn.Module):
    def __init__(self, hidden_size=512, num_classes=126):
        super(CNN_LSTM_Model, self).__init__()

        # MobileNetV2 as feature extractor
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True).features  # Use just the features

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_size, batch_first=True)

        # Fully connected layer to output classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten for CNN

        # Pass through MobileNetV2
        features = self.mobilenetv2(x)  # [batch_size * seq_len, 1280, 1, 1]
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 1280]

        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(features)

        # Get output from last timestep
        out = self.fc(lstm_out[:, -1, :])  # [batch_size, num_classes]

        return out
