"""Compact CRNN (CNN + BiLSTM) for CTC-based CAPTCHA recognition."""
import torch
import torch.nn as nn

from .config import NUM_CLASSES, IMG_H


class CRNN(nn.Module):
    """Convolutional feature extractor -> BiLSTM -> per-timestep class logits.

    Input : (B, 1, IMG_H, IMG_W)
    Output: (T, B, NUM_CLASSES)   log-probabilities, ready for CTCLoss.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 128):
        super().__init__()

        def block(cin, cout, pool):
            layers = [
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
            if pool is not None:
                layers.append(nn.MaxPool2d(pool[0], pool[1]))
            return layers

        self.cnn = nn.Sequential(
            *block(1, 32, ((2, 2), (2, 2))),     # H/2  W/2
            *block(32, 64, ((2, 2), (2, 2))),    # H/4  W/4
            *block(64, 128, ((2, 1), (2, 1))),   # H/8  W/4
            *block(128, 128, ((2, 1), (2, 1))),  # H/16 W/4
        )
        # After the conv stack the height is IMG_H/16. Collapse it to 1.
        self.final_h = IMG_H // 16
        self.head_conv = nn.Conv2d(128, 128, (self.final_h, 1))

        self.rnn = nn.LSTM(128, rnn_hidden, num_layers=2,
                           bidirectional=True, batch_first=False, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)                 # (B, 128, H', W')
        f = self.head_conv(f)           # (B, 128, 1, W')
        f = f.squeeze(2)                # (B, 128, W')
        f = f.permute(2, 0, 1)          # (T=W', B, 128)
        f, _ = self.rnn(f)
        f = self.dropout(f)
        logits = self.fc(f)             # (T, B, num_classes)
        return logits.log_softmax(2)
