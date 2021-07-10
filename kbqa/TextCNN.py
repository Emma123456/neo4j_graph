import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_len, embedding_size, n_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embedding_size)

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[3, embedding_size])
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[4, embedding_size])
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[5, embedding_size])

        self.max_pool1 = nn.MaxPool1d(kernel_size=8)
        self.max_pool2 = nn.MaxPool1d(kernel_size=7)
        self.max_pool3 = nn.MaxPool1d(kernel_size=6)

        self.dropout = nn.Dropout(0.2)
        self.full_connect = nn.Linear(300, n_class)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.unsqueeze(1)

        cnn1_out = self.cnn1(embedding)
        cnn1_out = cnn1_out.squeeze(-1)
        cnn2_out = self.cnn2(embedding)
        cnn2_out = cnn2_out.squeeze(-1)
        cnn3_out = self.cnn3(embedding)
        cnn3_out = cnn3_out.squeeze(-1)

        out1 = self.max_pool1(cnn1_out)
        out2 = self.max_pool2(cnn2_out)
        out3 = self.max_pool3(cnn3_out)

        out = torch.cat([out1, out2, out3], dim=1).squeeze(-1)

        out = self.dropout(out)
        # 为什么不需要Sofmax ,因为评价函数使用的是CrossEntropyLoss，这里会做log_softmax
        out = self.full_connect(out)
        return out
