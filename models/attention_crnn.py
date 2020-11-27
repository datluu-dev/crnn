from .crnn import CRNN
from .attention import Attention 

class Attention_CRNN(CRNN):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super().__init__(self, imgH, nc, nclass, nh, n_rnn, leakyRelu)
        self.attention = Attention(nh, nh, nclass)

    def forward(self, input, length):
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        assert h == 1, "height of conv must be 1"
        conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        rnn = self.rnn(conv)
        output = self.attention(rnn, length)
        return output