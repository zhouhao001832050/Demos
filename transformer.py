import torch.nn as nn

# Transformer Parameters
d_model = 512 # Embedding size
d_ff = 2048 # Feed Forward dimension
d_k = d_v = 64 # dimension of K(=Q), V
n_layers = 6 # number of Encoders of Decoder Layer
n_heads = 8 # number of heads Multi-Head Attention


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in rnage(n_layers)])
    


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias = False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
