# import torch
# import torch.nn as nn

# # 第一层忘记层

# # f_t = sigmoid(W_f[h_t-1, x_t] + b_f)   忘记门
# # i_t = sigmoid(W_i[h_t-1, x_t] + b_i)
# # ~C_t = tanh(Wc[h_t-1, x_t] + b_c)
# # i_t * ~C_t  更新层
# # C_t = f_t * C_t-1 + i_t * ~C_t  旧的细胞状态C_t-1与忘记门f_t相乘来丢弃一部分信息在加上需要更新的部分i_t*~C_t
# # o_t = sigmoid(W_o[h_t-1, x_t] + b_o)
# # h_t = o_t * tanh(C_t)
# class LSTM(nn.Module):
#     def __init__(self, input_size, cell_size, hidden_size):
#         super(LSTM, self).__init__()
#         self.cell_size = cell_size
#         self.hidden_size = hidden_size
#         self.fl = nn.Linear(input_size + hidden_size, hidden_size)
#         self.il = nn.Linear(input_size + hidden_size, hidden_size)
#         self.ol = nn.Linear(input_size + hidden_size, hidden_size)
#         self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

#     def forward(self, input, Hidden_State, Cell_State):
#         value = torch.cat((input, Hidden_State), 1)
#         f = F.sigmoid(self.f1(value))
#         i = F.sigmoid(self.il(value))
#         o = F.sigmoid(self.ol(value))
#         C = F.tanh(self.Cl(value))
#         Cell_State = f * Cell_State + i * C
#         Hidden_State = o * F.tanh(Cell_State)
#         return Hidden_State, Cell_State

#     def next(self, inputs):
#         batch_size = inputs.size(0)
#         time_step = inputs.size(1)
#         Hidden_State, Cell_State = self.initHidden(batch_size)
#         for i in range(time_step):
#             Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
#         return Hidden_State, Cell_State
import math
import torch
import torch.nn as nn


class NaiveLSTM(nn.Module):
    """My own LSTM"""
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门的权重矩阵和bias矩阵
        self.w_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))


        # 遗忘门的权重矩阵和bias矩阵
        self.w_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵
        self.w_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))

        # cell的权重矩阵和bias矩阵
        self.w_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.reset_weights()

    def reset_weights(self):
        """reset weights"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs:torch.Tensor, state):
        """
        Args:
            inputs: [1,1,input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        if state is None:
            h_t = torch.zero(1, self.hidden_size).t()
            c_t = torch.zero(1, self.hidden_size).t()
        else:
            (h,c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []
        seq_size = 1

        for t in range(seq_size):
            x = inputs[:, t, :].t()

            # input_gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)

            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_ig + self.w_hg @ h_t + self.b_hf)

            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t + self.b_hg)

            # output_gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)

def reset_weights(model):
    for weight in model.parameters():
        nn.init.constant_(weight, 0.5)


if __name__ == '__main__':
    inputs = torch.ones(1, 1, 10)
    h0 = torch.ones(1, 1, 20)
    c0 = torch.ones(1, 1, 20)
    print(h0.shape, h0)
    print(c0.shape, c0)
    print(inputs.shape, inputs)
    naive_lstm = NaiveLSTM(10, 20)
    reset_weights(naive_lstm)
    output1, (hn1, cn1) = naive_lstm(inputs, (h0, c0))
    print(hn1.shape, cn1.shape, output1.shape)
    print(hn1)
    print(cn1)
    print(output1)




