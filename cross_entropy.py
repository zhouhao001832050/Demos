import torch
import numpy as np


# def CrossEntropyLoss(output, target):
#     res = -output.gather(dim=1, index=target.view(-1,1))
#     res += torch.log(torch.exp(output).sum(dim=1).view(-1, 1))
#     res = res.mean()
#     print(res)

def myCrossEntropyLoss(output, label):
    loss = []
    for i, cls in enumerate(label):
        nll = -output[i][cls]
        log_x = np.log(sum([np.exp(j) for j in output[i]]))
        loss.append(nll + log_x)
    return np.mean(loss)


if __name__ == "__main__":
    # output = torch.tensor([
    # [1, 2, 3],
    # [4, 5, 6]
    #         ], dtype=torch.float32)

    # target = torch.tensor(
    # [0, 1])
    # print(CrossEntropyLoss(output, target))
    # ========================================
    x = np.array([
                [ 0.1545 , -0.5706, -0.0739 ],
                [ 0.2990, 0.1373, 0.0784],
                [ 0.1633, 0.0226, 0.1038 ]
            ])

    # 分类标签
    label = np.array([0, 1, 0])
    print("my CrossEntropyLoss output: %.4f"% myCrossEntropyLoss(x, label))


