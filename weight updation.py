import numpy as np
inp = [0.05, 0.1]
b = [0.35, 0.6]
wforh = [[0.15, 0.20], [0.25, 0.3]]
wfory = [[0.4, 0.45], [0.5, 0.55]]
out = [0.01, 0.99]


def sigmoid(x):
    return 1/(1+(np.e**(-x)))


def mse(pred, true):
    error = 0
    for i in range(len(true)):
        error += (true[i] - pred[i])**2
    return error/len(true)


def gradslayer2(h):
    ans = [[], []]
    sig1 = -(out[0]-yout[0])*yout[0]*(1-yout[0])
    for i in h:
        ans[0].append(sig1*i)
    sig2 = -(out[1]-yout[1])*yout[1]*(1-yout[1])
    for i in h:
        ans[1].append(sig2*i)
    return ans


def gradslayer1(hout):
    ans = [[],[]]
    k1 = -(out[0]-yout[0])*yout[0]*(1-yout[0])*wfory[0][0]*hout[0]*(1-hout[0])
    l1 = -(out[1]-yout[1])*yout[1]*(1-yout[1])*wfory[1][0]*hout[0]*(1-hout[0])
    ans[0].append(inp[0]*(k1+l1))
    ans[0].append(inp[1]*(k1+l1))
    k2 = -(out[0]-yout[0])*yout[0]*(1-yout[0])*wfory[0][1]*hout[1]*(1-hout[1])
    l2 = -(out[1]-yout[1])*yout[1]*(1-yout[1])*wfory[1][1]*hout[1]*(1-hout[1])
    ans[1].append(inp[0]*(k2+l2))
    ans[1].append(inp[1]*(k2+l2))
    return ans


epochs = int(input("Enter Epochs here - "))
lr = float(input("Enter Learning rate here - "))

for _ in range(epochs):

    hout = []
    for i in range(len(inp)):
        hout.append(sigmoid(np.sum(np.multiply(inp, wforh[i])) + b[0]))
    yout = []
    for i in range(len(hout)):
        yout.append(sigmoid(np.sum(np.multiply(hout, wfory[i])) + b[1]))
    e = mse(yout, out)
    print("Error = ", e)

    gr_lay_2 = gradslayer2(hout)
    gr_lay_1 = gradslayer1(hout)

    print("Layer 1 weights ", wforh)
    print("Layer 2 weights ", wfory)
    print("Layer 1 gradients ",gr_lay_1)
    print("Layer 2 gradients ",gr_lay_2)

    newwh = [[],[]]

    for i in range(len(wforh[0])):
        n = wforh[0][i] - (lr * gr_lay_1[0][i])
        newwh[0].append(n)
    for i in range(len(wforh[1])):
        n = wforh[1][i] - (lr * gr_lay_1[1][i])
        newwh[1].append(n)
    print("New Layer 1 = ", newwh)

    newwy = [[],[]]
    for i in range(len(wfory[0])):
        n = wfory[0][i] - (lr * gr_lay_2[0][i])
        newwy[0].append(n)
    for i in range(len(wfory[1])):
        n = wfory[1][i] - (lr * gr_lay_2[1][i])
        newwy[1].append(n)
    print("New Layer 2 = ", newwy)

    wforh = newwh[:]
    wfory = newwy[:]
    print("\n\n")


print("Outputs after training = ", yout)
print("True outputs = ", out)