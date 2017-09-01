import numpy as np
alpha_ = 0.25
gamma_ = 2

def softmax(x):
    sums = np.sum(np.exp(x))
    return np.exp(x) / sums
    
def pow(x,y):
    return np.power(x,y)

def log(x):
    return np.log(x)

def focal_loss(x, label):
    sm = softmax(x)
    pt = sm[label]
    return -alpha_ * pow(1-pt, gamma_) * log(pt)

def focal_diff(x, label):
    sm = softmax(x)
    pt = sm[label]
    diff = np.zeros(x.shape)
    for i, pc in enumerate(sm):
            if i == label:
                diff[i] =  alpha_ * pow(1 - pt, gamma_) * (gamma_ * pt * log(pt) + pt - 1);
            else:
                diff[i] = alpha_ * (pow(1 - pt, gamma_ - 1) * (-gamma_ * log(pt) * pt * pc) + \
                   pow(1 - pt, gamma_) * pc)
    return diff

def cross_entropy(x, label):
    sm = softmax(x)
    sm[label] -= 1
    return sm

def focal_num_diff(x, label):
    delta = 0.00001
    loss = focal_loss(x, label)
    num_diff = np.zeros(x.shape)
    for i in range(len(x)):
       pads = x.copy()
       pads[i] += delta 
       loss1 = focal_loss(pads, label)
       pads[i] -= delta * 2 
       loss2 = focal_loss(pads, label)
       num_diff[i] = (loss1 - loss2) / (delta * 2)
    return num_diff
       

prob = np.random.normal(5, 1, 20)
label = np.random.randint(20)

diff = focal_diff(prob, label)
num_diff = focal_num_diff(prob, label)
for  i in range(len(diff)):
    print("%-10.8f - %10.8f = %.8f" % (diff[i], num_diff[i], diff[i] - num_diff[i]))
