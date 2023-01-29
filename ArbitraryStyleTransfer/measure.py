import torch
import time

from function import exact_feature_distribution_matching

X = torch.rand((1, 3, 512, 512)).cuda()
Y = torch.rand((1, 3, 512, 512)).cuda()
tick = time.time()
O = exact_feature_distribution_matching(X, Y)
tock = time.time()
print("Time: ", tock - tick)
