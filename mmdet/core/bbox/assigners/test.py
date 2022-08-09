import re
from scipy.optimize import linear_sum_assignment
import torch
cls = torch.rand(300, 550, 10)
reg = torch.rand(300, 550, 4)
tg_ids = [0, 3, 7, 8]
match_cls = cls.view(-1,cls.size(2))[:, tg_ids].sigmoid()
match_reg = reg.view(-1, len(tg_ids))
match_fliter = torch.rand(300, 550, 1).view(-1, 1)
match_cls = match_cls * match_fliter.sigmoid()

print(match_cls.shape, match_reg.shape)
onenet = match_cls+match_reg
print(match_cls.t().shape)
defcn = match_cls * match_reg
print(onenet.shape, defcn.shape)
C_res, C_tgid = linear_sum_assignment(onenet)
D_res, D_tgid = linear_sum_assignment(defcn)
print(defcn)
print(C_res, C_tgid, D_res, D_tgid)

