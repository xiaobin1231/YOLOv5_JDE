# from utils.general import strip_optimizer
#
# if __name__ == "__main__":
#     path = '/home/xb/huawei/mymymymy/yolov5_JDE/weights/best.pt'
#     print(strip_optimizer(path))















import torch

pth = '/home/xb/huawei/mymymymy/new.pt'
model = torch.load(pth)
new_dict = {}
for k, v in model['model'].state_dict().items():
    new_dict[k] = v


torch.save(new_dict, '/home/xb/huawei/mymymymy/indeed.pt')






