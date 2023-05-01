import torch
for i in range(5):
    torch.cuda.empty_cache()
    print('released')
print(torch.cuda.is_available())