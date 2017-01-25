require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'class'
require 'data_augmenter'
require 'image'

outDir = "augmenterTestOut/"
torch.setdefaulttensortype('torch.FloatTensor')

nGpu = 1
local augmenter = DataAugmenter{nGpu = nGpu}
probe = image.load('lena.jpg')
probe = image.scale(probe, 256, 256):cuda()

for i=1, 50 do
   inImg = torch.Tensor(probe:size()):type(torch.type(probe)):copy(probe)
   --print(torch.type(inImg))
   --timer = torch.Timer()
output = augmenter:Crop(inImg)
   output = augmenter:Augment(output)
output = augmenter:Normalize(output)
   --totalTimer = totalTimer + timer:time().real
   --print(torch.type(output))
   image.save(paths.concat(outDir, "out_".. i ..".jpg"), output)
   
end
