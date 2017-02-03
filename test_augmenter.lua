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

for i=1, 100000 do
   inImg = torch.Tensor(probe:size()):type(torch.type(probe)):copy(probe)
   --print(torch.type(inImg))
   --timer = torch.Timer()
   inImg = augmenter:Crop(inImg)
   inImg = augmenter:Augment(inImg)
   inImg = augmenter:Normalize(inImg)
   --totalTimer = totalTimer + timer:time().real
   --print(torch.type(inImg))
   --image.save(paths.concat(outDir, "out_".. i ..".jpg"), inImg)
   
end
