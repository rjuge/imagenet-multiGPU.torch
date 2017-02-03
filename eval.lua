require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

local pl = require('pl.import_into')()

local args = pl.lapp([[
  -m,--model    (default 'model.t7')
  -p,--probe    (default 'probe.jpg')
]])

function imgPreProcess(img)
  img = image.scale(img, 224,224)
  return img:view(1, img:size(1), img:size(2), img:size(3))
end

  
print '==> Loading Model'

convnet = torch.load(args.model)
convnet = convnet:cuda()
convnet:evaluate()

print(convnet)

print '==> Loading and Preprocessing Input Image...'
local img = image.load(args.probe, 3)
img = imgPreProcess(img):cuda()

print '==> Attempting Forward Pass...'
outputs = convnet:forward(img)
local pred = outputs:float()

local _, pred_sorted = pred:sort(2, true)
print(pred_sorted[1])

--l = convnet:get(1):get(1):get(41)
--print(l.output:resize(l.output:size(2)))
