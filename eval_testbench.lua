require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'string'
require 'optim'

local pl = require('pl.import_into')()

local top1, top3
local path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_hv02_testbench/"

local args = pl.lapp([[
  -m,--model    (default 'model.t7')
]])

local file = io.open ('./testbench_results/testbench_'..string.sub(args.model,1,-4)..'.txt', 'w')
file:write('Testbench results for model :'..args.model..'\n')
file:write('\n')

local function imgPreProcess(im)
   im = image.scale(im, 224,224)
   for i=1,3 do -- channels
      if model.img_mean then im[i]:add(-model.img_mean[i]) end
      if model.img_std then im[i]:div(model.img_std[i]) end
   end
   if(model.tensor_dim == 4) then
      im:view(1, im:size(1), im:size(2), im:size(3))
   end
   return im
end

local function isTop3(X_hat, X)
   bool = false
   for i=1,3 do
      if X_hat[i] == X then
	 bool=true
      end
   end
   return bool
end

local function getErrors(mat, nclasses) 
   local tp  = torch.diag(mat):resize(1, nclasses)
   local fn = (torch.sum(mat,2)-torch.diag(mat)):t()
   local fp = torch.sum(mat,1)-torch.diag(mat)
   local tn  = torch.Tensor(1,nclasses):fill(torch.sum(mat)):typeAs(tp) - tp - fn - fp
   return tp, tn, fp, fn
end

model = torch.load(args.model)
print '==> Loading Model'
model.convnet:add(nn.SoftMax())
model.convnet:cuda()
model.convnet:evaluate()

print(model.convnet)

local top1_classes = {}
local top3_classes = {}
local nTest = 0

top1 = 0
top3 = 0

confusion = optim.ConfusionMatrix(400)

print '==> Attempting Forward Passes...'
for dir in paths.iterdirs(path) do

   local top1_class = 0
   local top3_class = 0
   local n_class = 0
   
   for file in paths.iterfiles(path..dir) do
      nTest = nTest + 1
      n_class = n_class + 1
      
      local img = image.load(path..dir..'/'..file, 3)
         print(img)
	 io.read()
      img = imgPreProcess(img):cuda()

      local outputs = model.convnet:forward(img)
      local preds = outputs:float()
      local _, pred_sorted = preds:sort(2, true)
      print(pred_sorted)
      io.read()
      local lab = tonumber(string.sub(dir,6,-1))
      if pred_sorted[1][1] == lab then 
	 top1 = top1 + 1 
	 top1_class = top1_class + 1
      end 
      if isTop3(pred_sorted[1], lab) == true then 
	 top3 = top3 + 1
	 top3_class = top3_class + 1
      end
      confusion:add(pred_sorted[1][1], lab)
   end
   
   top1_class = top1_class * 100 / n_class
   top1_classes[dir] = top1_class
   top3_class = top3_class * 100 / n_class
   top3_classes[dir] = top3_class
   collectgarbage()
end

confusion:updateValids()
print('CM Test accuracy:', confusion.totalValid * 100)

top1 = top1 * 100 / nTest
top3 = top3 * 100 / nTest

print('==> Test ran on ' .. nTest .. ' samples')

print('Top1 '..top1)
print('Top3 '..top3)
--print(confusion:farFrr()[1])
--file:write('FAR/FRR :'..confusion:farFrr()..'\n')
file:write('\n')

--print(torch.type(confusion))

--file:write('True positive :'..getErrors(confusion,400)[1]..'\n')
--file:write('True negative :'..getErrors(confusion,400)[2]..'\n')
--file:write('False positive :'..getErrors(confusion,400)[3]..'\n')
--file:write('False positive :'..getErrors(confusion,400)[4]..'\n')
--file:write('\n')


file:write('Global Top1 :'..top1..'\n')
file:write('Global Top3 :'..top3..'\n')
file:write('\n')

file:write('|-----------------------|')
file:write('\n')
file:write('| Class  | TOP1 | TOP3  |')
file:write('\n')
file:write('|-----------------------|')
file:write('\n')

for k,v in pairs(top1_classes) do
   file:write('|'..k..'| '..string.format('%.1f',v)..'  '..string.format('%.1f\t',top3_classes[k])..'|\n')
end
file:write('\n')

confusion:zero()
file:close()
