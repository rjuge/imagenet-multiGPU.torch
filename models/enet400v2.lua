----------------------------------------------------------------------
-- Create model and loss to optimize.
--
-- Adam Paszke,
-- May 2016.
----------------------------------------------------------------------
require 'cudnn'

local ct = 0
function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
   local internal = output / internal_scale
   local input_stride = downsample and 2 or 1

   local sum = nn.ConcatTable()

   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.SpatialConvolution(input, internal, input_stride, input_stride, input_stride, input_stride, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   if not asymetric and not dilated then
      main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   elseif asymetric then
      local pad = (asymetric-1) / 2
      main:add(cudnn.SpatialConvolution(internal, internal, asymetric, 1, 1, 1, pad, 0):noBias())
      main:add(cudnn.SpatialConvolution(internal, internal, 1, asymetric, 1, 1, 0, pad))
   elseif dilated then
      main:add(nn.SpatialDilatedConvolution(internal, internal, 3, 3, 1, 1, dilated, dilated, dilated, dilated))
   else
      assert(false, 'You shouldn\'t be here')
   end
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(output, 1e-3))
   main:add(nn.SpatialDropout((ct < 5) and 0.01 or 0.1))
   ct = ct + 1

   other:add(nn.Identity())
   if downsample then
      other:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   end
   if input ~= output then
      other:add(nn.Padding(1, output-input, 3))
   end

   return nn.Sequential():add(sum):add(nn.CAddTable()):add(nn.PReLU(output))
end


function createModel(nGPU)
local model = nn.Sequential()

local _ = require 'moses'
local bottleneck = _.bindn(_bottleneck, 4, true, false, false)
local cbottleneck = _.bindn(_bottleneck, 4, true, false, false)
local xbottleneck = _.bindn(_bottleneck, 4, true, 7, false)
local wbottleneck = _.bindn(_bottleneck, 4, true, 5, false)
local dbottleneck = _.bindn(_bottleneck, 4, true, false, 2)
local xdbottleneck = _.bindn(_bottleneck, 4, true, false, 4)
local xxdbottleneck = _.bindn(_bottleneck, 4, true, false, 8)
local xxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 16)
local xxxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 32)




local initial_block = nn.ConcatTable(2)
initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
initial_block:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(initial_block)                                         -- 128x256
model:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
model:add(nn.SpatialBatchNormalization(16, 1e-3))
model:add(nn.PReLU(16))
-- model:add(bottleneck(16, 64, true))                              -- 64x128
-- for i = 1,4 do
--    model:add(bottleneck(64, 64))
-- end
-- model:add(bottleneck(64, 128, true))                             -- 32x64
-- for i = 1,2 do
--    model:add(cbottleneck(128, 128))
--    model:add(dbottleneck(128, 128))
--    model:add(wbottleneck(128, 128))
--    model:add(xdbottleneck(128, 128))
--    model:add(cbottleneck(128, 128))
--    model:add(xxdbottleneck(128, 128))
--    model:add(wbottleneck(128, 128))
--    model:add(xxxdbottleneck(128, 128))
-- end
-- model:add(cudnn.SpatialConvolution(128, 1000, 1, 1)) -- Bx1000x12x12
-- 1st block
model:add(bottleneck(16, 64, true)) -- 56x56
model:add(bottleneck(64, 128))
model:add(bottleneck(128, 128))

-- 2nd block: dilation of 2
model:add(bottleneck(128, 256, true)) -- 28x28
model:add(bottleneck(256, 256))
model:add(dbottleneck(256, 256))

-- 3rd block: dilation 4
model:add(bottleneck(256, 400, true)) -- 14x14
model:add(bottleneck(400, 400))
model:add(xdbottleneck(400, 400))

-- global average pooling 1x1
model:add(cudnn.SpatialAveragePooling(14, 14, 1, 1, 0, 0))
model:add(nn.View(-1):setNumInputDims(3))
local gpu_list = {}
for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
if nGPU == 1 then
   gpu_list[1] = opt.devid
else
   for i = 1, nGPU do gpu_list[i] = i end
end
--model = nn.DataParallelTable(1, true, true):add(model, gpu_list)
local m = nn.Sequential()
      :add(makeDataParallel(model, nGPU))
      :add(nn.LogSoftMax())
print(opt.nGPU .. " GPUs being used")

---- Loss: NLL
--print('defining loss function:')
--local normHist = histClasses / histClasses:sum()
--local classWeights = torch.Tensor(#classes):fill(1)
--for i = 1, #classes do
--   if histClasses[i] < 1 or i == 1 then -- ignore unlabeled
--      classWeights[i] = 0
--   else
--      classWeights[i] = 1 / (torch.log(1.2 + normHist[i]))
--   end
--end

--loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

--loss:cuda()
----------------------------------------------------------------------
   --model.imageSize = 224
   --model.imageCrop = 224
   --model.name = "superresnetupconv_down_prelu"
   m.name = "ENet"
   --model.inputChannels = 3
   m.imageSize = 256
   m.imageCrop = 224
   print(m.name .. " created")
   return m:cuda()
end


