function residual_block(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
local concat = nn.ConcatTable()
local m = nn.Sequential()
m:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH):noBias())
m:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
m:add(nn.ReLU(true))
m:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH):noBias())
m:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
m:add(nn.SpatialDropout(0.1))
concat:add(m)
concat:add(nn.Identity())
return nn.Sequential():add(concat):add(nn.CAddTable()):add(nn.ReLU(true))
end

function residualWithConv_block(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
local concat = nn.ConcatTable()
local m = nn.Sequential()
local res = nn.Sequential()
m:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH):noBias())
m:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
m:add(nn.ReLU(true))
m:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, kW, kH, dW/2, dH/2, padW, padH):noBias())
m:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
m:add(nn.SpatialDropout(0.1))
res:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, 2, 2):noBias())
res:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
res:add(nn.SpatialDropout(0.01))
concat:add(m):add(res)
return nn.Sequential():add(concat):add(nn.CAddTable()):add(nn.ReLU(true))
end

function createModel(nGPU)
local model = nn.Sequential()

local initial_block = nn.ConcatTable(2)
initial_block:add(nn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1):noBias())
initial_block:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(initial_block)                                         -- 128x256
model:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
model:add(nn.SpatialBatchNormalization(16, 1e-3))
model:add(nn.ReLU(true))
-- first block
local block1 = nn.Sequential()
block1:add(residualWithConv_block(16, 32, 3, 3, 2, 2, 1, 1))
block1:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))
block1:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))

-- second block
local block2 = nn.Sequential()
block2:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))
block2:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))
block2:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))
block2:add(residual_block(32, 32, 3, 3, 1, 1, 1, 1))

-- third block
local block3 = nn.Sequential()
block3:add(residualWithConv_block(32, 64, 3, 3, 2, 2, 1, 1))
block3:add(residual_block(64, 64, 3, 3, 1, 1, 1, 1))
block3:add(residual_block(64, 64, 3, 3, 1, 1, 1, 1))
block3:add(residual_block(64, 64, 3, 3, 1, 1, 1, 1))
block3:add(residual_block(64, 64, 3, 3, 1, 1, 1, 1))
block3:add(residual_block(64, 64, 3, 3, 1, 1, 1, 1))

-- fourth block
local block4 = nn.Sequential()
block4:add(residualWithConv_block(64, 128, 3, 3, 2, 2, 1, 1))
block4:add(residual_block(128, 128, 3, 3, 1, 1, 1, 1))
block4:add(residual_block(128, 128, 3, 3, 1, 1, 1, 1))

model:add(block1):add(block2):add(block3):add(block4)
model:add(nn.Identity())

-- global average pooling 1x1
model:add(nn.SpatialAveragePooling(14, 14, 1, 1, 0, 0))
model:add(nn.View(-1):setNumInputDims(3))

local m = nn.Sequential()
      :add(makeDataParallel(model, nGPU))
      :add(nn.Linear(128,400))
      :add(nn.LogSoftMax())

m.imageSize = 256
m.imageCrop = 224
return m:cuda()
end
