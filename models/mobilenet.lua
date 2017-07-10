function createModel(nGPU)

  local mobilenet = nn.Sequential()

  --point-wise convolution
  local function conv_pw(nInput, nOutput)
    mobilenet:add(nn.SpatialConvolution(nInput,nOutput,1,1,1,1,0,0):noBias())
    mobilenet:add(nn.SpatialBatchNormalization(nOutput,1e-3))
    mobilenet:add(nn.ReLU(true))
  end

  -- depth-wise convolution
  local function conv_dw(nInput, s)
    mobilenet:add(nn.SpatialDepthWiseConvolution(nInput,1,3,3,s,s,1,1):noBias())
    mobilenet:add(nn.SpatialBatchNormalization(nInput,1e-3))
    mobilenet:add(nn.ReLU(true))
  end

  -- initial block
  --inptut 224x224x3
  local initial_block = nn.Sequential()
  initial_block:add(nn.SpatialConvolution(3,32,3,3,2,2,1,1))
  initial_block:add(nn.SpatialBatchNormalization(32,1e-3))
  initial_block:add(nn.ReLU(true))
  mobilenet:add(initial_block)
  --112x112x32
  conv_dw(32, 1)
  conv_pw(32, 64)
  --112x112x64
  conv_dw(64, 2)
  --56x56x64
  conv_pw(64, 128)
  --56x56x128
  conv_dw(128, 1)
  conv_pw(128, 128)
  conv_dw(128, 2)
  --28x28x128
  conv_pw(128, 256)
  --28x28x256
  conv_dw(256, 1)
  conv_pw(256, 256)
  conv_dw(256, 2)
  --14x14x256
  conv_pw(256, 512)
  --14x14x512
  conv_dw(512, 1)
  conv_pw(512, 512)     
  conv_dw(512, 1)
  conv_pw(512, 512)
  conv_dw(512, 1)
  conv_pw(512, 512)
  conv_dw(512, 1)
  conv_pw(512, 512)
  conv_dw(512, 1)
  conv_pw(512, 512)       
  --14x14x512
  conv_dw(512, 2)
  --7x7x512
  conv_pw(512, 1024)       
  --7x7x1024
  conv_dw(1024, 1)
  conv_pw(1024, 1024)    
   
  mobilenet:add(nn.SpatialAveragePooling(7, 7, 1, 1))
  mobilenet:add(nn.View(-1):setNumInputDims(3))

  local model = nn.Sequential()
      :add(makeDataParallel(mobilenet, nGPU))
      :add(nn.Linear(1024,400))
      :add(nn.LogSoftMax())

  model.imageSize = 256
  model.imageCrop = 224

  return model:cuda()
end
