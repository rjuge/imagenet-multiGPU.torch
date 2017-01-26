--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'logger'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

-- Create loggers
local train_loss = 0
local val_loss = 0
local top1_train = 0
local top1_val = 0

lossLogFileName = 'loss.log'
perfLogFileName = 'perf.log'
lossLoggerPath = paths.concat(opt.save, lossLogFileName)
perfLoggerPath = paths.concat(opt.save, perfLogFileName)
print("Initializing Loggers: ", lossLogFileName, perfLogFileName)
lossLogger = Logger(lossLoggerPath)
lossLogger:setNames{'Training Loss','Validation Loss' }
perfLogger = Logger(perfLoggerPath)
perfLogger:setNames{'% top1 accuracy (train set)', '% top1 accuracy (val set)'}

for i=1,opt.nEpochs do
   train_loss, top1_train = train()
   val_loss, top1_val = test()

   lossLogger:add{train_loss, val_loss}
   lossLogger:style{'-', '-'}

   perfLogger:add{top1_train, top1_val} 
   perfLogger:style{'-', '-'}

   lossLogger:plot()
   perfLogger:plot()

   epoch = epoch + 1
   train_loss, top1_train = 0,0
   val_loss, top1_val = 0,0
end
