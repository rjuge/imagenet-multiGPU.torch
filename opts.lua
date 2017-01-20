--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './tinyimagenet/checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './tinyimagenetdataset/', 'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | nn')
    cmd:option('-cudnnAutotune',     0, 'Enable the cudnn auto tune feature Options: 1 | 0')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        16, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         30,    'Number of total epochs to run')
    -- cmd:option('-epochSize',       20000, 'Number of batches per epoch') 	-- for batch size 64
	-- cmd:option('-epochSize',       10000, 'Number of batches per epoch') 	-- for batch size 128
	cmd:option('-epochSize',     5000, 'Number of batches per epoch')	-- for batch size 256
	-- cmd:option('-epochSize',     2500, 'Number of batches per epoch')	-- for batch size 512
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       256,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'alexnetowtbn', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
