require 'cunn'
local ffi=require 'ffi'

function makeDataParallel(model, nGPU)   
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      if opt.backend == 'cudnn' and opt.cudnnAutotune == 1 then
        local gpu_table = torch.range(1, nGPU):totable()
        local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(function() require 'cudnn'
                                       cudnn.benchmark = true  end)
        dpt.gradInput = nil
        model = dpt:cuda()
      else
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
           cutorch.setDevice(i)
           model:add(model_single:clone():cuda(), i)
        end
        cutorch.setDevice(opt.GPU)
      end
   else
      if (opt.backend == 'cudnn' and opt.cudnnAutotune == 1) then
        require 'cudnn'
        cudnn.benchmark = true
      end
   end

   return model
end

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   --local newDPT = nn.DataParallelTable(1)
   --cutorch.setDevice(opt.GPU)
   --newDPT:add(module:get(1), opt.GPU)
   return module:get(1)
   --return newDPT
end

function saveDataParallel(model)
   if torch.type(model) == 'nn.DataParallelTable' then
      local temp_model = cleanDPT(model)
      return temp_model
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      return temp_model
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         elseif torch.type(module) == 'nn.Sequential' then
            model.modules[i] = makeDataParallel(module:float(), nGPU)
 	 end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

local function countModules(model)
   if torch.type(model) == 'nn.Sequential' then
      ft_model = nn.Sequential()
      local containers = #model
      local mod = 0
      local mod_cnt = 0
      for i=1,containers do
	 if torch.type(model:get(i))=='nn.Sequential' and #model:get(i):listModules() ~= 1 then
	    mod_cnt = mod_cnt + #model:get(i):listModules() - 1
	 elseif torch.type(model:get(i))=='nn.Sequential' and #model:get(i):listModules() == 1 then
	    mod_cnt = mod_cnt + 1
	 else
	    mod_cnt = mod_cnt + #model:get(i):listModules()
	 end
      end
      mod = mod_cnt
      return mod
   else
      error'Unsupported model type'
   end
end

function splitModel(m, layer_nb)
   assert(torch.type(m)=='nn.Sequential', 'Unsupported model type')
   local model = m:clone('weight','bias')
   local ft_model = nn.Sequential() 
   local containers = #model
   len = #model:get(1)
   for i=layer_nb,len do
      ft_model:add(model:get(1):get(layer_nb))
      model:get(1):remove(layer_nb)
   end
   for i=2,containers do
      model:remove()
   end
   collectgarbage()
   return model:get(1), ft_model
end
