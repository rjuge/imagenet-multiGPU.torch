require 'torch'
require 'nn'
require 'cunn'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Horus Object Recognition Model Packager')
cmd:text()
cmd:text('Options')
cmd:option('-m', 'nin_bn_final.t7','path to model')
cmd:option('-o', 'horus_objectrecognitionmodel[ninbn].t7','output path')
cmd:option('-cudnn', false, 'convert to cudnn')

function load_synsets()
local list = {}
for line in io.lines 'synset_words.txt' do
    table.insert(list,string.sub(line,11))
end
return list 
end

local opt = cmd:parse(arg)
model = {}
--checkpoint = torch.load(opt.m,'b64')
--convnet = checkpoint.model
convnet = torch.load(opt.m,'b64')
convnet:remove()
convnet:add(nn.SoftMax()) --remove LogSoftMax and add SoftMax

if(opt.cudnn == true) then
    convnet = cudnn.convert(convnet, cudnn)

end
classids = torch.load('map_horusv2.t7','b64')
--classids = checkpoint.model.classes
--mean = checkpoint.model.transform.mean
--std = checkpoint.model.transform.std
meanstd = torch.load('meanstdCache.t7','b64')
mean = meanstd['mean']
std = meanstd['std']

model.version = "v3"
model.convnet = convnet
model.img_width = 224
model.img_height = 224
model.img_format = "RGB"
model.img_encoding = "Float"
model.tensor_dim = 4
model.img_mean = mean
model.img_std = std
model.classids = classids
model.labelOffset = 0
torch.save(opt.o, model)
