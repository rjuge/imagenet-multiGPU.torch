local Augmentations = require 'data_augmentations'

local DataAugmenter = torch.class('DataAugmenter')

--------------------------------------------------
----- Initialization
--------------------------------------------------
function DataAugmenter:__init(opt)
  
  self.pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

  self.meanstd = 
  {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
  }

  self.maxblurkernelwidth = 7
  --self.gaussianblurlayers = {}
  --self.validblurs = {}

  if(opt.nGpu > 0) then
    self.pca.eigval = self.pca.eigval:cuda()
    self.pca.eigvec = self.pca.eigvec:cuda()
  end
  
  Augmentations.InitGaussianKernels(7, opt.nGpu)
  
  self.augmentationPipeline = Augmentations.Compose
  {
    Augmentations.RandomLightning(0.80, self.pca),
    Augmentations.RandomHueJitter(0.5),
    Augmentations.RandomTinge(0.5),
    Augmentations.RandomBlurAndNoise(0.50, 0.75),
    Augmentations.RandomHorizontalFlip(0.5),
    Augmentations.RandomAffine(0.6),
  }
  
end

--------------------------------------------------
----- External Interface
--------------------------------------------------

function DataAugmenter:Augment(input)
  
  output = self.augmentationPipeline(input)
  return output
  
end
