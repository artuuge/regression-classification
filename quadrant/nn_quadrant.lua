quad = function (x, y)
   local z = 0; 
   if (y > 0) then
      if (x > 0) then z = 1;
      else
	 z = 2;
      end
   else
      if (x > 0) then z = 4;
      else
	 z = 3;
      end
   end
   return z
end; 


dataset={};
numExamples = 1e4;
function dataset:size() return numExamples end -- number of training examples 
for i=1,dataset:size() do 
   local input = torch.randn(2);     -- normally distributed example in 2d
   local output = quad(input[1], input[2]); 
   dataset[i] = {input, output}; 
end



require 'nn'
mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 4; HUs = 32; -- parameters

mlp:add(nn.Linear(inputs, HUs))

-- mlp:add(nn.Sigmoid())
mlp:add(nn.ReLU())
-- mlp:add(nn.Tanh())
-- mlp:add(nn.HardTanh())

mlp:add(nn.Linear(HUs, outputs))
mlp:add(nn.SoftMax())


--criterion = nn.ClassNLLCriterion()  
criterion = nn.CrossEntropyCriterion()

trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)


zs = torch.Tensor(4, 2)
zs[1] = torch.Tensor({0.5, 0.5})
zs[2] = torch.Tensor({-0.5, 0.5})
zs[3] = torch.Tensor({-0.5, -0.5})
zs[4] = torch.Tensor({0.5, -0.5})

for i = 1, (zs:size())[1] do
   local z1 = zs[i][1]
   local z2 = zs[i][2]
   local w = mlp:forward(zs[i])
   print(i, z1, z2)
   print(w)
end
