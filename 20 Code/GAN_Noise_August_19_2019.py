# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:44:27 2019

@author: michael
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

mean = 3.0
stddev = 0.2

channels = 30

g_input_size = 20    
g_hidden_size = 150  
g_output_size = channels  

d_input_size = channels
d_hidden_size = 75   
d_output_size = 1

batch_size = 50

num_epochs = 15000
print_interval = 1000

d_learning_rate = 3e-3
g_learning_rate = 8e-3




def get_real_sampler(mu, sigma):
    dist = Normal(mu, sigma)
    return lambda m, n: dist.sample((m, n)).requires_grad_()

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian

actual_data = get_real_sampler(mean, stddev)
noise_data = get_noise_sampler()



class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        super(Generator, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.nonLinearity = torch.nn.LeakyReLU()



    def forward(self, x):
        x = self.nonLinearity(self.input(x))
        x = self.nonLinearity( self.hidden(x) )
        return self.nonLinearity(self.output(x))
    
    
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        # Fill this in
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.nonLinearity = torch.nn.ELU()
        self.nonLinearityOutput = torch.sigmoid

    def forward(self, x):
        x = self.nonLinearity(self.input(x))
        x = self.nonLinearity(self.hidden(x))
        return self.nonLinearityOutput(self.output(x))
    
    
    
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)



criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)



def train_D_on_real():
    real_data = actual_data(batch_size, d_input_size)
    decision = D(real_data)
    error = criterion(decision, torch.ones(batch_size, 1))
    error.backward() 
    
    
    
def train_D_on_generated() :
  # Fill this in
    noise = noise_data(batch_size, g_input_size)
    generated_data = G(noise) 
    decision = D(generated_data)
    error = criterion(decision, torch.zeros(batch_size, 1))  # zeros = fake
    error.backward()
    
    
    
def train_G():
  # Fill this in
  noise = noise_data(batch_size, g_input_size)
  generated_data = G(noise)
  generated_decision = D(generated_data)
  error = criterion(generated_decision, torch.ones(batch_size, 1))  # we want to fool, so pretend it's all genuine

  error.backward()
  return error.item(), generated_data



# Training loop

losses = []

for epoch in range(num_epochs):
    D.zero_grad()
    
    train_D_on_real()    
    train_D_on_generated()
    d_optimizer.step()
    
    G.zero_grad()
    loss,generated = train_G()
    g_optimizer.step()
    
    losses.append(loss)
    
    if( epoch % print_interval) == (print_interval - 1):
        print("Epoch %6d. Loss %5.3f" % (epoch+1, loss))
        
print("Training complete")



import matplotlib.pyplot as plt

def draw(data): 
    plt.clf()
    plt.figure()
    d = data.tolist() if isinstance(data, torch.Tensor ) else data
    plt.plot(d) 
    plt.show()
    
    

# Generated distributions

generated_distributions = torch.empty(generated.size(0), 15) 
for i in range(0, generated_distributions.size(0)) :
    generated_distributions[i] = torch.histc(generated[i], min=0, max=15, bins=15)
draw(generated_distributions.t())



# Real distributions

real_distributions = torch.empty(generated.size(0), 15) 
real_data = actual_data(batch_size, d_input_size)

for i in range(generated.size(0)):
  real_distributions[i] = torch.histc(real_data[i], min=0, max=15, bins=15)
draw(real_distributions.t())







