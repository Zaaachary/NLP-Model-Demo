#! -*- encoding:utf-8 -*-
"""
@File    :   RNN_demo.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   a RNN model predict sin seris. 
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


class Net(nn.Module):
    '''
    a simple rnn model
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_prev_shape = (2, 1, 30)

        self.rnn = nn.RNN(
            input_size = 1, # feature of x
            hidden_size = 30,
            num_layers= 2,
            batch_first= True,  # x [batch, seq_len, feature]
        )
        
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=1.0)
 
        self.linear = nn.Linear(30, 1)

    def forward(self, x, hidden_prev):
        # out: [batch, seq_len, hidden_size]   ht in every time_step
        # hidden_prev: [batch, num_layers, hidden_size]
        out, hidden_prev = self.rnn(x, hidden_prev)

        # out = out.view(-1, 10) # [batch * seq_len, hidden_size]
        out = self.linear(out)  # out [batch * seq_len, hidden_size]
        return out, hidden_prev 


def generate_data(time_step: int=50):
    '''
    generate sin data
    return: 
        tensor x, y  x1,x2,x3 = y0,y1,y2  yi = xi-1 next
        x [batch, seq_len, feature]
    '''
    # start = np.random.randint(start, size=1)[0]  # np.int32
    start = np.random.randint(5)    # int
    time_steps = np.linspace(start, start+10, time_step)  # start, stop, num
    data = np.sin(time_steps).reshape(time_step , 1) # [time_step,] -> [time_step, 1]

    # yi = xi-1 next
    x = torch.tensor(data[:-1]).float().view(1, time_step-1, 1)
    y = torch.tensor(data[1:]).float().view(1, time_step-1, 1)
    return x, y, time_steps


def train_model(model, itertime=3000):
    min_loss = float('inf')
    best_model = None

    criterion = nn.MSELoss()    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for iter in range(itertime):
        hidden_prev = torch.zeros(model.hidden_prev_shape)
        x, y, _ = generate_data()

        # input the first three points for prediction to generate init hidden info
        with torch.no_grad():
            output, hidden_prev = model(x[:,:3,:], hidden_prev)
        
        output, hidden_prev = model(x[:,3:,:], hidden_prev)
        # hidden_prev = hidden_prev.detach() # detached from the current graph.

        loss = criterion(output, y[:,3:,:])
        if loss.item() < min_loss: # save current best model
            min_loss = loss.item()
            best_model = deepcopy(model)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            torch.nn.utils.clip_grad_norm_(p, 10)
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} Loss: {}".format(iter, loss.item()))
    print(min_loss)
    torch.save(best_model.state_dict(), './RNN_demo_best_model.pt')


def test_model(model):
    state_dict = torch.load('./RNN_demo_best_model.pt')
    model.load_state_dict(state_dict)
    model.eval()

    x, y, time_steps = generate_data()

    predictions = []  # x [batch, seq_len, feature]

    # input the first three points for prediction to generate init hidden info
    with torch.no_grad():
        input = x[:,:3,:]
        hidden_prev = torch.zeros(model.hidden_prev_shape)
        pred, hidden_prev = model(input, hidden_prev)
        # predictions.extend(pred.detach().numpy().ravel())
    
    # start sampling
    input = x[:, 3, :]
    for _ in range(x.shape[1]):
        input = input.view(1, 1, 1)
        pred, hidden_prev = model(input, hidden_prev)
        input = pred
        predictions.append(pred.detach().numpy().ravel()[0])

    x = x.data.numpy().ravel()

    # plot data,  adjust x, because the first three had been used to generate init hidden info
    x = x[3:]
    predictions = predictions[:-3]
    time_steps = time_steps[4:]
    print('x{},time{},pred{}'.format(x.shape, time_steps.shape, len(predictions)))
    plt.scatter(time_steps, x.ravel(), s=90)
    plt.plot(time_steps, x.ravel())

    plt.scatter(time_steps, predictions)
    plt.show()

if __name__ == "__main__":
    model = Net()
    model.train()
    
    if input('input "A" to train, "B" to test:') == 'B':
        test_model(model)
    else:
        import time
        start = time.time()
        train_model(model, 6000)
        print('total time:{}'.format(time.time()-start))
