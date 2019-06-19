import sys
import os
import argparse
import time
import numpy as np

import logging

from scipy.integrate import odeint as scp_od

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=400)
parser.add_argument('--max_mass', type=int, default=20)
parser.add_argument('--batch_time', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# true_y0 = torch.tensor([[1., 0.]])
# t = torch.linspace(0., 25., args.data_size)
# true_A = torch.tensor([[0., 1.0], [-2.0, -0.1]])

N = args.max_mass
time_space = args.data_size


n_concentration = torch.zeros(N)
n_concentration[1] = 1
K = torch.ones((N, N))
t = torch.linspace(0., 8., args.data_size)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

#         ax_traj.cla()
#         ax_traj.set_title('Trajectories')
#         ax_traj.set_xlabel('t')
#         ax_traj.set_ylabel('x,y')
#         ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
#         ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
#         ax_traj.set_xlim(t.min(), t.max())
#         ax_traj.set_ylim(-2, 2)
#         ax_traj.legend()

#         ax_phase.cla()
#         ax_phase.set_title('Phase Portrait')
#         ax_phase.set_xlabel('x')
#         ax_phase.set_ylabel('y')
#         ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
#         ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
#         ax_phase.set_xlim(-2, 2)
#         ax_phase.set_ylim(-2, 2)

#         ax_vecfield.cla()
#         ax_vecfield.set_title('Learned Vector Field')
#         ax_vecfield.set_xlabel('x')
#         ax_vecfield.set_ylabel('y')

#         y, x = np.mgrid[-2:2:21j, -2:2:21j]
#         dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
#         mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
#         dydt = (dydt / mag)
#         dydt = dydt.reshape(21, 21, 2)

#         ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#         ax_vecfield.set_xlim(-2, 2)
#         ax_vecfield.set_ylim(-2, 2)

#         fig.tight_layout()
#         plt.savefig('png/{:03d}'.format(itr))
#         plt.draw()
#         plt.pause(0.001)
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('n[1]. n[2]')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 1], t.numpy(), true_y.numpy()[:, 2], t.numpy(), true_y.numpy()[:, 3], 'g-', label='Test')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 1], '--', t.numpy(), pred_y.numpy()[:, 2],'--', t.numpy(), pred_y.numpy()[:, 3], 'b--', label='Predict')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(0, 1)
        ax_traj.legend()

#         ax_phase.cla()
#         ax_phase.set_title('Phase Portrait')
#         ax_phase.set_xlabel('x')
#         ax_phase.set_ylabel('y')
#         ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
#         ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
#         ax_phase.set_xlim(-2, 2)
#         ax_phase.set_ylim(-2, 2)

#         ax_vecfield.cla()
#         ax_vecfield.set_title('Learned Vector Field')
#         ax_vecfield.set_xlabel('x')
#         ax_vecfield.set_ylabel('y')

#         y, x = np.mgrid[-2:2:21j, -2:2:21j]
#         dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
#         mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
#         dydt = (dydt / mag)
#         dydt = dydt.reshape(21, 21, 2)

#         ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#         ax_vecfield.set_xlim(-2, 2)
#         ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

        
if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
#     ax_phase = fig.add_subplot(132, frameon=False)
#     ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)
    
def get_batch(true_y):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]
    batch_t = t[:args.batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
    return batch_y0, batch_t, batch_y
    

def dif_func(n, t, K):
    if type(n) == np.ndarray:
        dn = np.zeros_like(n)
        for i in range(1, n.shape[0]):
            dn[i] = np.sum([K[i - j, j] * n[i - j] * n[j] for j in range(1, i)]) - n[i] * np.sum([K[i, j] * n[j] for j in range(1, n.shape[0])])
        return dn
    elif type(n) == torch.Tensor:
        dn = torch.from_numpy(np.zeros_like(n.numpy()))
        for i in range(1, n.shape[0]):
            dn[i] = np.sum([K[i - j, j] * n[i - j] * n[j] for j in range(1, i)]) - n[i] * np.sum([K[i, j] * n[j] for j in range(1, n.shape[0])])
        return dn

class Lambda(nn.Module):
    def __init__(self, K):
        super(Lambda, self).__init__()
        self.K = K
        
    def forward(self, t, n):
        dn = n.clone()
        for i in range(1, n.shape[0]):
            a = torch.Tensor([self.K[i - j, j] * n[i - j] * n[j] for j in range(1, i)])
            b = torch.Tensor([self.K[i, j] * n[j] for j in range(1, n.shape[0])])
            dn[i] = (torch.sum(a)
                     - n[i] * torch.sum(b))
        return dn

class ODEFunc(nn.Module):
    def __init__(self, inp_dim, hide_dim):
        super(ODEFunc, self).__init__()
        self.inp_layer = nn.Linear(inp_dim, hide_dim)
        self.tanh = nn.Tanh()
        self.hide_layer = nn.Linear(hide_dim, hide_dim)
        self.out_layer = nn.Linear(hide_dim, inp_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
#         self.net = nn.Sequential(
#             nn.Linear(inp_dim, hide_dim),
#             nn.Tanh(),
#             nn.Linear(hide_dim, hide_dim),
#             nn.Tanh(),
#             nn.Linear(hide_dim, inp_dim),
#         )
#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)
        
    
    
    def forward(self, t ,y):
        y = self.inp_layer(y)
        y = self.tanh(y)
        y = self.hide_layer(y)
        y = self.tanh(y)
        y = self.hide_layer(y)
        y = self.tanh(y)
        y = self.out_layer(y)
        
        return y
    
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val    
        
        
if __name__ == '__main__':
    
#     fileHandler = logging.FileHandler('{}/{}.log'.format(logPath, fileName))
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)

    logger = logging.getLogger('log/myLog')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    logFormatter = logging.Formatter('%(asctime)s [%(threadName) - 12.12s] [%(levelname)-5.5s] %(message)s')
    
    ch.setFormatter(logFormatter)
    logger.addHandler(ch)
    
    print('n.size = ', n_concentration.shape)
    print('k.size = ', K.shape)
    
    logger.info('Create value')
    lmbd = Lambda(K)
    with torch.no_grad():
        true_y = odeint(Lambda(K), n_concentration, t, method='dopri5')
    print('y.shape = {}, type = {}'.format(true_y.shape, type(true_y)))
    logger.info('..Done')
    
    ii = 0
    
    logger.info('Create architecture')
    func = ODEFunc(args.max_mass, 100)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    
    logger.info('..Done')
    
    
    logger.info('Calculate')
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y)
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, n_concentration, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()