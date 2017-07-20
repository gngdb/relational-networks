import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Kai's implementation, copied from pytorch branch
def relation(input, g, f=None, embedding=None, max_pairwise=None):
    # Batch size, number of objects, feature size
    b, o, c = input.size()
    # embedding is two dimensional
    # either (b, <embedding size>) or (1, <embedding size>)
    # Construct pairwise indices
    i = Variable(torch.arange(0, o).long().repeat(o))
    j = Variable(torch.arange(0, o).long().repeat(o, 1).t().contiguous().view(-1))
    if input.is_cuda:
        i, j = i.cuda(), j.cuda()
    # Create pairwise matrix
    pairs = torch.cat((torch.index_select(input, 1, i), torch.index_select(input, 1, j)), 2)
    # Append embedding if provided
    if embedding is not None:
        pairs = torch.cat((pairs, embedding.unsqueeze(1).expand(b, o ** 2, embedding.size(1))), 2)
    # Calculate new feature size
    c = pairs.size(2)
    # Pack into batches
    pairs = pairs.view(b * o ** 2, c)
    # Pass through g
    if max_pairwise is None:
        output = g(pairs)
    else:
        outputs = []
        for batch in range(0, b * o ** 2, max_pairwise):
            outputs.append(g(pairs[batch:batch + max_pairwise]))
        output = torch.cat(outputs, 0)
    # Unpack
    output = output.view(b, o ** 2, output.size(1)).sum(1).squeeze(1)
    # Pass through f if given
    if f is not None:
        output = f(output)
    return output


class RN(nn.Module):

    def __init__(self,args):
        super(RN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        self.coord_lst = [torch.from_numpy(np.array([self.cvt_coord(i) for _ in range(args.batch_size)])) for i in range(25)]

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

    def cvt_coord(self, i):
        return [(i/5-2)/2., (i%5-2)/2.]

    def forward(self, img, qst):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        ## x = (64 x 24 x 5 x 5)
        mb, n_channels, xdim, ydim = x.size()
        d = xdim*ydim
        x = x.view(mb, n_channels, d)
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d).permute(0,2,1)
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        return relation(x_flat, self.g, f=self.f, embedding=qst)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        x_j = torch.cat([x_j,qst],3)
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        # reshape for passing through network
        x_ = x_full.view(mb*d*d,63)
        x_ = self.g(x_)
        # reshape again and sum
        x_g = x_.view(mb,d*d,256)
        x_g = x_g.sum(1).squeeze()
        """f"""
        x_f = self.f(x_g)

        return F.log_softmax(x_f)

    def g(self, x_):
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        return x_

    def f(self, x):
        x_f = self.f_fc1(x)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f)
        x_f = self.f_fc3(x_f)
        return x_f

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        

    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy


    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}.pth'.format(epoch))
