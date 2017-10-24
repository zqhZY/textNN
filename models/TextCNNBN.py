import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()

        self.args = args
        V = args.vocab_size
        D = args.embed_dim
        C = args.num_classes
        Cin = 1
        Cout = args.kernel_num
        Ks = args.kernel_sizes
        args.linear_hidden_size = 1024

        self.embeding = nn.Embedding(V, D)
        # self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (K, D)) for K in Ks])
        convs_bn =[ nn.Sequential(
                                nn.Conv2d(in_channels = Cin,
                                        out_channels = Cout,
                                        kernel_size = (K, D)),
                                nn.BatchNorm2d(Cout),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.2),
                                # nn.ReLU(inplace=True),
                                # nn.MaxPool1d(kernel_size = (opt.title_seq_len - kernel_size + 1))
                            )
         for K in Ks]
        self.convs = nn.ModuleList(convs_bn)
        self.fc = nn.Sequential(
            nn.Linear(len(Ks)*Cout, args.linear_hidden_size),
            nn.BatchNorm1d(args.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_hidden_size, C)
        )

        if args.embedding_path:
            self.embeding.weight.data.copy_(torch.from_numpy(np.load(args.embedding_path)))

        #self.dropout = nn.Dropout(args.dropout)
        #self.fc = nn.Linear(len(Ks)*Cout, C)

    def conv_and_pool(self, x, conv):
        #x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.x.squeeze(3)  # (N,Co,W)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embeding(x)

        x = x.unsqueeze(1)
        x= [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        '''
                x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
                x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
                x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
                x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        #x = self.dropout(x)
        out = self.fc(x)
        return out


