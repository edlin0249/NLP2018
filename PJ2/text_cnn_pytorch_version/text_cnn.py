import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TextCNN(nn.Module):
    def __init__(self, sequence_length, num_classes, text_vocab_size, text_embedding_size, pos_vocab_size, 
        pos_embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.0):
        super(TextCNN, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.text_vocab_size = text_vocab_size
        self.text_embedding_size = text_embedding_size
        self.pos_vocab_size = pos_vocab_size
        self.pos_embedding_size = pos_embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda

        self.W_text = nn.Embedding(text_vocab_size, text_embedding_size)
        self.W_position = nn.Embedding(pos_vocab_size, pos_embedding_size)

        embedding_size = self.text_embedding_size + 2 * self.pos_embedding_size
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.num_filters, (filter_size, embedding_size)) for filter_size in self.filter_sizes])

        self.dropout_keep_prob = dropout_keep_prob

        self.dropout = nn.Dropout(self.dropout_keep_prob)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.fc1 = nn.Linear(num_filters_total, self.num_classes)



    def forward(self, text, pos1, pos2):
        text_embed = self.W_text(text)
        text_embed = text_embed.unsqueeze(1)
        pos1_embed = self.W_position(pos1)
        pos1_embed = pos1_embed.unsqueeze(1)
        pos2_embed = self.W_position(pos2)
        pos2_embed = pos2_embed.unsqueeze(1)
        #print(self.text_embed.data.type())
        #print(self.pos1_embed.data.type())
        #print(self.pos2_embed.data.type())
        embeds = torch.cat([text_embed, pos1_embed, pos2_embed], 3)
        #embedding_size = self.text_embedding_size + 2 * self.pos_embedding_size
        h = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs1]
        #print(h[0].size())
        #print(h[1].size())
        #print(h[2].size())
        #print(h[3].size())

        pooled_outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        #pooled_outputs = []
        #for i, filter_size in enumerate(self.filter_sizes):
        #    filter_shape = [self.num_filters, 1, filter_size, embedding_size]
            #print(filter_shape)
        #    W = Variable(torch.randn(filter_shape))
            #print(W.size())
            #b = Variable(torch.Tensor([0.1]*self.num_filters))
            #print(self.embeds.size())
        #    conv = F.conv2d(self.embeds, W, stride=(1,1), padding=0)
            #print(conv.size())
        #    h = F.relu(conv+0.1)
            #print(h.size())
        #    pooled = F.max_pool2d(h, kernel_size=[self.sequence_length - filter_size + 1, 1], stride=(1, 1), padding=0)
            #print(pooled.size())
        #    pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        #print(len(pooled_outputs))
        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, num_filters_total)
        #print(self.h_pool_flat.size())
        h_drop = self.dropout(h_pool_flat)

        logits = self.fc1(h_drop)
        return logits
