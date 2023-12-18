import torch
import torch.nn as nn



class CE_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labels = torch.eye(args.class_num)
        self.criterion = nn.CrossEntropyLoss().cuda()


    def forward(self, outputs, labels=None):
        device = (torch.device('cuda')
                  if outputs.is_cuda
                  else torch.device('cpu'))
        if labels == None:
            self.labels = self.labels.to(device)
            loss = self.criterion(outputs, self.labels)
        else:
            loss = self.criterion(outputs, labels)

        return loss



def compute_pda_loss(feats_q, feats_q1, prototypes_src, prototypes_tar):

    siml_q_src = torch.mm(feats_q, prototypes_src.t())
    siml_q_tar = torch.mm(feats_q, prototypes_tar.t())
    dist_siml_q = (siml_q_src - siml_q_tar).abs().sum(1).mean()
    siml_q1_src = torch.mm(feats_q1, prototypes_src.t())
    siml_q1_tar = torch.mm(feats_q1, prototypes_tar.t())
    dist_siml_q1 = (siml_q1_src - siml_q1_tar).abs().sum(1).mean()
    dist_siml = dist_siml_q + dist_siml_q1
    return dist_siml



def update_loss_weight(epoch, max):
    weight = min((epoch / max) * 1.0, 1.0)
    return weight




