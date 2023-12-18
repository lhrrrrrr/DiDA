import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



class DiDA(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.proto_weight = args.proto_m

        self.encoder = base_encoder(name=args.arch, low_dim=args.low_dim, class_num=args.class_num, pretrained=True)

        # create the prototypes
        self.register_buffer("prototypes_src", torch.zeros(args.class_num, 512))
        self.register_buffer("prototypes_tar", torch.zeros(args.class_num, 512))

    def set_prototype_update_weight(self, epoch, args):
        start = args.pro_weight_range[0]
        end = args.pro_weight_range[1]
        self.proto_weight = 1. * epoch / args.epoch * (end - start) + start


    def forward(self, img_q, img_q1=None, partial_Y=None, domain=None, args=None, eval_only=False):

        # for testing
        if eval_only:
            return self.encoder.encoder(img_q)

        # for training
        logits_q, feats_q = self.encoder(img_q)
        logits_q1, feats_q1 = self.encoder(img_q1)

        # obtain pseudo_label
        pred_scores_q = torch.softmax(logits_q, dim=1) * partial_Y
        pred_scores_q_norm = pred_scores_q / pred_scores_q.sum(dim=1).repeat(args.class_num, 1).transpose(0, 1)
        _, pseudo_labels_q = torch.max(pred_scores_q_norm, dim=1)


        if domain == 0:
            self.prototypes_src = self.prototypes_src.detach()

            # update momentum prototypes with pseudo labels
            for feat_q, label_q in zip(feats_q, pseudo_labels_q):
                self.prototypes_src[label_q] = self.proto_weight * self.prototypes_src[label_q] + (
                            1 - self.proto_weight) * feat_q
            # normalize prototypes
            self.prototypes_src = F.normalize(self.prototypes_src, p=2, dim=1)


            prot_scores = self.encoder.fc(self.prototypes_src)

            return logits_q, logits_q1, self.prototypes_src, prot_scores, pseudo_labels_q, feats_q, feats_q1

        elif domain == 1:
            self.prototypes_tar = self.prototypes_tar.detach()

            # update momentum prototypes with pseudo labels
            for feat_q, label_q in zip(feats_q, pseudo_labels_q):
                self.prototypes_tar[label_q] = self.proto_weight * self.prototypes_tar[label_q] + (
                            1 - self.proto_weight) * feat_q
            # normalize prototypes
            self.prototypes_tar = F.normalize(self.prototypes_tar, p=2, dim=1)

            prot_scores = self.encoder.fc(self.prototypes_tar)

            return logits_q, logits_q1, self.prototypes_tar, prot_scores, pseudo_labels_q, feats_q, feats_q1

        else:
            raise NotImplementedError("domain must be 0 or 1.")













