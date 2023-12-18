import argparse
import os, sys

import torch.optim

from utils.utils import adjust_learning_rate

sys.path.append('./')
import random
import shutil
from Trainer import trainer
from data_load import load_idx
from model import *
from resnet import Base_ResNet
from utils.utils_pll import *
from utils.utils_loss import *
import time


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def get_args():
    parser = argparse.ArgumentParser(
        description='DiDA: Disambiguated Domain Alignment for Cross-Domain Retrieval with Partial Labels')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")

    parser.add_argument('--dset', type=str, default='a2c', help="domain")
    parser.add_argument('--dataset', default='office_home',
                        choices=['office_home', 'office31', 'image_CLEF'])
    parser.add_argument('--class_num', type=int, default=65)

    parser.add_argument('--PLL_method', default='DiDA',
                        help="the method of Cross-Domain Retrieval with Partial Labels")
    parser.add_argument('--partial_rate', type=float, default=0.3)
    parser.add_argument('--start_epoch',
                        type=int,
                        default=0,
                        help="start epoch")
    parser.add_argument('--epoch',
                        type=int,
                        default=50,
                        help="maximum epoch")

    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help="model path")
    parser.add_argument('--arch',
                        type=str,
                        default="resnet50",
                        choices=["resnet50", "resnet18"])
    parser.add_argument('--low_dim', type=int, default=512)
    parser.add_argument('--exp_dir', type=str, default='experiment',
                        help='experiment directory for saving checkpoints and logs')
    parser.add_argument('--fea_path', type=str, default='features',
                        help='path for save features')
    parser.add_argument('--file', type=str, default='target')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="batch_size")
    parser.add_argument('--feat_dim',
                        type=int,
                        default=128,
                        help='embedding dimension')

    parser.add_argument('--proto_m',
                        default=0.99,
                        type=float,
                        help='momentum for computing the momving average of prototypes')
    parser.add_argument('--pro_weight_range', default='0.9, 0.5', type=str,
                        help='prototype updating coefficient')

    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--lr',
                        type=float,
                        default=3e-3,
                        help="learning rate 3e-3,1e-4")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--pretretrained', type=str2bool,
                        default=True,
                        help='use pretrained model')

    parser.add_argument('--lr_decay_epochs', type=str, default='20,40',
                        help='where to decay lr, can be a list')
    parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='use cosine lr schedule')

    parser.add_argument('--alpha_weight', default=5.0, type=float, help='alpha weight')
    parser.add_argument('--beta_weight', default=0.01, type=float, help='beta weight')

    parser.add_argument('--alpha_t', default=5, type=float)
    parser.add_argument('--beta_t', default=25, type=float)


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    seed_torch(SEED)

    args.pro_weight_range = [float(item) for item in args.pro_weight_range.split(',')]
    args.lr_decay_epochs = [int(item) for item in args.lr_decay_epochs.split(',')]

    current_folder = "./"
    args.output_dir = os.path.join(current_folder, args.exp_dir, args.PLL_method, args.dataset, args.dset, "rate:" + str(args.partial_rate))
    args.fea_path = os.path.join(args.output_dir, args.fea_path)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.fea_path):
        os.system('mkdir -p ' + args.fea_path)

    args.out_file = open(os.path.join(args.output_dir, args.file + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    print(args)

    # create dataset
    dset_loaders = load_idx(args)
    print("Finish load datasets!\n")

    # create model
    model = DiDA(args, Base_ResNet)
    model = model.cuda()


    if args.model_path != None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])


    # set optimizer
    optimizer = torch.optim.SGD(model.encoder.encoder.parameters(),
                                args.lr,
                                momentum=0.9,
                                weight_decay=1e-5)
    optimizer_fc = torch.optim.SGD(model.encoder.fc.parameters(),
                                   lr=1e-4,
                                   momentum=0.9,
                                   weight_decay=0)

    # init confidence
    confidence_src, confidence_tar = confidence_init(dset_loaders)

    # set loss functions
    KL_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    CE_criterion = CE_loss(args)

    best_map_s2t = 0.
    map_s2t = trainer.test_map(args, dset_loaders['source_te'], dset_loaders['target_te'], model)
    print("First_mAP:{:.3f}".format(map_s2t))


    if args.epoch > 0:
        for epoch in range(args.start_epoch, args.epoch):
            print("---train: epoch[{}]/[{}]".format(epoch, args.epoch - 1))
            start_time = time.time()

            adjust_learning_rate(args, optimizer, epoch, args.lr)
            adjust_learning_rate(args, optimizer_fc, epoch, 1e-4)

            # train
            loss, confidence_src, confidence_tar = trainer.train_model(args, model, dset_loaders["source_tr"],
                                                                        dset_loaders["target_tr"], KL_criterion,
                                                                        CE_criterion, optimizer, optimizer_fc,
                                                                        confidence_src, confidence_tar, epoch)
            model.set_prototype_update_weight(epoch, args)

            # test mAP
            map_s2t = trainer.test_map(args, dset_loaders['source_te'], dset_loaders['target_te']
                                       , model, path_for_save_features=args.fea_path + "/features.h5")
            is_best = False
            if map_s2t > best_map_s2t:
                is_best = True
                best_map_s2t = map_s2t


            with open(os.path.join(args.output_dir, 'train_result.log'), 'a') as f:
                f.write(
                    f"Epoch:{epoch},map:{map_s2t:.3f},Best_map:{best_map_s2t:.3f},loss:{loss:.7f}\n")


            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'dset': args.dset,
                'dataset': args.dataset,
                'state_dict': model.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.output_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.output_dir))


            end_time = time.time()
            cost_time = (end_time - start_time) / 60
            str_s2t = "Task:{}-{}, epoch:{}, BestMap:{:.3f}, LastMap:{:.3f}, cost_time:{:.2f} min\n".format(
                args.PLL_method,
                args.dset,
                epoch,
                best_map_s2t,
                map_s2t,
                cost_time)
            print(str_s2t)

    # end
    print("end")


