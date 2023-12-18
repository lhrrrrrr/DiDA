import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import *
from utils.utils_pll import generate_uniform_cv_candidate_labels
from utils.randaugment import RandomAugment


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [val[:val.rfind(' ')] for val in image_list]
            labels = [int(val[val.rfind(' ')+1:].strip()) for val in image_list]
            return images, np.array(labels)
        else:
            images = [val.split()[0] for val in image_list]
            labels = [int(val.split()[1]) for val in image_list]
            return images, np.array(labels)
    return images


class ImageList_idx(Dataset):
    def __init__(self,
                 args,
                 image_list,
                 labels=None,
                 transform=None,
                 mode='RGB',
                 domain_id=0,
                 partial_label=False):
        self.imgs, self.labels = make_dataset(image_list, labels)
        if partial_label:
            self.labels = generate_uniform_cv_candidate_labels(self.labels, args.partial_rate)

        self.domain_id = domain_id
        self.transform = transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index], self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)
        return img1, target, self.domain_id, index

    def __len__(self):
        return len(self.imgs)


class Aug_ImageList_idx(Dataset):
    def __init__(self,
                 args,
                 image_list,
                 labels=None,
                 mode='RGB',
                 domain_id=0,
                 partial_label=True):
        self.imgs, self.labels = make_dataset(image_list, labels)
        self.true_labels = generate_uniform_cv_candidate_labels(self.labels, 0)
        if partial_label:
            self.labels = generate_uniform_cv_candidate_labels(self.labels, args.partial_rate)

        self.domain_id = domain_id
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.weak_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                ####################
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                ####################
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        self.strong_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                ####################
                RandomAugment(3, 5),
                ####################
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path, target, true_label = self.imgs[index], self.labels[index], self.true_labels[index]

        img = self.loader(path)
        img_w = self.weak_transform(img)
        img_s = self.strong_transform(img)

        return img_w, img_s, target, true_label, self.domain_id, index

    def __len__(self):
        return len(self.imgs)


def load_idx(args):
    train_bs = args.batch_size

    if args.dataset == 'office_home':
        assert args.class_num == 65
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'Art'
        elif ss == 'c':
            s = 'Clipart'
        elif ss == 'p':
            s = 'Product'
        elif ss == 'r':
            s = 'Real_World'
        else:
            raise NotImplementedError

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'
        else:
            raise NotImplementedError

        s_tr_path = './data_split/office-home/' + s + '_train.txt'
        s_ts_path = './data_split/office-home/' + s + '_test.txt'
        t_tr_path = './data_split/office-home/' + t + '_train.txt'
        t_ts_path = './data_split/office-home/' + t + '_test.txt'

        if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
            s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()

        if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
            t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()

    elif args.dataset == 'image_CLEF':
        assert args.class_num == 12
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'b':
            s = 'bList'
        elif ss == 'c':
            s = 'cList'
        elif ss == 'i':
            s = 'iList'
        elif ss == 'p':
            s = 'pList'
        else:
            raise NotImplementedError
        if tt == 'b':
            t = 'bList'
        elif tt == 'c':
            t = 'cList'
        elif tt == 'i':
            t = 'iList'
        elif tt == 'p':
            t = 'pList'
        else:
            raise NotImplementedError

        s_tr_path = './data_split/image_CLEF/' + s + '_train.txt'
        s_ts_path = './data_split/image_CLEF/' + s + '_test.txt'
        t_tr_path = './data_split/image_CLEF/' + t + '_train.txt'
        t_ts_path = './data_split/image_CLEF/' + t + '_test.txt'

        if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
            s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
            t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()


    elif args.dataset == 'office31':
        assert args.class_num == 31
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'amazon_31'
        elif ss == 'd':
            s = 'dslr_31'
        elif ss == 'w':
            s = 'webcam_31'
        else:
            raise NotImplementedError
        if tt == 'a':
            t = 'amazon_31'
        elif tt == 'd':
            t = 'dslr_31'
        elif tt == 'w':
            t = 'webcam_31'
        else:
            raise NotImplementedError

        s_tr_path = './data_split/office31/' + s + '_train.txt'
        s_ts_path = './data_split/office31/' + s + '_test.txt'
        t_tr_path = './data_split/office31/' + t + '_train.txt'
        t_ts_path = './data_split/office31/' + t + '_test.txt'

        if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
            s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
            t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()



    prep_dict = {}
    prep_dict['source'] = image_train()
    prep_dict['target'] = image_target()
    prep_dict['test'] = image_test()

    train_source = Aug_ImageList_idx(args, s_tr, domain_id=0, partial_label=True)
    test_source = ImageList_idx(args, s_ts, transform=prep_dict['test'], domain_id=0, partial_label=False)
    train_target = Aug_ImageList_idx(args, t_tr, domain_id=1, partial_label=True)
    test_target = ImageList_idx(args, t_ts, transform=prep_dict['test'], domain_id=1, partial_label=False)

    dset_loaders = {}
    dset_loaders["source_tr_nobatch"] = train_source
    dset_loaders["target_tr_nobatch"] = train_target

    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=True)
    dset_loaders["source_te"] = DataLoader(test_source,
                                            batch_size=train_bs * 3,
                                            shuffle=False,
                                            num_workers=args.worker,
                                            drop_last=False)

    dset_loaders["target_tr"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target_te"] = DataLoader(test_target,
                                        batch_size=train_bs * 3,
                                        shuffle=False,
                                        num_workers=args.worker,
                                        drop_last=False)
    return dset_loaders



