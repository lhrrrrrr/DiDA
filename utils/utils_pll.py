import numpy as np
import torch
import torch.nn.functional as F



def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    # if partial_rate==0, return true label
    train_labels = torch.from_numpy(train_labels)
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    num_class = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    num_label = train_labels.shape[0]

    partialY = torch.zeros(num_label, num_class)
    partialY[torch.arange(num_label), train_labels] = 1.0
    if partial_rate == 0:
        return partialY
    transition_matrix =  np.eye(num_class)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate

    random_n = np.random.uniform(0, 1, size=(num_label, num_class))

    for j in range(num_label):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    # print("Finish Generating Candidate Label Sets!\n")
    return partialY



def confidence_init(dset_loaders):
    all_plLabel_src = dset_loaders["source_tr_nobatch"].labels
    all_plLabel_tar = dset_loaders["target_tr_nobatch"].labels

    temp_src = all_plLabel_src.sum(dim=1).unsqueeze(1).repeat(1, all_plLabel_src.shape[1])
    confidence_src = all_plLabel_src.float() / temp_src
    confidence_src = confidence_src.cuda()

    temp_tar = all_plLabel_tar.sum(dim=1).unsqueeze(1).repeat(1, all_plLabel_tar.shape[1])
    confidence_tar = all_plLabel_tar.float() / temp_tar
    confidence_tar = confidence_tar.cuda()
    print(f"confidence done!")
    return confidence_src, confidence_tar



def confidence_update(confidence, logits, plLabel, indexes):
    logits = F.softmax(logits, dim=1)
    confidence[indexes, :] = plLabel * logits
    base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
    confidence = confidence / base_value
    return confidence



