import sys
sys.path.append("..")
from utils.utils import *
from utils.utils_pll import *
from utils.utils_loss import *
import h5py




def train_model(args, model, source_loader, target_loader, KL_criterion, CE_criterion, optimizer, optimizer_fc, confidence_src, confidence_tar, epoch):

    alpha_t = update_loss_weight(epoch, args.alpha_t)
    beta_t = update_loss_weight(epoch, args.beta_t)

    model.train()
    for batch_idx, (inputs_src_w, inputs_src_s, plLabel_src, _, domainIDs_src, indexes_src) in enumerate(source_loader):
        try:
            inputs_tar_w, inputs_tar_s, plLabel_tar, _, domainIDs_tar, indexes_tar = next(target_loader_unl_iter)
        except:
            target_loader_unl_iter = iter(target_loader)
            inputs_tar_w, inputs_tar_s, plLabel_tar, _, domainIDs_tar, indexes_tar = next(target_loader_unl_iter)

        inputs_src_w, inputs_src_s, plLabel_src, domainIDs_src, indexes_src = inputs_src_w.cuda(), inputs_src_s.cuda(), plLabel_src.cuda(), domainIDs_src.cuda(), indexes_src.cuda()
        inputs_tar_w, inputs_tar_s, plLabel_tar, domainIDs_tar, indexes_tar = inputs_tar_w.cuda(), inputs_tar_s.cuda(), plLabel_tar.cuda(), domainIDs_tar.cuda(), indexes_tar.cuda()

        logits_z_src, logits_z1_src, prototypes_src, prot_scores_src, pseudo_labels_src, feats_q_src, feats_q1_src = model(inputs_src_w, inputs_src_s, plLabel_src, 0, args)
        logits_z_tar, logits_z1_tar, prototypes_tar, prot_scores_tar, pseudo_labels_tar, feats_q_tar, feats_q1_tar = model(inputs_tar_w, inputs_tar_s, plLabel_tar, 1, args)

        logits_z_src_log_soft = torch.log_softmax(logits_z_src, dim=-1)
        logits_z1_src_log_soft = torch.log_softmax(logits_z1_src, dim=-1)
        logits_z_tar_log_soft = torch.log_softmax(logits_z_tar, dim=-1)
        logits_z1_tar_log_soft = torch.log_softmax(logits_z1_tar, dim=-1)


        loss_cls_src = KL_criterion(logits_z_src_log_soft, confidence_src[indexes_src]) + KL_criterion(logits_z1_src_log_soft, confidence_src[indexes_src])
        loss_cls_tar = KL_criterion(logits_z_tar_log_soft, confidence_tar[indexes_tar]) + KL_criterion(logits_z1_tar_log_soft, confidence_tar[indexes_tar])
        loss_cls = loss_cls_src + loss_cls_tar

        loss_psu_src = CE_criterion(prot_scores_src)
        loss_psu_tar = CE_criterion(prot_scores_tar)
        loss_psu = loss_psu_src + loss_psu_tar

        loss_pda_src = compute_pda_loss(feats_q_src, feats_q1_src, prototypes_src, prototypes_tar)
        loss_pda_tar = compute_pda_loss(feats_q_tar, feats_q1_tar, prototypes_src, prototypes_tar)
        loss_pda = loss_pda_src + loss_pda_tar


        # loss
        loss = loss_cls + alpha_t * args.alpha_weight * loss_psu + beta_t * args.beta_weight * loss_pda


        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_fc.step()

        # update confidence
        confidence_src = confidence_update(confidence_src, logits_z_src.clone().detach(), plLabel_src, indexes_src)
        confidence_tar = confidence_update(confidence_tar, logits_z_tar.clone().detach(), plLabel_tar, indexes_tar)


        if batch_idx % 30 == 0:
            print(f"Step [{batch_idx}/{len(source_loader)}] \n"
                  f"loss: {loss:.3f}  loss_cls_src: {loss_cls_src:.3f} loss_cls_tar: {loss_cls_tar:.3f} \n"
                       f"             loss_psu_src: {loss_psu_src:.3f} loss_psu_tar: {loss_psu_tar:.3f} \n"
                       f"             loss_pda_src: {loss_pda_src:.3f} loss_pda_tar: {loss_pda_tar:.3f} \n")

    return loss, confidence_src, confidence_tar





def test_map(args, src_loader, tgt_loader, model, path_for_save_features=None):
    model.eval()

    # print('Prepare Gallery Features.....')
    features_gallery, gt_labels_gallery = get_features(src_loader, model)

    # print('Prepare Query Features of Target Domain.....')
    print('---test:')
    features_query, gt_labels_query = get_features(tgt_loader, model)

    if path_for_save_features:
        with h5py.File(path_for_save_features, 'w') as hf:
            hf.create_dataset('features_gallery', data=features_gallery)
            hf.create_dataset('gt_labels_gallery', data=gt_labels_gallery)
            hf.create_dataset('features_query', data=features_query)
            hf.create_dataset('gt_labels_query', data=gt_labels_query)

    map_t = cal_map_sda(features_query, gt_labels_query,
                        features_gallery, gt_labels_gallery)
    return map_t






