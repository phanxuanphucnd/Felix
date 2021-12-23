import torch
import torch.nn as nn
from tqdm.auto import tqdm
from seqeval.metrics import f1_score, classification_report
import tagging.constants as constants
# from sklearn.metrics import classification_report
import numpy as np
from collections import Counter


class DiceLoss(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def onehot_initialization_v2(self, a, num_label):
        ncols = num_label
        out = np.zeros((a.size, ncols), dtype=np.uint8)
        out[np.arange(a.size), a.ravel()] = 1
        out.shape = a.shape + (ncols,)
        return out


    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch: [batch, seq_len, num_label]
        target: tensor with first dimension as batch: [batch, seq_len] or [batch, seq_len, num_label]
        """
        # print(pred.size(), target.size())
        if pred.size() != target.size():
            size = list(pred.size())
            target_ = target.cpu().numpy()
            target_ = self.onehot_initialization_v2(target_, size[1])
            
            # print(target_.size)
            # if target_.size != np.prod(size):
            #     print(np.unique(target_))
            #     with open('outfile.txt','wb') as f:
            #         for line in target_:
            #             np.savetxt(f, line, fmt='%.2f')
            # # target = target_.reshape(size)
            
            target = torch.tensor(target_).to(self.device)

        smooth = 1.

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def tagging_evaluate(y_true_tag, y_pred_tag, y_true_point, y_pred_point):
    pres, trues = [], []
    for sent_true, sent_out in zip(y_true_tag, y_pred_tag):
        tmp = ["B-"+constants.ID2TAGS[i] if constants.ID2TAGS[i] != "PAD" else "O" for i in sent_true if i != constants.TAGS2ID["PAD"]]
        trues.append(tmp)
        pres.append(["B-"+constants.ID2TAGS[i] if constants.ID2TAGS[i] != "PAD" else "O" for i in sent_out[:len(tmp)]])
    # tag_f1 = f1_score(trues, pres)
    # trues = sum(trues, [])
    # pres = sum(pres, [])
    print(Counter(sum(pres, [])))
    # print(Counter(y_pred_tag))
    report = classification_report(trues, pres, output_dict=True, zero_division=0)
    tag_f1 = report["macro avg"]["f1-score"]

    # print(report)
    print(classification_report(trues, pres, zero_division=0))
    print("F1 TAGGING:", tag_f1)


    pres, trues = [], []
    for sent_true, sent_out in zip(y_true_point, y_pred_point):
        tmp = ["B-"+str(i) for i in sent_true if i != -100]
        trues.append(tmp)
        pres.append(["B-"+str(i) for i in sent_out[:len(tmp)]])
    # trues = sum(trues, [])
    # pres = sum(pres, [])
    report = classification_report(trues, pres, output_dict=True, zero_division=0)
    point_f1 = report["macro avg"]["f1-score"]
    # print(classification_report(trues, pres))
    # point_f1 = f1_score(trues, pres)
    print("F1 POINTER:", point_f1)

    return (tag_f1+point_f1)/2


def train_fn(
    dataloader, model, tag_criterion, pointer_criterion, optimizer, scheduler, device="cuda", accu_step=1
):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, total=len(dataloader), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')
    for i, (batch) in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        input_mask_words = batch["input_mask_words"].to(device)
        word_matrix = batch["word_matrix"].to(device)
        tag_labels = batch["tag_labels"].to(device)
        point_labels = batch["point_labels"].to(device)
        
        inputs = (input_ids, tag_labels, point_labels, word_matrix, input_mask, input_mask_words)

        tag_logits, point_logits = model(inputs)

        # Loss calculate

        tag_logits = torch.transpose(tag_logits, 2, 1)  # loss
        tag_loss = tag_criterion(tag_logits, tag_labels)

        point_logits = torch.transpose(point_logits, 2, 1)  # loss
        point_loss = pointer_criterion(point_logits, point_labels)

        loss = tag_loss + point_loss

        # Loss backward
        loss.backward()
        if (i + 1) % accu_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()

        total_loss += loss.item()

    total_loss /= len(dataloader)

    return total_loss



def validation_fn(dataloader, model, tag_criterion, pointer_criterion, device="cuda"):
    model.eval()
    total_loss = 0
    tag_pres, tag_golds, point_pres, point_golds = [], [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')
        for i, (batch) in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            input_mask_words = batch["input_mask_words"].to(device)
            word_matrix = batch["word_matrix"].to(device)
            tag_labels = batch["tag_labels"].to(device)
            point_labels = batch["point_labels"].to(device)
            inputs = (input_ids, tag_labels, point_labels, word_matrix, input_mask, input_mask_words)

            tag_logits, point_logits = model(inputs)

            # Loss calculate

            tag_logits = torch.transpose(tag_logits, 2, 1)  # loss
            tag_loss = tag_criterion(tag_logits, tag_labels)

            point_logits = torch.transpose(point_logits, 2, 1)  # loss
            point_loss = pointer_criterion(point_logits, point_labels)

            loss = tag_loss + point_loss

            total_loss += loss.item()

            # Evaluate
            tag_logits = torch.transpose(tag_logits, 2, 1)
            tag_outputs = torch.argmax(tag_logits, dim=-1)
            tag_outputs = tag_outputs.detach().cpu().numpy()
            
            tag_labels = tag_labels.detach().cpu().numpy()
            tag_pres.extend(tag_outputs)
            tag_golds.extend(tag_labels)

            point_logits = torch.transpose(point_logits, 2, 1)
            point_outputs = torch.argmax(point_logits, dim=-1)
            point_outputs = point_outputs.detach().cpu().numpy()
            
            point_labels = point_labels.detach().cpu().numpy()
            point_pres.extend(point_outputs)
            point_golds.extend(point_labels)

        entity_f1 = tagging_evaluate(tag_golds, tag_pres, point_golds, point_pres)

        print("F1 score: ", entity_f1)

        total_loss /= len(dataloader)

        return total_loss, entity_f1