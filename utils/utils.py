import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score

def cnt_sample_num(labeled_loader, num_classes):
    num = torch.zeros(num_classes).cuda()
    for _, (_, data) in enumerate(labeled_loader):
        label = data['label']
        num += torch.tensor([(label==i).sum() for i in range(num_classes)]).cuda()

    return num

def statis_acc(model, global_param, local_param, dataloader):
    model.eval()
    with torch.no_grad():
        pred_list = torch.tensor([]).cuda()
        label_list = torch.tensor([]).cuda()
        model.load_state_dict(global_param)
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()

            logit = model(image)[0]    

            pred_list = torch.cat((pred_list, torch.argmax(logit, dim=1)))
            label_list = torch.cat((label_list, label))

        global_acc = balanced_accuracy_score(label_list.cpu().numpy(), pred_list.cpu().numpy())

        pred_list = torch.tensor([]).cuda()
        label_list = torch.tensor([]).cuda()
        model.load_state_dict(local_param)
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()

            logit = model(image)[0]    

            pred_list = torch.cat((pred_list, torch.argmax(logit, dim=1)))
            label_list = torch.cat((label_list, label))

        local_acc = balanced_accuracy_score(label_list.cpu().numpy(), pred_list.cpu().numpy())
    return global_acc, local_acc
    
