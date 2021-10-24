from utils import MolecularImageDataset, collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from unet import UNet
import matplotlib.pyplot as plt
import os
from meter import AverageMeter
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='multi-gpu-training')
parser.add_argument('--data',metavar='dir',default='train_data',help='path to dataset')
parser.add_argument('--epoch',default=15,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-b','--batch-size',default=60,type=int,metavar='N',help='mini batch size')
parser.add_argument('--lr','--learning-rate',default=2.5e-4,type=float,
                    metavar='LR',help='initial learning rate',dest='lr')


def reduce_mean(tensor,nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main():
    print('start')
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    print(args.nprocs)
    mp.spawn(main_worker,nprocs=args.nprocs,args=(args.nprocs,args))
    print('done')


def main_worker(local_rank,nprocs,args):
    print('enter')
    args.local_rank = local_rank

    dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:12345',
                            world_size=nprocs,rank=local_rank)

    df = pd.read_csv('../data/indigo_train_data/processed_chembl.csv')
    train_df1 = df[:80000].copy().reset_index(drop=True)
    test_df1 = df[90000:91000].copy().reset_index(drop=True)

    df = pd.read_csv('../data/train_data/processed_chembl.csv')
    train_df2 = df[:80000].copy().reset_index(drop=True)
    test_df2 = df[90000:91000].copy().reset_index(drop=True)

    train_df = pd.concat([train_df1, train_df2], axis=0).reset_index(drop=True)
    test_df = pd.concat([test_df1, test_df2], axis=0).reset_index(drop=True)

    epoch_nums = 15
    amount = 0.1

    train_dataset = MolecularImageDataset(train_df, amount=amount)
    test_dataset = MolecularImageDataset(test_df, amount=amount)

    # imgs, atom_targets, atom_types, atom_charges, Hs, bond_targets, bond_types, bond_rhos, bond_omega_types = train_dataset[0]
    # plt.subplot(131)
    # plt.imshow(imgs[0])
    # plt.subplot(132)
    # plt.imshow(atom_targets[0])
    # plt.subplot(133)
    # plt.imshow(bond_targets[0])
    # for x, y in zip(*((bond_targets[0]==1).nonzero())):
    #     rho = bond_rhos[:, x, y]
    #     omega = bond_omega_types[:, x, y].argmax() * (2*np.pi/60) + np.pi/60 - np.pi/2
    #
    #     plt.plot([y - rho* np.sin(omega), y +  rho*np.sin(omega)], [x -  rho*np.cos(omega), x +  rho*np.cos(omega)])
    #
    # plt.show()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=collate_fn,num_workers=3,prefetch_factor=10,sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size,collate_fn=collate_fn,num_workers=3,prefetch_factor=10,sampler=test_sampler)

    model = UNet(in_channels=1, heads=[1, 14, 3, 2, 1, 360, 60, 60])
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if local_rank==0:
        torch.save(model.state_dict(),'temp.pth')

    dist.barrier()

    model.load_state_dict(torch.load('temp.pth',map_location={'cuda:0':'cuda:{}'.format(local_rank)}))


    optimizer = optim.Adam(model.parameters(), lr=2.5e-4, weight_decay=1e-8)

    torch.cuda.empty_cache()

    train_atom_targets_precision = AverageMeter()
    train_atom_targets_recall = AverageMeter()
    train_atom_targets_precision3 = AverageMeter()
    train_atom_targets_recall3 = AverageMeter()

    train_atom_types_acc = AverageMeter()
    train_atom_charges_acc = AverageMeter()
    train_atom_hs_acc = AverageMeter()

    train_bond_targets_precision = AverageMeter()
    train_bond_targets_recall = AverageMeter()
    train_bond_targets_precision3 = AverageMeter()
    train_bond_targets_recall3 = AverageMeter()

    train_bond_types_acc = AverageMeter()
    train_bond_rhos_mae = AverageMeter()

    train_bond_omega_precision = AverageMeter()
    train_bond_omega_recall = AverageMeter()
    train_bond_omega_precision3 = AverageMeter()
    train_bond_omega_recall3 = AverageMeter()

    for epoch in range(epoch_nums):
        if epoch == int(epoch_nums / 3):
            optimizer = optim.Adam(model.parameters(), lr=2.5e-5, weight_decay=1e-8)
        for i, (imgs, atom_targets, atom_types, atom_charges, atom_hs,
                bond_targets, bond_types, bond_rhos, bond_omega_types) in enumerate(train_dataloader):

            model.train()
            imgs = imgs.to(device)
            atom_targets, atom_types, atom_charges, atom_hs = atom_targets.to(device), atom_types.to(
                device), atom_charges.to(device), atom_hs.to(device)
            bond_targets, bond_types, bond_rhos, bond_omega_types = bond_targets.to(device), bond_types.to(
                device), bond_rhos.to(device), bond_omega_types.to(device)

            atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred, bond_targets_pred, bond_types_pred, bond_rhos_pred, bond_omega_types_pred = model(
                imgs)
            atom_targets_pred = torch.clamp(torch.sigmoid(atom_targets_pred), 1e-5, 1 - 1e-5)
            atom_types_pred = torch.clamp(torch.softmax(atom_types_pred, dim=1), 1e-5, 1 - 1e-5)
            atom_charges_pred = torch.clamp(torch.softmax(atom_charges_pred, dim=1), 1e-5, 1 - 1e-5)
            atom_hs_pred = torch.clamp(torch.softmax(atom_hs_pred, dim=1), 1e-5, 1 - 1e-5)

            bond_targets_pred = torch.clamp(torch.sigmoid(bond_targets_pred), 1e-5, 1 - 1e-5)
            bond_types_pred = torch.clamp(torch.softmax(bond_types_pred.view(-1, 6, 60, 128, 128), dim=1), 1e-5, 1 - 1e-5)

            bond_omega_types_pred = torch.clamp(torch.sigmoid(bond_omega_types_pred), 1e-5, 1 - 1e-5)

            bond_rhos_pred = torch.abs(bond_rhos_pred)

            atom_targets_loss = torch.sum(
                -(atom_targets == 1).float() * (1 - atom_targets_pred) ** 2 * torch.log(atom_targets_pred)
                - (1 - atom_targets) ** 4 * (atom_targets_pred) ** 2 * torch.log(1 - atom_targets_pred)) / torch.sum(
                atom_targets == 1)
            atom_types_loss = torch.sum(-atom_types * (1 - atom_types_pred) ** 2 * torch.log(atom_types_pred)) / torch.sum(
                atom_types)

            atom_charges_loss = torch.sum(
                -atom_charges * (1 - atom_charges_pred) ** 2 * torch.log(atom_charges_pred)) / torch.sum(atom_charges)

            # atom_hs_weights = torch.FloatTensor([0.25,1]).reshape(1,2,1,1).to(device)
            atom_hs_loss = torch.sum(-atom_hs * (1 - atom_hs_pred) ** 2 * torch.log(atom_hs_pred)) / (
                        torch.sum(atom_hs) + 0.1)

            bond_targets_loss = torch.sum(
                -(bond_targets == 1).float() * (1 - bond_targets_pred) ** 2 * torch.log(bond_targets_pred)
                - (1 - bond_targets) ** 4 * (bond_targets_pred) ** 2 * torch.log(1 - bond_targets_pred)) / torch.sum(
                bond_targets == 1)

            bond_types_loss = torch.sum(-bond_types * (1 - bond_types_pred) ** 2 * torch.log(bond_types_pred)) / torch.sum(
                bond_types)

            bond_rhos_loss = torch.sum(torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / torch.sum(
                bond_types)

            bond_omega_types_loss = -torch.sum(torch.sum(bond_omega_types, dim=1, keepdim=True) * (
                        (bond_omega_types == 1) * ((1 - bond_omega_types_pred) ** 2) * torch.log(bond_omega_types_pred) +
                        (1 - bond_omega_types) ** 4 * (bond_omega_types_pred ** 2) * torch.log(
                    1 - bond_omega_types_pred))) / torch.sum(bond_omega_types)

            atom_targets_loss *= torch.exp(-model.module.s[0]) + model.module.s[0]
            bond_targets_loss *= torch.exp(-model.module.s[1]) + model.module.s[1]
            atom_types_loss *= torch.exp(-model.module.s[2]) + model.module.s[2]
            atom_charges_loss *= torch.exp(-model.module.s[3]) + model.module.s[3]

            bond_types_loss *= torch.exp(-model.module.s[4]) + model.module.s[4]
            bond_rhos_loss *= 0.5 * torch.exp(-model.module.s[6]) + model.module.s[6]
            bond_omega_types_loss *= torch.exp(-model.module.s[7]) + model.module.s[7]
            atom_hs_loss *= torch.exp(-model.module.s[9]) + model.module.s[9]

            loss = atom_targets_loss + bond_targets_loss + atom_types_loss + atom_charges_loss + bond_types_loss + bond_rhos_loss + bond_omega_types_loss + atom_hs_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ####metrics

            temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3,
                                                  stride=1, padding=1)
            atom_targets_pred = ((temp == atom_targets_pred) * (atom_targets_pred > 0.25)).float()

            temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3,
                                                  stride=1, padding=1)
            bond_targets_pred = ((temp == bond_targets_pred) * (bond_targets_pred > 0.25)).float()

            atom_targets = (atom_targets == 1).float()
            train_atom_targets_precision.update(
                ((atom_targets_pred * atom_targets).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
                atom_targets_pred.sum().cpu().detach().numpy())
            train_atom_targets_precision3.update(
                ((atom_targets_pred * torch.nn.functional.max_pool2d(atom_targets, kernel_size=3, padding=1,
                                                                     stride=1)).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
                atom_targets_pred.sum().cpu().detach().numpy())

            train_atom_targets_recall.update(
                ((atom_targets * atom_targets_pred).sum() / atom_targets.sum()).cpu().detach().numpy(),
                atom_targets.sum().cpu().detach().numpy())
            train_atom_targets_recall3.update(
                ((atom_targets * torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3, padding=1,
                                                                stride=1)).sum() / atom_targets.sum()).cpu().detach().numpy(),
                atom_targets.sum().cpu().detach().numpy())

            train_atom_types_acc.update((torch.sum(torch.sum(atom_types, dim=1) * ((atom_types.argmax(1) ==
                                                                                    atom_types_pred.argmax(
                                                                                        1)).float())) / torch.sum(
                atom_types)).cpu().detach().numpy(), torch.sum(atom_types).cpu().detach().numpy())

            train_atom_charges_acc.update((torch.sum(torch.sum(atom_charges, dim=1) * ((atom_charges.argmax(1) ==
                                                                                        atom_charges_pred.argmax(
                                                                                            1)).float())) / torch.sum(
                atom_charges)).cpu().detach().numpy(), torch.sum(atom_charges).cpu().detach().numpy())

            train_atom_hs_acc.update((torch.sum(torch.sum(atom_hs, dim=1) * ((atom_hs.argmax(1) ==
                                                                              atom_hs_pred.argmax(1)).float())) / (
                                                  0.01 + torch.sum(atom_hs))).cpu().detach().numpy(),
                                     (0.01 + torch.sum(atom_hs)).cpu().detach().numpy())

            bond_targets = (bond_targets == 1).float()

            train_bond_targets_precision.update(
                ((bond_targets_pred * bond_targets).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
                bond_targets_pred.sum().cpu().detach().numpy())
            train_bond_targets_precision3.update(
                ((bond_targets_pred * torch.nn.functional.max_pool2d(bond_targets, kernel_size=3, padding=1,
                                                                     stride=1)).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
                bond_targets_pred.sum().cpu().detach().numpy())

            train_bond_targets_recall.update(
                ((bond_targets * bond_targets_pred).sum() / bond_targets.sum()).cpu().detach().numpy(),
                bond_targets.sum().cpu().detach().numpy())
            train_bond_targets_recall3.update(
                ((bond_targets * torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3, padding=1,
                                                                stride=1)).sum() / bond_targets.sum()).cpu().detach().numpy(),
                bond_targets.sum().cpu().detach().numpy())

            train_bond_types_acc.update((torch.sum(
                torch.sum(bond_types, dim=1) * ((bond_types.argmax(1) == bond_types_pred.argmax(1)).float())) / torch.sum(
                bond_types)).cpu().detach().numpy(), torch.sum(bond_types).cpu().detach().numpy())

            train_bond_rhos_mae.update((torch.sum(torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / \
                                        torch.sum(bond_types)).cpu().detach().numpy(),
                                       torch.sum(bond_types).cpu().detach().numpy())

            temp = torch.cat([bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]],
                             dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)

            temp = ((torch.nn.functional.max_pool1d(temp, stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
                                                                                                      60).permute(0, 3, 1,
                                                                                                                  2) == bond_omega_types_pred) * \
                    (bond_omega_types_pred > 0.25)).float() * bond_targets

            bond_omega_types = (bond_omega_types == 1)
            train_bond_omega_precision.update((((bond_omega_types) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                                              temp.sum().cpu().detach().numpy())

            temp2 = torch.cat([temp[:, 59:], temp, temp[:, :1]],
                              dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
            temp2 = (torch.nn.functional.max_pool1d(temp2.float(), stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
                                                                                                               60).permute(
                0, 3, 1, 2)).float()

            train_bond_omega_recall3.update(
                (((bond_omega_types) * temp2).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
                bond_omega_types.sum().cpu().detach().numpy())

            train_bond_omega_recall.update(
                (((bond_omega_types) * temp).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
                bond_omega_types.sum().cpu().detach().numpy())

            temp3 = torch.cat([bond_omega_types[:, 59:], bond_omega_types, bond_omega_types[:, :1]],
                              dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
            temp3 = (torch.nn.functional.max_pool1d(temp3.float(), stride=1, kernel_size=3, padding=0).reshape(-1, 128, 128,
                                                                                                               60).permute(
                0, 3, 1, 2)).float()
            train_bond_omega_precision3.update((((temp3) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                                               temp.sum().cpu().detach().numpy())

            if i % 100 == 0:
                model.eval()
                print(epoch, i)
                print(loss.cpu().detach().numpy())
                with torch.no_grad():
                    if local_rank==0:
                        print('loss_________________')
                        print('atom_targets_loss:', atom_targets_loss.cpu().detach().numpy())
                        print('atom_types_loss:', atom_types_loss.cpu().detach().numpy())
                        print('atom_charges_loss:', atom_charges_loss.cpu().detach().numpy())
                        print('atom_hs_loss:', atom_hs_loss.cpu().detach().numpy())
                        print('bond_targets_loss:', bond_targets_loss.cpu().detach().numpy())
                        print('bond_types_loss:', bond_types_loss.cpu().detach().numpy())

                        print('bond_rho_loss:', bond_rhos_loss.cpu().detach().numpy())
                        print('bond_omega_types:', bond_omega_types_loss.cpu().detach().numpy())
                        print('train_________________')

                        print('atom_target_precision:', train_atom_targets_precision.avg)
                        print('atom_target_recall:', train_atom_targets_recall.avg)
                        print('atom_target_precision3:', train_atom_targets_precision3.avg)
                        print('atom_target_recall3:', train_atom_targets_recall3.avg)

                        print('atom_types_acc:', train_atom_types_acc.avg)
                        print('atom_charges_acc:', train_atom_charges_acc.avg)
                        print('atom_hs_acc:', train_atom_hs_acc.avg)

                        print('bond_target_precision:', train_bond_targets_precision.avg)
                        print('bond_target_recall:', train_bond_targets_recall.avg)
                        print('bond_target_precision3:', train_bond_targets_precision3.avg)
                        print('bond_target_recall3:', train_bond_targets_recall3.avg)

                        print('bond_types_acc:', train_bond_types_acc.avg)
                        print('bond_rhos_mae:', train_bond_rhos_mae.avg)

                        print('bond_omega_precision:', train_bond_omega_precision.avg)
                        print('bond_omega_recall:', train_bond_omega_recall.avg)
                        print('bond_omega_precision3:', train_bond_omega_precision3.avg)
                        print('bond_omega_recall3:', train_bond_omega_recall3.avg)

                    train_atom_targets_precision.reset()
                    train_atom_targets_recall.reset()
                    train_atom_targets_precision3.reset()
                    train_atom_targets_recall3.reset()

                    train_atom_types_acc.reset()
                    train_atom_charges_acc.reset()
                    train_atom_hs_acc.reset()

                    train_bond_targets_precision.reset()
                    train_bond_targets_recall.reset()
                    train_bond_targets_precision3.reset()
                    train_bond_targets_recall3.reset()

                    train_bond_types_acc.reset()
                    train_bond_rhos_mae.reset()

                    train_bond_omega_precision.reset()
                    train_bond_omega_recall.reset()
                    train_bond_omega_precision3.reset()
                    train_bond_omega_recall3.reset()

                    train_steps = 0

                    test_atom_targets_precision = AverageMeter()
                    test_atom_targets_recall = AverageMeter()
                    test_atom_targets_precision3 = AverageMeter()
                    test_atom_targets_recall3 = AverageMeter()

                    test_atom_types_acc = AverageMeter()
                    test_atom_charges_acc = AverageMeter()
                    test_atom_hs_acc = AverageMeter()

                    test_bond_targets_precision = AverageMeter()
                    test_bond_targets_recall = AverageMeter()
                    test_bond_targets_precision3 = AverageMeter()
                    test_bond_targets_recall3 = AverageMeter()

                    test_bond_types_acc = AverageMeter()

                    test_bond_rhos_mae = AverageMeter()

                    test_bond_omega_precision = AverageMeter()
                    test_bond_omega_recall = AverageMeter()
                    test_bond_omega_precision3 = AverageMeter()
                    test_bond_omega_recall3 = AverageMeter()

                    count = 0
                    for j, (imgs, atom_targets, atom_types, atom_charges, atom_hs,
                            bond_targets, bond_types, bond_rhos, bond_omega_types) in enumerate(test_dataloader):
                        count += 1
                        imgs = imgs.to(device)
                        atom_targets, atom_types, atom_charges, atom_hs = atom_targets.to(device), atom_types.to(
                            device), atom_charges.to(device), atom_hs.to(device)
                        bond_targets, bond_types, bond_rhos, bond_omega_types = bond_targets.to(
                            device), bond_types.to(device), bond_rhos.to(
                            device), bond_omega_types.to(device)

                        atom_targets_pred, atom_types_pred, atom_charges_pred, atom_hs_pred, bond_targets_pred, bond_types_pred, bond_rhos_pred, bond_omega_types_pred = model(
                            imgs)
                        atom_targets_pred = torch.clamp(torch.sigmoid(atom_targets_pred), 1e-5, 1 - 1e-5)
                        atom_types_pred = torch.clamp(torch.softmax(atom_types_pred, dim=1), 1e-5, 1 - 1e-5)
                        atom_charges_pred = torch.clamp(torch.softmax(atom_charges_pred, dim=1), 1e-5, 1 - 1e-5)

                        atom_hs_pred = torch.clamp(torch.softmax(atom_hs_pred, dim=1), 1e-5, 1 - 1e-5)

                        bond_targets_pred = torch.clamp(torch.sigmoid(bond_targets_pred), 1e-5, 1 - 1e-5)
                        bond_types_pred = torch.clamp(torch.softmax(bond_types_pred.view(-1, 6, 60, 128, 128), dim=1), 1e-5,
                                                      1 - 1e-5)

                        bond_omega_types_pred = torch.clamp(torch.sigmoid(bond_omega_types_pred.view(-1, 60, 128, 128)),
                                                            1e-5, 1 - 1e-5)
                        bond_rhos_pred = torch.abs(bond_rhos_pred)

                        temp = torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3,
                                                              stride=1, padding=1)
                        atom_targets_pred = (temp == atom_targets_pred) * (atom_targets_pred > 0.25).float()

                        temp = torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3,
                                                              stride=1, padding=1)
                        bond_targets_pred = (temp == bond_targets_pred) * (bond_targets_pred > 0.25).float()

                        atom_targets = (atom_targets == 1).float()
                        test_atom_targets_precision.update(
                            ((atom_targets_pred * atom_targets).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
                            atom_targets_pred.sum().cpu().detach().numpy())
                        test_atom_targets_precision3.update(
                            ((atom_targets_pred * torch.nn.functional.max_pool2d(atom_targets, kernel_size=3, padding=1,
                                                                                 stride=1)).sum() / atom_targets_pred.sum()).cpu().detach().numpy(),
                            atom_targets_pred.sum().cpu().detach().numpy())

                        test_atom_targets_recall.update(
                            ((atom_targets * atom_targets_pred).sum() / atom_targets.sum()).cpu().detach().numpy(),
                            atom_targets.sum().cpu().detach().numpy())
                        test_atom_targets_recall3.update(
                            ((atom_targets * torch.nn.functional.max_pool2d(atom_targets_pred, kernel_size=3, padding=1,
                                                                            stride=1)).sum() / atom_targets.sum()).cpu().detach().numpy(),
                            atom_targets.sum().cpu().detach().numpy())

                        test_atom_types_acc.update((torch.sum(torch.sum(atom_types, dim=1) * ((atom_types.argmax(1) ==
                                                                                               atom_types_pred.argmax(
                                                                                                   1)).float())) / torch.sum(
                            atom_types)).cpu().detach().numpy(), torch.sum(atom_types).cpu().detach().numpy())

                        test_atom_charges_acc.update((torch.sum(torch.sum(atom_charges, dim=1) * ((atom_charges.argmax(1) ==
                                                                                                   atom_charges_pred.argmax(
                                                                                                       1)).float())) / torch.sum(
                            atom_charges)).cpu().detach().numpy(), torch.sum(atom_charges).cpu().detach().numpy())

                        test_atom_hs_acc.update((torch.sum(torch.sum(atom_hs, dim=1) * ((atom_hs.argmax(1) ==
                                                                                         atom_hs_pred.argmax(
                                                                                             1)).float())) / (
                                                             0.01 + torch.sum(atom_hs))).cpu().detach().numpy(),
                                                (0.01 + torch.sum(atom_hs)).cpu().detach().numpy())

                        bond_targets = (bond_targets == 1).float()
                        test_bond_targets_precision.update(
                            ((bond_targets_pred * bond_targets).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
                            bond_targets_pred.sum().cpu().detach().numpy())
                        test_bond_targets_precision3.update(
                            ((bond_targets_pred * torch.nn.functional.max_pool2d(bond_targets, kernel_size=3, padding=1,
                                                                                 stride=1)).sum() / bond_targets_pred.sum()).cpu().detach().numpy(),
                            bond_targets_pred.sum().cpu().detach().numpy())

                        test_bond_targets_recall.update(
                            ((bond_targets * bond_targets_pred).sum() / bond_targets.sum()).cpu().detach().numpy(),
                            bond_targets.sum().cpu().detach().numpy())
                        test_bond_targets_recall3.update(
                            ((bond_targets * torch.nn.functional.max_pool2d(bond_targets_pred, kernel_size=3, padding=1,
                                                                            stride=1)).sum() / bond_targets.sum()).cpu().detach().numpy(),
                            bond_targets.sum().cpu().detach().numpy())

                        test_bond_types_acc.update((torch.sum(torch.sum(bond_types, dim=1) * (
                            (bond_types.argmax(1) == bond_types_pred.argmax(1)).float())) / torch.sum(
                            bond_types)).cpu().detach().numpy(), torch.sum(bond_types).cpu().detach().numpy())

                        test_bond_rhos_mae.update(
                            (torch.sum(torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / \
                             torch.sum(bond_types)).cpu().detach().numpy(), torch.sum(bond_types).cpu().detach().numpy())

                        temp = torch.cat(
                            [bond_omega_types_pred[:, 59:], bond_omega_types_pred, bond_omega_types_pred[:, :1]],
                            dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
                        temp = ((torch.nn.functional.max_pool1d(temp, stride=1, kernel_size=3, padding=0).reshape(-1, 128,
                                                                                                                  128,
                                                                                                                  60).permute(
                            0, 3, 1, 2) == bond_omega_types_pred) * \
                                (bond_omega_types_pred > 0.25)).float() * bond_targets

                        bond_omega_types = (bond_omega_types == 1)
                        test_bond_omega_precision.update(
                            (((bond_omega_types) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                            temp.sum().cpu().detach().numpy())

                        temp2 = torch.cat([temp[:, 59:], temp, temp[:, :1]],
                                          dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
                        temp2 = (
                            torch.nn.functional.max_pool1d(temp2.float(), stride=1, kernel_size=3, padding=0).reshape(-1,
                                                                                                                      128,
                                                                                                                      128,
                                                                                                                      60).permute(
                                0, 3, 1, 2)).float()

                        test_bond_omega_recall3.update(
                            (((bond_omega_types) * temp2).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
                            bond_omega_types.sum().cpu().detach().numpy())

                        test_bond_omega_recall.update(
                            (((bond_omega_types) * temp).sum() / (bond_omega_types).sum()).cpu().detach().numpy(),
                            bond_omega_types.sum().cpu().detach().numpy())

                        temp3 = torch.cat([bond_omega_types[:, 59:], bond_omega_types, bond_omega_types[:, :1]],
                                          dim=1).permute(0, 2, 3, 1).reshape(-1, 128 * 128, 62)
                        temp3 = (
                            torch.nn.functional.max_pool1d(temp3.float(), stride=1, kernel_size=3, padding=0).reshape(-1,
                                                                                                                      128,
                                                                                                                      128,
                                                                                                                      60).permute(
                                0, 3, 1, 2)).float()

                        test_bond_omega_precision3.update((((temp3) * temp).sum() / (temp).sum()).cpu().detach().numpy(),
                                                          temp.sum().cpu().detach().numpy())

                    if local_rank==0:
                        print('test_________________')

                        print('atom_target_precision:', test_atom_targets_precision.avg)
                        print('atom_target_recall:', test_atom_targets_recall.avg)
                        print('atom_target_precision3:', test_atom_targets_precision3.avg)
                        print('atom_target_recall3:', test_atom_targets_recall3.avg)

                        print('atom_types_acc:', test_atom_types_acc.avg)
                        print('atom_charges_acc:', test_atom_charges_acc.avg)
                        print('atom_hs_acc:', test_atom_hs_acc.avg)

                        print('bond_target_precision:', test_bond_targets_precision.avg)
                        print('bond_target_recall:', test_bond_targets_recall.avg)
                        print('bond_target_precision3:', test_bond_targets_precision3.avg)
                        print('bond_target_recall3:', test_bond_targets_recall3.avg)

                        print('bond_types_acc:', test_bond_types_acc.avg)
                        print('bond_rhos_mae:', test_bond_rhos_mae.avg)

                        print('bond_omega_precision:', test_bond_omega_precision.avg)
                        print('bond_omega_recall:', test_bond_omega_recall.avg)
                        print('bond_omega_precision3:', test_bond_omega_precision3.avg)
                        print('bond_omega_recall3:', test_bond_omega_recall3.avg)

        if local_rank==0:
            torch.save(model.state_dict(), 'weights2_{}/unet_model_weights{}.pkl'.format(amount, epoch))


if __name__ == '__main__':
    import time
    start=time.time()
    main()
    print('run time:{}'.format(time.time()-start))





