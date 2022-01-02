from utils import MolecularImageDataset,collate_fn
from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from unet import UNet
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

parser = argparse.ArgumentParser(description='multi-gpu-training')
parser.add_argument('--data',metavar='dir',default='indigo_train_data',help='path to dataset')
parser.add_argument('--epoch',default=5,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-b','--batch-size',default=160,type=int,metavar='N',help='mini batch size')
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

    model = UNet(in_channels=1, heads=[1, 20, 5, 1, 90, 90, 30, 30])
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    args.batch_size = int(args.batch_size/args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)


    train_df = pd.read_csv('../data/indigo_train_data_test/processed_inchi.csv')[0:100000].copy().reset_index(drop=True)
    test_df = pd.read_csv('../data/indigo_train_data_test/processed_inchi.csv')[100000:101000].copy().reset_index(drop=True)

    train_dataset = MolecularImageDataset(train_df)
    test_dataset = MolecularImageDataset(test_df)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=collate_fn,num_workers=3,prefetch_factor=10,sampler=train_sampler)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,collate_fn=collate_fn,num_workers=3,prefetch_factor=10,sampler=test_sampler)

    #model = SwinNet(in_channels=1 , heads=[1,20,5,1,90,90,30,30])
    #model = DeepLabv3_plus(heads=[1,20,5,1,90,90,30,30])

    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        if epoch == 4:
            optimizer = optim.Adam(model.parameters(),lr=2.5e-5)
        for step,(imgs, atom_targets,atom_types,atom_charges,
               bond_targets,bond_types,bond_stereos,bond_rhos,bond_omega_types) in enumerate(train_loader):

            model.train()
            imgs = imgs.cuda(local_rank,non_blocking=True)
            atom_targets, atom_types, atom_charges = atom_targets.cuda(local_rank,non_blocking=True), atom_types.cuda(local_rank,non_blocking=True), atom_charges.cuda(local_rank,non_blocking=True)
            bond_targets, bond_types, bond_stereos,bond_rhos,bond_omega_types = bond_targets.cuda(local_rank,non_blocking=True), bond_types.cuda(local_rank,non_blocking=True), bond_stereos.cuda(local_rank,non_blocking=True),bond_rhos.cuda(local_rank,non_blocking=True),bond_omega_types.cuda(local_rank,non_blocking=True)

            atom_targets_pred, atom_types_pred, atom_charges_pred,bond_targets_pred, bond_types_pred, bond_stereos_pred,bond_rhos_pred, bond_omega_types_pred = model(imgs)
            atom_targets_pred =  torch.clamp(torch.sigmoid(atom_targets_pred),1e-5,1-1e-5)
            atom_types_pred = torch.clamp(torch.softmax(atom_types_pred,dim=1),1e-5,1-1e-5)
            atom_charges_pred = torch.clamp(torch.softmax(atom_charges_pred, dim=1), 1e-5, 1-1e-5)

            bond_targets_pred = torch.clamp(torch.sigmoid(bond_targets_pred), 1e-5, 1-1e-5)
            bond_types_pred = torch.clamp(torch.softmax(bond_types_pred.view(-1,3,30,120,120), dim=1), 1e-5, 1-1e-5)
            bond_stereos_pred = torch.clamp(torch.softmax(bond_stereos_pred.view(-1,3,30,120,120), dim=1), 1e-5, 1-1e-5)

            bond_omega_types_pred = torch.clamp(torch.sigmoid(bond_omega_types_pred), 1e-5, 1 - 1e-5)
            bond_rhos_pred = (bond_rhos_pred)**2

            atom_targets_loss =  torch.sum(-(atom_targets==1).float() *(1-atom_targets_pred)**2 * torch.log(atom_targets_pred)
                 -(1-atom_targets)**4 * (atom_targets_pred)**2 * torch.log(1-atom_targets_pred))/torch.sum(atom_targets==1)
            atom_types_loss = torch.sum(-atom_types*(1-atom_types_pred)**2*torch.log(atom_types_pred))/torch.sum(atom_types)
            atom_charges_loss = torch.sum(-atom_charges*(1-atom_charges_pred)**2*torch.log(atom_charges_pred))/torch.sum(atom_charges)

            bond_targets_loss = torch.sum(-(bond_targets==1).float() *(1-bond_targets_pred)**2 * torch.log(bond_targets_pred)
                 -(1-bond_targets)**4 * (bond_targets_pred)**2 * torch.log(1-bond_targets_pred))/torch.sum(bond_targets==1)
            bond_types_loss = torch.sum(-bond_types*(1-bond_types_pred)**2*torch.log(bond_types_pred))/torch.sum(bond_types)
            bond_stereos_loss = torch.sum(
                -bond_stereos * ((1 - bond_stereos_pred) ** 2) * torch.log(bond_stereos_pred)) / torch.sum(bond_stereos+1)

            bond_rhos_loss = 0.1 * torch.sum(torch.abs(bond_rhos_pred-bond_rhos)*torch.sum(bond_types,dim=1,keepdim=True))/torch.sum(bond_types)
            bond_omega_types_loss = 0.1 * torch.sum(torch.sum(bond_omega_types,dim=1,keepdim=True)*(-bond_omega_types* ((1-bond_omega_types_pred)**2) * torch.log(bond_omega_types_pred)-
                                                   (1-bond_omega_types)* (bond_omega_types_pred**2) * torch.log(1-bond_omega_types_pred)))/torch.sum(bond_omega_types)


            loss =  atom_targets_loss + bond_targets_loss + atom_types_loss + atom_charges_loss  + bond_types_loss + bond_stereos_loss +  bond_rhos_loss + bond_omega_types_loss

            torch.distributed.barrier()

            reduce_loss = reduce_mean(loss,args.nprocs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%100 == 0:
                model.eval()
                if local_rank==0:
                    print(epoch,step)
                    print(reduce_loss.cpu().detach().numpy())
                with torch.no_grad():
                    # print('loss_________________')
                    # print('atom_targets_loss:',atom_targets_loss.cpu().detach().numpy())
                    # print('atom_types_loss:',atom_types_loss.cpu().detach().numpy())
                    # print('atom_charges_loss:',atom_charges_loss.cpu().detach().numpy())
                    # print('bond_targets_loss:',bond_targets_loss.cpu().detach().numpy())
                    # print('bond_types_loss:',bond_types_loss.cpu().detach().numpy())
                    # print('bond_stereos_loss:',bond_stereos_loss.cpu().detach().numpy())
                    # print('bond_omega_types:',bond_omega_types_loss.cpu().detach().numpy())

                    print('train_________________')
                    atom_targets_acc = (torch.sum(((atom_targets==1)*(atom_targets_pred>0.4)).float()) / (torch.sum(atom_targets==1)).float()).cpu().detach().numpy()
                    print('atom_target_acc:',atom_targets_acc)

                    atom_types_acc = (torch.sum(torch.sum(atom_types,dim=1)*((atom_types.argmax(1)==
                        atom_types_pred.argmax(1)).float()))/torch.sum(atom_types)).cpu().detach().numpy()
                    print('atom_types_acc:',atom_types_acc)

                    atom_charges_acc = (torch.sum(torch.sum(atom_charges,dim=1)*((atom_charges.argmax(1)==
                        atom_charges_pred.argmax(1)).float()))/torch.sum(atom_charges)).cpu().detach().numpy()
                    print('atom_charges_acc:',atom_charges_acc)

                    bond_targets_acc = (torch.sum(((bond_targets==1) * (bond_targets_pred > 0.4)).float()) / (
                         torch.sum(bond_targets == 1)).float()).cpu().detach().numpy()
                    print('bond_target_acc:', bond_targets_acc)

                    bond_types_acc = (torch.sum(torch.sum(bond_types, dim=1) * ((bond_types.argmax(1) ==
                                 bond_types_pred.argmax(1)).float())) / torch.sum(bond_types)).cpu().detach().numpy()
                    print('bond_types_acc:', bond_types_acc)

                    bond_stereo_acc = (torch.sum(torch.sum(bond_stereos, dim=1) * ((bond_stereos.argmax(1) == bond_stereos_pred.argmax(1)).float()))/torch.sum(
                                                 bond_stereos)).cpu().detach().numpy()
                    print('bond_stereos_acc:',  bond_stereo_acc)

                    bond_rhos_mae = (torch.sum(torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / \
                                        torch.sum(bond_types)).cpu().detach().numpy()
                    print('bond_rhos_mae:', bond_rhos_mae)

                    temp = torch.sum(bond_rhos) / torch.sum((bond_rhos >= 1) * 1)
                    bond_rhos_max = ((torch.max(torch.abs(bond_rhos-bond_rhos_pred)* torch.sum(bond_types,dim=1))/ temp) ).cpu().detach().numpy()
                    print('bond_rhos_max_error:', bond_rhos_max)

                    bond_omega_acc = (torch.sum(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * (torch.abs(bond_omega_types.argmax(dim=1)-bond_omega_types_pred.argmax(dim=1))<4) )/torch.sum(torch.sum(bond_types==1))).cpu().detach().numpy()
                    print('bond_omega_acc:', bond_omega_acc)

                    bond_omega_mae = (torch.sum(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * torch.min( torch.abs(bond_omega_types.argmax(dim=1)-bond_omega_types_pred.argmax(dim=1)),
                                        30-torch.abs(bond_omega_types.argmax(dim=1)-bond_omega_types_pred.argmax(dim=1)) ))/torch.sum(bond_types==1)).cpu().detach().numpy()
                    print('bond_omega_mae:', bond_omega_mae)

                    bond_omega_max = torch.max(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * torch.min( torch.abs(bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1)),
                                                                                         (30 - torch.abs(bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1))) )).cpu().detach().numpy()
                    print('bond_omega_max:', bond_omega_max)

                    test_atom_target_acc = 0
                    test_atom_types_acc= 0
                    test_atom_charges_acc= 0
                    test_bond_target_acc= 0
                    test_bond_types_acc= 0
                    test_bond_stereos_acc= 0
                    test_bond_rhos_mae= 0
                    test_bond_rhos_max_error= 0
                    test_bond_omega_acc= 0
                    test_bond_omega_mae= 0
                    test_bond_omega_max= 0
                    count = 0
                    for j, (imgs, atom_targets, atom_types, atom_charges,
                            bond_targets, bond_types, bond_stereos, bond_rhos, bond_omega_types) in enumerate(test_loader):
                        count +=1
                        imgs = imgs.cuda(local_rank,non_blocking=True)
                        atom_targets, atom_types, atom_charges = atom_targets.cuda(local_rank,non_blocking=True), \
                                                                 atom_types.cuda(local_rank,non_blocking=True), atom_charges.cuda(local_rank,non_blocking=True)
                        bond_targets, bond_types, bond_stereos, bond_rhos, bond_omega_types = bond_targets.cuda(local_rank,non_blocking=True), bond_types.cuda(local_rank,non_blocking=True), \
                            bond_stereos.cuda(local_rank,non_blocking=True), bond_rhos.cuda(local_rank,non_blocking=True), bond_omega_types.cuda(local_rank,non_blocking=True)

                        atom_targets_pred, atom_types_pred, atom_charges_pred, bond_targets_pred, bond_types_pred, bond_stereos_pred, bond_rhos_pred, bond_omega_types_pred = model(
                            imgs)
                        atom_targets_pred = torch.clamp(torch.sigmoid(atom_targets_pred), 1e-5, 1 - 1e-5)
                        atom_types_pred = torch.clamp(torch.softmax(atom_types_pred, dim=1), 1e-5, 1 - 1e-5)
                        atom_charges_pred = torch.clamp(torch.softmax(atom_charges_pred, dim=1), 1e-5, 1 - 1e-5)

                        bond_targets_pred = torch.clamp(torch.sigmoid(bond_targets_pred), 1e-5, 1 - 1e-5)
                        bond_types_pred = torch.clamp(torch.softmax(bond_types_pred.view(-1, 3, 30, 120, 120), dim=1), 1e-5,
                                                      1 - 1e-5)
                        bond_stereos_pred = torch.clamp(torch.softmax(bond_stereos_pred.view(-1, 3, 30, 120, 120), dim=1),
                                                        1e-5, 1 - 1e-5)

                        bond_omega_types_pred = torch.clamp(torch.sigmoid(bond_omega_types_pred.view(-1, 30, 120, 120)),
                                                            1e-5, 1 - 1e-5)
                        bond_rhos_pred = (bond_rhos_pred.view(-1, 30, 120, 120)) ** 2

                        atom_targets_acc = (torch.sum(((atom_targets == 1) * (atom_targets_pred > 0.4)).float()) / (
                            torch.sum(atom_targets == 1)).float())

                        test_atom_target_acc += atom_targets_acc
                        atom_types_acc = (torch.sum(torch.sum(atom_types, dim=1) * ((atom_types.argmax(1) ==
                                                                                     atom_types_pred.argmax(
                                                                                         1)).float())) / torch.sum(
                            atom_types))

                        test_atom_types_acc += atom_types_acc

                        atom_charges_acc = (torch.sum(torch.sum(atom_charges, dim=1) * ((atom_charges.argmax(1) ==
                                                                                         atom_charges_pred.argmax(
                                                                                             1)).float())) / torch.sum(
                            atom_charges))
                        test_atom_charges_acc += atom_charges_acc

                        bond_targets_acc = (torch.sum(((bond_targets == 1) * (bond_targets_pred > 0.4)).float()) / (
                            torch.sum(bond_targets == 1)).float())

                        test_bond_target_acc += bond_targets_acc

                        bond_types_acc = (torch.sum(torch.sum(bond_types, dim=1) * ((bond_types.argmax(1) ==
                                                                                     bond_types_pred.argmax(
                                                                                         1)).float())) / torch.sum(
                            bond_types))

                        test_bond_types_acc += bond_types_acc

                        bond_stereo_acc = (torch.sum(torch.sum(bond_stereos, dim=1) * (
                            (bond_stereos.argmax(1) == bond_stereos_pred.argmax(1)).float())) / torch.sum(
                            bond_stereos))

                        test_bond_stereos_acc += bond_stereo_acc

                        bond_rhos_mae = (torch.sum(
                            torch.abs(bond_rhos_pred - bond_rhos) * torch.sum(bond_types, dim=1)) / \
                                         torch.sum(bond_types))

                        test_bond_rhos_mae += bond_rhos_mae

                        temp = torch.sum(bond_rhos) / torch.sum((bond_rhos >= 1) * 1)
                        bond_rhos_max = ((torch.max(torch.abs(bond_rhos - bond_rhos_pred) * torch.sum(bond_types,
                                                                                                      dim=1)) / temp))
                        test_bond_rhos_max_error  += bond_rhos_max

                        bond_omega_acc = (torch.sum(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * (torch.abs(
                            bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1)) < 4)) / torch.sum(
                            torch.sum(bond_types == 1)))
                        test_bond_omega_acc += bond_omega_acc

                        bond_omega_mae = (torch.sum(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * torch.min(
                            torch.abs(bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1)),
                            30 - torch.abs(
                                bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1)))) / torch.sum(
                            bond_types == 1))
                        test_bond_omega_mae += bond_omega_mae

                        bond_omega_max = torch.max(torch.sum(torch.sum(bond_types==1,dim=1),dim=1) * torch.min(
                            torch.abs(bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(dim=1)),
                            (30 - torch.abs(bond_omega_types.argmax(dim=1) - bond_omega_types_pred.argmax(
                                dim=1)))))
                        test_bond_omega_max += bond_omega_max

                    t1 = test_atom_target_acc / count
                    t2 = test_atom_types_acc/count
                    t3 = test_atom_charges_acc/count
                    t4 = test_bond_target_acc/count
                    t5 = test_bond_types_acc/count
                    t6 = test_bond_stereos_acc/count
                    t7 = test_bond_omega_mae/count
                    t8 = test_bond_rhos_max_error/count
                    t9 = test_bond_omega_acc/count
                    t10 = test_bond_omega_mae/count
                    t11 = test_bond_omega_max/count
                    torch.distributed.barrier()
                    t1 = reduce_mean(t1,args.nprocs)
                    t2 = reduce_mean(t2, args.nprocs)
                    t3 = reduce_mean(t3, args.nprocs)
                    t4 = reduce_mean(t4, args.nprocs)
                    t5 = reduce_mean(t5, args.nprocs)
                    t6 = reduce_mean(t6, args.nprocs)
                    t7 = reduce_mean(t7, args.nprocs)
                    t8 = reduce_mean(t8, args.nprocs)
                    t9 = reduce_mean(t9, args.nprocs)
                    t10 = reduce_mean(t10, args.nprocs)
                    t11 = reduce_mean(t11, args.nprocs)

                    if local_rank==0:
                        print('test________________________')
                        print('test_atom_target_acc:', t1.detach().cpu().numpy())
                        print('test_atom_types_acc:', t2.detach().cpu().numpy())
                        print('test_atom_charges_acc:', t3.detach().cpu().numpy())
                        print('test_bond_target_acc:', t4.detach().cpu().numpy())
                        print('test_bond_types_acc:', t5.detach().cpu().numpy())
                        print('test_bond_stereos_acc:', t6.detach().cpu().numpy())
                        print('test_bond_rhos_mae:', t7.detach().cpu().numpy())
                        print('test_bond_rhos_max_error:', t8.detach().cpu().numpy())
                        print('test_bond_omega_acc:', t9.detach().cpu().numpy())
                        print('test_bond_omega_mae:', t10.detach().cpu().numpy())
                        print('test_bond_omega_max:', t11.detach().cpu().numpy())

        if local_rank==0:
            torch.save(model.module.state_dict(),'unet_model_weights.pkl')

if __name__ == '__main__':
    import time
    start=time.time()
    main()
    print('run time:{}'.format(time.time()-start))




