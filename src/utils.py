import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from cv2 import resize
import skimage.morphology as sm
from skimage.transform import rotate

# {'C': 6136637, 'N': 915240, 'O': 937100, 'P': 19613, 'F': 124551,
# 'Cl': 74160, 'S': 104677, 'Br': 57488, 'B': 2969, 'Se': 1633, 'I': 10490, 'H': 1394, 'Si': 1795}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
atom_vocab = {'<unkonw>':0,'C': 1, 'N': 2, 'O': 3, 'P': 4, 'F': 5,'Cl': 6, 'S': 7, 'Br': 8, 'B': 9,
         'Se': 10, 'I': 11,'H': 12, 'Si': 13}
charge_vocab = {0 : 0, 1 : 1, -1: 2}
bond_vocab = {1 : 0, 2 : 1, 3 : 2, 4 : 3}
stereo_vocab = {0 : 0, 1 : 1, 6 : 2}

# def gaussian2D(shape, sigma=1.5):
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m+1,-n:n+1]
#
#     h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     return h

class MolecularImageDataset(Dataset):
    def __init__(self, df, transform=None, amount=0.1):
        super(MolecularImageDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.amount = amount

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx,'path']
        path = '../data/'+ path
        atoms_string = self.df.loc[idx,'atoms_string']
        bonds_string = self.df.loc[idx,'bonds_string']

        temp_img = cv2.imread(path,flags=0).astype('float32')

        scale_x = 1
        scale_y = 1

        if  np.random.rand()<0.2:
            if np.random.rand()<0.5:
                scale_x = np.random.uniform(0.8,1)
                temp_img = resize(temp_img,(512,int(scale_x*512)))

            else:
                scale_y = np.random.uniform(0.8,1)
                temp_img = resize(temp_img,(int(scale_y*512),512))


        ddx = (512-temp_img.shape[0])//2
        ddy = (512-temp_img.shape[1])//2

        temp_img2 = np.ones((512,512)).astype('float32')*255
        temp_img2[ddx:(ddx+temp_img.shape[0]),ddy:(ddy+temp_img.shape[1])] = temp_img

        temp_img = ((temp_img2/255)<0.6)*1

        #if np.random.rand()<0.1: 
        #    aaa = np.random.rand()
        #    if aaa<0.5:
        #        temp_img = sm.thin(temp_img,max_iter=np.random.randint(0,2))
        #    else:
        #        temp_img = sm.thin(temp_img,max_iter=1)
        #        temp_img = sm.dilation(temp_img,selem=sm.disk(1))

        salt_amount = np.random.uniform(0,self.amount/100)
        salt = np.random.uniform(0,1,temp_img.shape)<salt_amount
        temp_img = np.logical_or(temp_img,salt)
        pepper_amount = np.random.uniform(0,self.amount)
        pepper = np.random.uniform(0,1,temp_img.shape)<pepper_amount
        temp_img = np.logical_or(1-temp_img,pepper)

        img = np.zeros([1,512,512]).astype('float32')
        img[0] = 1-temp_img

        atom_target = np.zeros([1, 128,128]).astype('float32')
        atom_type = np.zeros([14,128,128]).astype('float32')
        atom_charge = np.zeros([3,128,128]).astype('float32')
        atom_hs = np.zeros([2,128,128]).astype('float32')

        bond_target = np.zeros([1, 128,128]).astype('float32')
        bond_type = np.zeros([6, 60, 128,128]).astype('float32')
        delta_omega = np.pi / 30
        bond_rho = np.zeros([60, 128, 128])
        bond_omega_type = np.zeros([60, 128, 128])

        for atom_string in atoms_string.split(';')[:-1]:
            atom,position = atom_string.split(':')
            if len(atom)==1:
                atom = atom.upper()
            idx = atom_vocab.get(atom,0)
            
            if len(position.split(','))==4:
                x, y, charge,hs = position.split(',')
                x, y, charge = int(int(x)*scale_x+ddx)//4, int(int(y)*scale_y+ddy)//4, int(charge)
                hs = int(hs)
            else:
                x, y, charge = position.split(',')
                x, y, charge = int(int(x)*scale_x+ddx) // 4, int(int(y)*scale_y+ddy) // 4, int(charge)
                hs = -1

            x_begin = x-1
            if x==0:
                x_begin = 0
            y_begin = y-1
            if y==0:
                y_begin=0

            atom_target[0, x_begin:(x + 2), y_begin:(y + 2)] = 0.8
            atom_target[0, x, y] = 1

            atom_type[idx,x_begin:(x+2),y_begin:(y+2)] = 0.5
            atom_type[idx, x,y] = 1
            atom_charge[charge_vocab.get(charge,0), x_begin:(x+2),y_begin:(y+2)] = 0.5
            atom_charge[charge_vocab.get(charge,0), x,y] = 1

            if (hs==0) or (hs==1):
                atom_hs[hs,x_begin:(x+2),y_begin:(y+2)] = 0.5
                atom_hs[hs,x,y] = 1

        for bond_string in bonds_string.split(';')[:-1]:
            bond, position = bond_string.split(':')
            type_idx = bond_vocab.get(int(bond),0)
            x, y, delta_x, delta_y,bond_stereo,bond_direction = position.split(',')
            x, y, delta_x, delta_y = int(int(x)*scale_x+ddx)//4, int(int(y)*scale_y+ddy)//4, (int(delta_x)*scale_x)/4, (int(delta_y)*scale_y)/4


            bond_stereo = int(bond_stereo)
            bond_direction = int(bond_direction)

            if bond_stereo == 5 or bond_stereo==1:
                type_idx = 4
            elif bond_stereo==6:
                type_idx = 5

            bond_target[0, x, y] = 1

            if delta_x<0:
                delta_x = -delta_x
                delta_y = -delta_y
            elif delta_x==0:
                if delta_y>0:
                    bond_direction = 1
                delta_y = -abs(delta_y)

            rho = np.sqrt(delta_x*delta_x + delta_y*delta_y)
            omega = np.math.atan(delta_y/(delta_x+1e-6))
            omega_idx = int(np.floor((omega+np.pi/2)/delta_omega))

            x_begin = x-1
            if x==0:
                x_begin=0
            y_begin = y-1
            if y==0:
                y_begin = 0


            bond_target[0, x_begin:(x+2),y_begin:(y+2)] = 0.8
            bond_target[0, x, y] = 1

            if type_idx==4 or type_idx==5:
                if bond_direction==1:
                    omega_idx = omega_idx+30

                omega_idx_begin = omega_idx-1
                if omega_idx==0:
                    omega_idx_begin = 0

                bond_rho[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = rho
                
                bond_omega_type[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.8
                bond_omega_type[omega_idx,x,y] = 1

                bond_type[type_idx,omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.5
                bond_type[type_idx,omega_idx, x,y] = 1


                if omega_idx == 0:
                    bond_rho[-1, x_begin:(x + 2), y_begin:(y + 2)] = rho
                    bond_omega_type[-1, x_begin:(x+2),y_begin:(y+2)] = 0.8
                    bond_type[type_idx, -1, x_begin:(x+2),y_begin:(y+2)] = 0.5
                if omega_idx == 59:
                    bond_rho[0, x_begin:(x + 2), y_begin:(y + 2)] = rho
                    bond_omega_type[0, x_begin:(x+2),y_begin:(y+2)] = 0.8
                    bond_type[type_idx, 0, x_begin:(x+2),y_begin:(y+2)] = 0.5

            else:
                
                omega_idx_begin = omega_idx-1
                if omega_idx==0:
                    omega_idx_begin = 0

                bond_rho[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = rho 

                bond_omega_type[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.8
                bond_omega_type[omega_idx,x,y] = 1

                bond_type[type_idx,omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.5
                bond_type[type_idx,omega_idx, x,y] = 1 

                if omega_idx == 0:
                    bond_rho[-1, x_begin:(x + 2), y_begin:(y + 2)] = rho
                    bond_omega_type[-1, x_begin:(x+2),y_begin:(y+2)] = 0.8
                    bond_type[type_idx, -1, x_begin:(x+2),y_begin:(y+2)] = 0.5

                omega_idx = omega_idx+30

                omega_idx_begin = omega_idx-1

                bond_rho[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = rho

                bond_omega_type[omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.8
                bond_omega_type[omega_idx,x,y] = 1

                bond_type[type_idx,omega_idx_begin:(omega_idx+2),x_begin:(x+2),y_begin:(y+2)] = 0.5
                bond_type[type_idx,omega_idx, x,y] = 1

                if omega_idx == 59:
                    bond_rho[0, x_begin:(x + 2), y_begin:(y + 2)] = rho
                    bond_omega_type[0, x_begin:(x+2),y_begin:(y+2)] = 0.8
                    bond_type[type_idx, 0, x_begin:(x+2),y_begin:(y+2)] = 0.5

        #import matplotlib.pyplot as plt
        #plt.subplot(131)
        #plt.imshow(img[0])
        #plt.subplot(132)
        #plt.imshow(atom_target[0])
        #plt.subplot(133)
        #plt.imshow(bond_target[0])
        # for x, y in zip(*((bond_target[0] == 1).nonzero())):
        #     rho = bond_rho[:, x, y]
        #     omega = bond_omega_type[:, x, y].argmax() * (np.pi / 30) + np.pi / 60 - np.pi / 2
        #     plt.plot([y - rho * np.sin(omega), y + rho * np.sin(omega)],
        #              [x - rho * np.cos(omega), x + rho * np.cos(omega)])
        #
        #plt.show()
        # pass
        # filter_center = gaussian2D([7,7],7/6)
        # atom_target[0] = cv2.filter2D(atom_target[0],-1,filter_center)
        # bond_target[0] = cv2.filter2D(bond_target[0],-1,filter_center)
        
        # atom_target[atom_target>1] = 1
        # bond_target[bond_target>1] = 1
        return img,atom_target,atom_type,atom_charge,atom_hs,bond_target,bond_type,bond_rho,bond_omega_type


def collate_fn(batch):
    """
    Collate to apply the padding to the captions with dataloader
    """
    imgs,atom_targets,atom_types,atom_charges,atom_hs,bond_targets,bond_types, bond_rhos,bond_omega_types = zip(*batch)
    imgs = [np.expand_dims(img,0) for img in imgs]
    imgs = np.concatenate(imgs,axis=0)

    atom_targets = [np.expand_dims(atom_target,0) for atom_target in atom_targets]
    atom_targets = np.concatenate(atom_targets, axis=0)

    atom_types = [np.expand_dims(atom_type, 0) for atom_type in atom_types]
    atom_types = np.concatenate(atom_types, axis=0)

    atom_charges = [np.expand_dims(atom_charge, 0) for atom_charge in atom_charges]
    atom_charges = np.concatenate(atom_charges, axis=0)


    atom_hs = [np.expand_dims(atom_h, 0) for atom_h in atom_hs]
    atom_hs = np.concatenate(atom_hs, axis=0)


    bond_targets = [np.expand_dims(bond_target, 0) for bond_target in bond_targets]
    bond_targets = np.concatenate(bond_targets, axis=0)

    bond_types = [np.expand_dims(bond_type, 0) for bond_type in bond_types]
    bond_types = np.concatenate(bond_types, axis=0)

    bond_rhos = [np.expand_dims(bond_rho, 0) for bond_rho in bond_rhos]
    bond_rhos = np.concatenate(bond_rhos, axis=0)

    bond_omega_types = [np.expand_dims(bond_omega_type, 0) for bond_omega_type in bond_omega_types]
    bond_omega_types = np.concatenate(bond_omega_types, axis=0)

    imgs = torch.from_numpy(imgs)

    atom_targets = torch.from_numpy(atom_targets)
    atom_types = torch.from_numpy(atom_types)
    atom_charges = torch.from_numpy(atom_charges)
    atom_hs = torch.from_numpy(atom_hs)

    bond_targets = torch.from_numpy(bond_targets)
    bond_types = torch.from_numpy(bond_types)
    bond_rhos = torch.from_numpy(bond_rhos)
    bond_omega_types = torch.from_numpy(bond_omega_types)

    return imgs, atom_targets,atom_types,atom_charges,atom_hs,bond_targets,bond_types,bond_rhos,bond_omega_types





