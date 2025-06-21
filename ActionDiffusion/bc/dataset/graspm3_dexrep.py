import torch
import os
import numpy as np
import os.path as osp
import math
import json
import h5py
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def obs_process_numpy(observation, pro_dim=128):
    """
    observation: numpy array of shape (B, 2582)
    """
    reshape84_149_part =np.arange(84,149).reshape(5,13)
    remove_indice_part = reshape84_149_part[:,-6:].reshape(-1) #fingertips pos&ang vel

    remmove28_56 = np.arange(28,56) # shadow vel
    remove56_84 = np.arange(56,84) # shadow force

    remove149_179 = np.arange(149, 179) #finger force
    remove216_222 = np.arange(216, 222) #obj pos&ang vel



    index_to_remove = np.concatenate([remove56_84, remove149_179])
    if pro_dim==134:
        final_indice_remove = np.concatenate([index_to_remove, remove_indice_part])
    elif pro_dim==128:
        final_indice_remove = np.concatenate([index_to_remove, remove_indice_part, remove216_222])
    elif pro_dim==100:
        final_indice_remove = np.concatenate([index_to_remove, remove_indice_part, remove216_222, remmove28_56])

    all_indices = np.arange(2582)
    mask = ~np.isin(all_indices, final_indice_remove)  # 生成布尔掩码

    filtered_observation = observation[..., mask]
    return filtered_observation


class GraspM3DexRepDataset(Dataset):
    def __init__(self,args, ds_name='test'):
        self.args = args
        self.ds_name =ds_name
        self.pro_dim = args.obs_dim
        if ds_name =='test':
            self.data_dir = args.test_data_dir
        else:
            self.data_dir = args.train_data_dir

        self.seq_num=args.seq_num


        self.unscale_actions_rescale = np.asarray(
            [[25, 25, 21, 1, 1.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(np.float32)

        if self.seq_num==100:
            self.seq_per_num = 100

        elif self.seq_num==350:
            self.seq_per_num = 120

        elif self.seq_num == 1000:
            self.seq_per_num = 100

        elif self.seq_num==10000:
            self.seq_per_num = 80

        elif self.seq_num==100000:
            self.seq_per_num=150

        else:
            self.seq_per_num = self.seq_num


        if ds_name=='test':
            # self.seq_num=10000
            # self.seq_per_num=120
            self.seq_num=500
            self.seq_per_num=100


        # file_path_list =sorted(glob(osp.join(self.data_dir,"*.npy")), key=lambda path: os.path.getsize(path),reverse=True)
        file_path_list = sorted( [f for f in glob(osp.join(self.data_dir, "*.npy")) if os.path.getsize(f) >= 1024],
            key=lambda path: os.path.getsize(path),reverse=True)

        if self.seq_num==2000 and  ds_name=='train':
            train_file_names = np.load('./train_file_names.npy',allow_pickle=True).item()
            file_path_list =[osp.join(self.data_dir, file_name) for file_name in train_file_names['train_obj_num_1h']]

        elif self.seq_num==20000 and ds_name=='train':
            train_file_names = np.load('./train_file_names.npy',allow_pickle=True).item()
            file_path_list =[osp.join(self.data_dir, file_name) for file_name in train_file_names['train_obj_num_1k']]
            a=1

        elif self.seq_num==100 and ds_name=='train':
            file_path_list = ['core-bottle-a02a1255256fbe01c9292f26f73f6538.npy']


        obj_id_list =[osp.basename(path) for path in file_path_list]
        if args.train_obj_code_list is not None and ds_name=='train':
            obj_id_list =[code_name+'.npy' for code_name in args.train_obj_code_list]
        a=1


        self.obj_glob_feats_all = np.load(args.obj_glob_feat_file)
        with open("./object_code_names.json", "r",
                  encoding="utf-8") as f:
            self.obj_code_name_list = json.load(f)

        # self.obj_code_sample_points = np.load('./object_code_sample_points.npy')

        self.is_flat = self.args.batch_seq_flat
        self.is_pad = self.args.start_frame_pad

        self.data = self.data_load(obj_id_list)

        self.keys = self.data.keys()
        print("{} dataset seq_num={}".format(ds_name,len(self.data['obj_code_idx'])))

        # self.num_frame=80


    def seq_filter(self,data_dct):
        seq_obj_pcds = data_dct['obj_pcds']
        x = seq_obj_pcds[..., 0]  # (N, T, 512)
        y = seq_obj_pcds[..., 1]
        z = seq_obj_pcds[..., 2]

        x_max = np.max(np.abs(x), axis=(1, 2))  # (N,)
        y_max = np.max(np.abs(y), axis=(1, 2))
        z_max = np.max(np.abs(z), axis=(1, 2))
        filter_mask = (x_max <= 1.0) & (y_max <= 1.0) & (z_max <= 2.0)  # (N,)


        for key, val in data_dct.items():
            data_dct[key] = val[filter_mask]


        return data_dct

    # def data_load(self, obj_id_list):
    #     data_store = {'obs': [], 'vis_unscale_actions': [], 'obj_code_idx': [],
    #                   'grasp_seqs': [], 'hand_pcds':[], 'obj_pcds':[],}
    #
    #     use_keys = ['obs', 'vis_unscale_actions','hand_pcds','obj_pcds','unscale_actions','grasp_seqs','h2o_vec']
    #     for file_name in obj_id_list:
    #         obj_name = file_name[:-len('.npy')]
    #         obj_code_idx = self.obj_code_name_list.index(obj_name)
    #         data_dct =np.load(osp.join(self.data_dir,file_name), allow_pickle=True).item()
    #
    #         if len(data_dct['grasp_seqs'])!=20:
    #             a=1
    #
    #         data_dct = self.seq_filter(data_dct)
    #
    #         self.num_frame = data_dct['obs'].shape[1]
    #         a=1
    #
    #         for key, val in data_dct.items():
    #             if key !='obj_code_idx' and key in use_keys:
    #                 part_val = val[: self.seq_per_num]
    #                 data_store[key].append(part_val)
    #
    #         data_store['obj_code_idx'].append([obj_code_idx]*len(part_val))
    #
    #         total_len =sum(len(sub_list) for sub_list in data_store['obj_code_idx'])
    #         if total_len>=self.seq_num:
    #             break
    #
    #     for key,val in data_store.items():
    #         val = np.concatenate(val,axis=0)
    #         if key !='obj_code_idx' and self.is_flat:
    #             N, T = val.shape[:2]
    #             val = val.reshape(N*T,*val.shape[2:])#(N*80, D)
    #
    #         if key != 'obj_code_idx' and self.is_pad and self.args.n_obs_steps>1:
    #             start_frame = val[:, 0:1, :]
    #             repeated_start = np.repeat(start_frame, repeats=self.args.n_obs_steps-1, axis=1)
    #             val = np.concatenate((repeated_start, val), axis=1) #(N,80+n_obs_steps, D)
    #             a=1
    #
    #         data_store[key] = val
    #     data_store['obj_code_idx'] = data_store['obj_code_idx'].astype(np.int64) #(N, )
    #     # data_store['obs'][:,:28]*= self.unscale_actions_rescale
    #     # data_store['unscale_actions']*= self.unscale_actions_rescale
    #
    #     data_store['obs'] = obs_process_numpy(data_store['obs'], pro_dim=self.pro_dim) #(...,2582)->(...,2488)
    #
    #
    #     if 'h2o_vec' in data_dct.keys():
    #         data_store['obs_h2o'] = self.get_obs_h2o(data_store['h2o_vec'])
    #     if 'hand_pcds' in data_dct.keys():
    #         data_store['obs_pcds'] = self.get_obs_pcds(data_store['hand_pcds'],data_store['obj_pcds'])
    #
    #     return data_store

    def data_load(self, obj_id_list, max_workers=16):
        use_keys = {'obs', 'vis_unscale_actions', 'hand_pcds', 'obj_pcds', 'unscale_actions', 'grasp_seqs', 'h2o_vec'}
        target_seq_num = self.seq_num
        all_results = []

        def load_one(file_name):
            obj_name = file_name[:-4]
            if obj_name not in self.obj_code_name_list:
                return None

            path = osp.join(self.data_dir, file_name)
            try:
                data_dct = np.load(path, allow_pickle=True).item()
            except Exception as e:
                return None

            if len(data_dct.get('grasp_seqs', [])) > 20:
                # for k, v in data_dct.items():
                #     if isinstance(v, np.ndarray):
                #         data_dct[k] = v[:20]
                # return None
                a=1

            data_dct = self.seq_filter(data_dct)

            item = {'obj_code_idx': self.obj_code_name_list.index(obj_name)}
            for key in use_keys & data_dct.keys():
                if key != 'obj_code_idx':
                    part_val = data_dct[key][:self.seq_per_num]
                    item[key] = part_val
            return item

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, file_name): file_name for file_name in obj_id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Multi-threaded loading"):
                result = future.result()
                if result is not None:
                    all_results.append(result)
                if sum(len(d['obs']) for d in all_results if 'obs' in d) >= target_seq_num:
                    break

        data_store = {key: [] for key in ['obs', 'vis_unscale_actions', 'obj_code_idx',
                                          'grasp_seqs', 'hand_pcds', 'obj_pcds']}

        for item in all_results:
            # n_seq = len(next(iter(item.values())))
            n_seq = len(item['grasp_seqs'])
            self.num_frame = item['grasp_seqs'].shape[1]
            for key in data_store:
                if key == 'obj_code_idx':
                    data_store[key].append(np.full(n_seq, item['obj_code_idx'], dtype=np.int64))
                elif key in item:
                    data_store[key].append(item[key])

        for key, val in data_store.items():
            val = np.concatenate(val, axis=0)
            if key != 'obj_code_idx' and self.is_flat:
                N, T = val.shape[:2]
                val = val.reshape(N * T, *val.shape[2:])  # (N*T, ...)
            if key != 'obj_code_idx' and self.is_pad and self.args.n_obs_steps > 1:
                start_frame = val[:, 0:1, :]
                repeated_start = np.repeat(start_frame, repeats=self.args.n_obs_steps - 1, axis=1)
                val = np.concatenate((repeated_start, val), axis=1)
            data_store[key] = val

        data_store['obs'] = obs_process_numpy(data_store['obs'], pro_dim=self.pro_dim)

        if 'h2o_vec' in all_results[0]:
            data_store['obs_h2o'] = self.get_obs_h2o(
                np.concatenate([item['h2o_vec'] for item in all_results if 'h2o_vec' in item]))
        if 'hand_pcds' in data_store:
            data_store['obs_pcds'] = self.get_obs_pcds(data_store['hand_pcds'], data_store['obj_pcds'])

        return data_store

    @staticmethod
    def get_obs_pcds(hand_pcd, obj_pcd,center=np.asarray([[0.00088756, 0.03031297, 0.21595094]],dtype=np.float32)):

        pcds_max = np.asarray([[1, 1, 2.0]],dtype=np.float32)
        pcds_min =  np.asarray([[-1, -1, -0.05]],dtype=np.float32)
        obs_pcds = np.concatenate([hand_pcd,obj_pcd], axis=-2,dtype=np.float32) #(N,T,1024,3) or(N*T,1024,3)
        # center = obs_pcds.reshape(-1,3).mean(axis=0,keepdims=True)
        obs_pcds-=center

        obs_pcds =(obs_pcds-pcds_min)/(pcds_max-pcds_min)
        obs_pcds=obs_pcds*2-1

        return obs_pcds

    def get_obs_h2o(self, h2o_vec):
        """
        h2o_vec (B,T,21,3)
        """
        h2o_max = np.asarray([[0.08, 0.05, 0.02]],dtype=np.float32)
        h2o_min =  np.asarray([[-0.18, -0.2, -0.23]],dtype=np.float32)
        h2o_vec = (h2o_vec - h2o_min) / (h2o_max - h2o_min)
        h2o_vec = h2o_vec * 2 - 1

        return h2o_vec

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):# idx:0-N*T, T=80
        data_out = {'obs':{}}


        if self.is_flat:
            seq_id = math.floor(idx/self.num_frame)
            frame_id = idx % self.num_frame

            if frame_id==self.num_frame-1:
                idx-=39

            obj_code_idx = self.data['obj_code_idx'][seq_id]
            # data_out['obj_emb'] = self.obj_glob_feats_all[obj_code_idx]

        else:
            obj_code_idx = self.data['obj_code_idx'][idx]

        if self.args.obs_type =='pcds':
            data_out['obs'] = self.data['obs_pcds'][idx] #(T, 1024, 3)

        elif self.args.obs_type =='h2o':
            data_out['obs'] = self.data['obs_h2o'][idx] #(T, 21, 3)

        else:

            if self.args.policy.actor_critic=='ActorCriticPNG':
                obj_emb = self.obj_glob_feats_all[obj_code_idx]
                data_out['obs'] = np.concatenate([self.data['obs'][idx],obj_emb],axis=0)
            else:
                data_out['obs'] = self.data['obs'][idx]  #(2488) if is_flat else (T, 2488)

            if self.args.add_noise and self.ds_name!='test':
                noise = np.random.uniform(-self.args.noise_val, self.args.noise_val, size=self.pro_dim)
                data_out['obs'][..., :self.pro_dim]+=noise

                # data_out['obs'][:207]+=noise
                # noise0_3 = np.random.uniform(-0.05, 0.05, size=3)
                # noise3_28= np.random.uniform(-0.1, 0.1, size=53)
                # noise56_92 = np.random.uniform(-0.02, 0.02, size=36)
                # noise97_100= np.random.uniform(-0.02, 0.02, size=3)
                # noise100_110 = np.random.uniform(-0.05, 0.05, size=10)
                # noise110_113 = np.random.uniform(-0.02, 0.02, size=3)
                # noise113_117 = np.random.uniform(-0.05, 0.05, size=4)


                # data_out['obs'][0:3]+=noise0_3
                # data_out['obs'][3:28]+=noise3_28
                # data_out['obs'][56:92]+=noise56_92
                # data_out['obs'][97:100]+=noise97_100
                # data_out['obs'][100:110]+=noise100_110
                # data_out['obs'][110:113]+=noise110_113
                # data_out['obs'][113:117]+=noise113_117
                a=1


        data_out['actions'] = self.data['vis_unscale_actions'][idx]  # (28) if is_flat else (T, 28)

        return data_out


    # @staticmethod
    # def collate_fn(batch, horizon=4):
    #     out = dict()
    #     batch_keys = batch[0].keys()
    #     skip_keys = ['obj_pcd', 'obj_scale']   # These will be manually collated
    #
    #     T = batch[0]['actions'].shape[0]
    #     t_start = random.randint(0, T - horizon) #最后一帧不取
    #     # For each not in skip_keys, use default torch collator
    #     for key in batch_keys:
    #         data = torch.utils.data._utils.collate.default_collate([d[key] for d in batch])
    #         if key in skip_keys:
    #             out[key] = data
    #         else:
    #             out[key] = data[:,t_start:t_start+horizon]
    #
    #     return out


    @staticmethod
    def collate_fn(batch, horizon=4):
        out = dict()
        batch_keys = batch[0].keys()
        skip_keys = ['obj_pcd', 'obj_scale']

        collated = {key: [] for key in batch_keys}

        for sample in batch:
            T = sample['actions'].shape[0]
            t_start = random.randint(0, T - horizon)
            for key in batch_keys:
                if key in skip_keys:
                    collated[key].append(sample[key])
                else:
                    collated[key].append(sample[key][t_start:t_start + horizon])

        for key in batch_keys:
            out[key] = torch.utils.data._utils.collate.default_collate(collated[key])

        return out



if __name__ == '__main__':
    from omegaconf import OmegaConf

    args = OmegaConf.load("{}/lhm_bc.yaml".format('../ActionDiffusion/bc/config'))

    ds_train = GraspM3DexRepDataset(args, ds_name='train')

    i=0
    for data in iter(ds_train):
            a=1
