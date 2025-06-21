import torch
import torch.nn as nn
import torch.nn.functional as F
from dexrep.pointnet_model.pointnet_utils import PointNetEncoder

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256,norm='bn',activation=nn.ReLU):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)

        if norm=='bn':
            self.bn1 = nn.BatchNorm1d(n_neurons)
            self.bn2 = nn.BatchNorm1d(Fout)

        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        # self.ll = nn.LeakyReLU(negative_slope=0.2)
        self.ll = activation(True)


    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


def pcd_downsample(xyz, sample_points):
    # fps_idx = farthest_point_sample(xyz, sample_points)  # [B, npoint, C]
    fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), sample_points).long()  # [B, s]

    new_xyz = index_points(xyz, fps_idx)
    return new_xyz


def feature_upsample(xyz1, xyz2, points2):
    """
    Input:
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        point2: input points data, [B, D, S]
    Return:
        new_points: upsampled points data, [B, D', N]
    """
    xyz1 = xyz1.permute(0, 2, 1)
    xyz2 = xyz2.permute(0, 2, 1)

    points2 = points2.permute(0, 2, 1)
    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape

    dists = square_distance(xyz1, xyz2)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
    interpolated_points = interpolated_points.permute(0, 2, 1)

    return interpolated_points


class AeEncoder(nn.Module):
    def __init__(self, in_chan=4, latent_dim=128, norm='bn', acti='relu'):
        super(AeEncoder, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, inchan=in_chan, outchan=latent_dim,
                                    norms=norm)
        # self.feat = PointNet2Encoder(in_chan, latent_dim, norm, acti)
        self.fc1 = nn.Sequential(nn.Linear(latent_dim + 4096, 256))
        self.fc2 = nn.Sequential(nn.Linear(256, 256))
        self.fc3 = nn.Sequential(nn.Linear(256, 2048))

        self.dec_bn1 = nn.BatchNorm1d(4096)

    def forward(self, xyz, feature, obj_bps):
        """
        :param xyz: (B.D,N)
        :param feature: (B,1,N)
        :return:
        """
        xyz_f = torch.cat([xyz, feature], dim=1)

        latent_code = self.feat(xyz_f)  # (B, 128)
        obj_bps = self.dec_bn1(obj_bps)
        de_input = torch.cat([latent_code, obj_bps], dim=1)  # (B,128+4096)

        x = F.relu(self.fc1(de_input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x_rec = x.contiguous().view(B, 2048, 3)

        return x


def transformer_create(t_args):
    transformers = []
    output_dims = t_args.input_dims[1:] + [1]

    config = BertConfig.from_pretrained(t_args.config_name if t_args.config_name else t_args.model_name_or_path)
    for i in range(len(output_dims)):
        config.max_position_embeddings = 512
        config.output_attentions = False
        config.hidden_dropout_prob = t_args.drop_out
        config.input_dim = t_args.input_dims[i]
        config.output_dim = output_dims[i]
        config.hidden_size = t_args.hidden_dims[i]
        config.intermediate_size = int(config.hidden_size * 4)
        config.num_attention_heads = t_args.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        model = Transformer(config)
        transformers.append(model)

    return transformers


class AeTransformer(nn.Module):
    def __init__(self, in_chan=7, latent_dim=512, norm='bn', acti='relu', t_args=None):
        super(AeTransformer, self).__init__()
        self.encoder = PointNet2Encoder(in_chan, latent_dim, norm, acti)
        # self.encoder = PointNetEncoder(global_feat=True, feature_transform=True, inchan=in_chan, outchan=latent_dim)

        transformers = transformer_create(t_args)
        self.decoder = nn.Sequential(*transformers)

    def forward(self, xyz, feature):
        """
        :param xyz: (B.D,N)
        :param feature: (B,1,N)
        :return:
        """

        xyz_f = torch.cat([xyz, feature], dim=1)
        latent_code, l0_xyz, l1_xyz = self.encoder(xyz_f)  # (B, 1024)

        t_pcd = l1_xyz.transpose(1, 2)  # (B, N, D)
        latent_code = latent_code.unsqueeze(1).expand(-1, 512, -1)
        queries = torch.cat([t_pcd, latent_code], dim=2)  # (B, 1024, 512+3)

        out = self.decoder(queries)  # (B,1024,1)
        out = feature_upsample(l0_xyz, l1_xyz, out.transpose(1, 2))

        out = F.tanh(out)
        out = (out + 1) / 2

        return out.squeeze(1)


class AeConv1d(nn.Module):
    def __init__(self, in_chan=3, out_chan=2048 * 3, latent_dim=1024, n_neuron=1024, norm='bn', acti='relu',
                 is_dec=True):
        super(AeConv1d, self).__init__()
        self.is_dec = is_dec
        self.feat = PointNet2Encoder(in_chan, latent_dim, norm, acti, return_local=True)
        self.conv1_1 = nn.Sequential(nn.Conv1d(latent_dim, n_neuron, 1))
        self.conv1_2 = nn.Sequential(nn.Conv1d(n_neuron, n_neuron, 1))
        self.conv1_3 = nn.Sequential(nn.Conv1d(n_neuron, out_chan, 1))


    def forward(self, xyz, fps_idx=None):
        """
        :param xyz: (B.D,3)
        :return:
        """
        bs = xyz.size(0)

        latent_code, _, l1_xyz, point_feat = self.feat(xyz, fps_idx)  # (B, 1024) (B,64+1024,1024)
        # dec_input= torch.cat([latent_code,obj_cds], dim=1) #(B,3+512,2048)

        if self.is_dec:
            glob_f = latent_code.unsqueeze(-1)
            x = F.relu(self.conv1_1(glob_f))
            x = F.relu(self.conv1_2(x))
            x = self.conv1_3(x)

            x_rec = x.contiguous().view(bs, 2048, 3)

            return {'pcd_rec': x_rec}
        else:
            return latent_code, l1_xyz,  #point_feat


class AePcd(nn.Module):
    def __init__(self, global_feat=False, latent_dim=1024,n_neuron=1024, norm='bn', return_pf=False, is_dec=True):
        super(AePcd, self).__init__()
        self.is_dec = is_dec
        self.return_pf = return_pf

        self.feat = PointNetEncoder(global_feat=global_feat, feature_transform=True, inchan=3, outchan=latent_dim,
                                    norms=norm)
        self.fc1 = nn.Sequential(nn.Linear(latent_dim, n_neuron), nn.BatchNorm1d(n_neuron))
        self.fc2 = nn.Sequential(nn.Linear(n_neuron, n_neuron), nn.BatchNorm1d(n_neuron))
        self.fc3 = nn.Linear(n_neuron, 2048 * 3)

    def forward(self, xyz):
        """
        :param xyz: (B.D,3)
        :return:
        """
        bs = xyz.size(0)

        # dec_input= torch.cat([latent_code,obj_cds], dim=1) #(B,3+512,2048)
        glob_feat, pcd_feat = self.feat(xyz)  # (B, 1024), (B,Fdim,2048)
        if self.is_dec:

            x = F.relu(self.fc1(glob_feat))
            x = self.fc2(x)
            x = self.fc3(x)

            x_rec = x.reshape(bs, 2048, 3)

            return {'pcd_rec': x_rec, 'pcd_feat': pcd_feat}
        return {'pcd_feat': pcd_feat}


class AeContact(nn.Module):
    def __init__(self, in_chan=4, out_chan=1, latent_dim=1024, norm='bn', acti='relu', is_dec=True):
        super(AeContact, self).__init__()
        self.is_dec = is_dec
        self.feat_encoder = PointNet2Encoder(in_chan, latent_dim, norm, acti)
        self.pcd_encoder = PointNetEncoder(global_feat=False, feature_transform=True, inchan=3, outchan=latent_dim,
                                           norms=norm)
        self.conv1 = nn.Sequential(nn.Conv1d(latent_dim + 1088, 512, 1))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1))
        self.conv4 = nn.Sequential(nn.Conv1d(128, out_chan, 1))

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv11 = nn.Conv1d(1088, 512, 1)
        self.conv12 = nn.Conv1d(512, 256, 1)
        self.conv13 = nn.Conv1d(256, 3, 1)
        self.bn11 = nn.BatchNorm1d(512)
        self.bn12 = nn.BatchNorm1d(256)

    def forward(self, xyz, feats):
        """
        :param xyz: (B.3,N)
        :param feats: (B.1,N)

        :return:
        """
        bs = xyz.size(0)
        xyz_f = torch.cat([xyz, feats], dim=1)
        latent_code, _, _ = self.feat_encoder(xyz_f)  # (B, 1024,1)

        if self.is_dec:
            # glob_f = latent_code.unsqueeze(-1)
            pcd_feats = self.pcd_encoder(xyz)  # (B, 1088,2048)
            latent_code = latent_code.view(-1, 1024, 1).repeat(1, 1, 2048)
            x = torch.cat([latent_code, pcd_feats], dim=1)  # (B, 1024+1088,2048)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            x = F.sigmoid(x.view(bs, -1))

            xyz = F.relu(self.bn11(self.conv11(pcd_feats)))
            xyz = F.relu(self.bn12(self.conv12(xyz)))
            xyz = self.conv13(xyz)  # (B, 3, 2048)

            return (x, xyz)
        else:
            return latent_code



class ShapeNetPnet2Encoder(nn.Module):
    def __init__(self, in_chan=3, out_chan=2048 * 3, globD=1024, neuron=1024):
        super(ShapeNetPnet2Encoder, self).__init__()
        self.globD=globD

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_chan + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(512, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(128, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.sa5 = PointNetSetAbstraction(None, None, None, 512 + 3, [512, 512, globD], True)

        self.conv1_1 = nn.Sequential(nn.Conv1d(globD, neuron, 1))
        self.conv1_2 = nn.Sequential(nn.Conv1d(neuron, neuron, 1))

        self.conv1_3 = nn.Sequential(nn.Conv1d(neuron, out_chan, 1))
        self.conv1_4 = nn.Sequential(nn.Conv1d(out_chan, out_chan, 1))


    def forward(self,xyz):
        bs = xyz.size(0)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        _, l5_points = self.sa5(l4_xyz, l4_points)  # (B, globD)

        glob_f = l5_points.view(bs, self.globD,1)
        x = F.relu(self.conv1_1(glob_f))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.conv1_4(x)

        x_rec = x.contiguous().view(bs, 2048, 3)


        return {'pcd_rec': x_rec}


class ShapeNetPnetEncoder(nn.Module):
    def __init__(self,batch_size, in_chan=3, cls_outchan=8, globD=1024, n_neurons=512):
        super(ShapeNetPnetEncoder, self).__init__()
        self.globD = globD
        self.pcd_encoder = PointNetEncoder(global_feat=True, feature_transform=True, inchan=in_chan, outchan=globD, bs=batch_size)
        self.bn = nn.BatchNorm1d(globD)

        self.cls_fc1 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256))
        self.cls_fc2 = nn.Linear(256, 128) #64
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.3)
        self.cls_fc3 = nn.Linear(128, cls_outchan)#64


        self.rec_conv1 = nn.Sequential(nn.Conv1d(1088,512,1),  nn.BatchNorm1d(512))
        self.rec_conv2 = nn.Sequential(nn.Conv1d(512,256,1),  nn.BatchNorm1d(256))
        self.rec_conv3 = nn.Sequential(nn.Conv1d(256,128,1),  nn.BatchNorm1d(128))
        self.rec_conv4 = nn.Conv1d(128, 128, 1)
        self.rec_conv_out = nn.Conv1d(128, 3, 1)



    def forward(self,xyz):
        glob_feat, point_feat = self.pcd_encoder(xyz)
        cls_out = self.cls_decode(glob_feat)
        rec_out = self.rec_decode(point_feat)

        return {'pcd_rec': rec_out.transpose(1,2),'pcd_cls':cls_out}


    def rec_decode(self,x0):
        x0 = F.relu(self.rec_conv1(x0))
        x0 = F.relu(self.rec_conv2(x0))
        x0 = F.relu(self.rec_conv3(x0))
        x0 = self.rec_conv4(x0)
        out = self.rec_conv_out(x0)
        return out

    def cls_decode(self, x):
        x = F.relu(self.cls_fc1(x))
        x = self.dropout(self.cls_fc2(x))
        x = self.bn2(x)
        x = F.relu(x)
        out = self.cls_fc3(x)

        # out = self.cls_acti(x)
        return out

    def inference(self,xyz):
        with torch.no_grad():
            glob_feat, _ = self.encode(xyz)
            return glob_feat


class ShapeNetAutoEncoder(nn.Module):
    def __init__(self, in_chan=3, cls_outchan=55, globD=1024, n_neurons=512):
        super(ShapeNetAutoEncoder, self).__init__()
        self.globD = globD
        self.pcd_encoder = PointNetEncoder(global_feat=True, feature_transform=True, inchan=in_chan, outchan=globD)
        self.bn = nn.BatchNorm1d(globD)

        self.cls_fc1 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256))
        self.cls_fc2 = nn.Linear(256, 128) #64
        self.bn2 = nn.BatchNorm1d(128) #64
        self.dropout = nn.Dropout(p=0.3)
        self.cls_fc3 = nn.Linear(128, cls_outchan) #64


        # self.rec_dens1 = nn.Sequential(nn.Linear(1024,2048),  nn.BatchNorm1d(2048))
        # self.rec_dens2 = nn.Sequential(nn.Linear(2048,2048),  nn.BatchNorm1d(2048))
        # self.rec_dens3_out = nn.Linear(2048,2048*3)

        self.rec_dens1 = nn.Sequential(nn.Linear(1024,512),  nn.BatchNorm1d(512)) #1024->2048
        self.rec_dens2 = nn.Sequential(nn.Linear(512,512),  nn.BatchNorm1d(512)) #2048->2048
        self.rec_dens3_out = nn.Linear(512,2048*3)#2048->2048*3



    def forward(self,xyz):
        glob_feat, _ = self.pcd_encoder(xyz)
        cls_out = self.cls_decode(glob_feat)
        rec_out = self.rec_decode(glob_feat)

        return {'pcd_rec': rec_out,'pcd_cls':cls_out}


    def rec_decode(self,x0):
        x0 = F.relu(self.rec_dens1(x0))
        x0 = F.relu(self.rec_dens2(x0))
        out = self.rec_dens3_out(x0)

        return out.view(-1, 2048, 3)

    def cls_decode(self, x):
        x = F.relu(self.cls_fc1(x))
        x = self.dropout(self.cls_fc2(x))
        x = self.bn2(x)
        x = F.relu(x)
        out = self.cls_fc3(x)

        # out = self.cls_acti(x)
        return out

    def inference(self,xyz):
        with torch.no_grad():
            glob_feat, _ = self.encode(xyz)
            return glob_feat




if __name__ == '__main__':
    from reconstruction.arguments import train_rec_parse_args

    args = train_rec_parse_args()
    args.model_name_or_path = '../contact2mesh/models/transformer/bert-base-uncased/'
    xyz = torch.rand(10, 3, 2048)
    contact = torch.rand(10, 1, 2048)

    # model = AeEncoder()
    # model = AeTransformer(t_args=args)
    model = AeConv1d()

    out = model(xyz, contact)
    a = 1
