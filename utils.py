import torch

def calc_mean_std(feat, eps=1e-5):
    # eps는 매우작은 값으로, 분산이 0이되면 분모가 정규화 과정 중 분모가 0이 되는 경우가 발생하므로 이를 방지하기
    # 위해 더해주는 값
    size = feat.data.size() # [B, N, H, W]
    assert (len(size) == 4)
    N, C = size[:2] # N: mini-batch / C: Channel
    feat_var = feat.view(N, C, -1).var(dim=2) + eps # [N, C, -1] -> [N, C, H*W]
    # torch.var(input, dim) dim: the dimension to reduce
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptvie_instance_normalization(content_feat, style_feat):
    assert (content_feat.data.size()[:2] == style_feat.data.size()[:2]) #TODO batch, channel같게?
    size = content_feat.data.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

def _calc_feat_flatten_mean_std(feat): #TODO 어디에 쓰이는지 확인 그리고 그냥 calc_mean_std차이점
    # feat[C, H, W], channels을 가진 mean ,std arrary 반환
    assert (feat.size()[0] == 3) # color 인지 확인
    assert (isinstance(feat, torch.FloatTensor)) # feat가 Tensor인지 확인
    feat_flatten = feat.view(3, -1) # [3, H*W]
    mean = feat_flatten.mean(dim=-1, keepdim=True) # [3, 1]
    std = feat_flatten.std(dim=-1, keepdim=True) # [3, 1]
    return feat_flatten, mean, std

def _mat_sqrt(x):# TODO 어떻게 찍히나 확인
    U, D, V = torch.svd(x)
    # torch.svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)
    # input = U*diag(D)*V^T
    # SVD 특이값 분해
    # https://angeloyeo.github.io/2019/08/01/SVD.html
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())
    # torch.diag(input, diagonal): diagonal 행렬 (대각행렬)
    # https://pytorch.org/docs/stable/generated/torch.diag.html

def coral(source, target):
    # targer과 source가 3D array라고 가정(C, H, W)
    # flatten - >f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
    # normalization

    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3) #todo 이 식의 의미
    # torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    # 그냥 단위행렬 n: row의 수, m: columns의 수 default로 n가 동일

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye), # target_f_conv_Eye 행렬인지 찍어보자
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )
    # torch.inverse(input, *, out=None) → Tensor
    # 역변환 만드는 것

    source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)
    # AdaIn(x,y) = std(target)*(content norm) + mean(style)

    return source_f_transfer.view(source.size())











