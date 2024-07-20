import jittor as jt
from jittor import nn
from jittor import init
from jittor import weightnorm
import numpy as np
from models.embedder import get_embedder
from collections import OrderedDict, Mapping

np.random.seed(0)
# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if (l == (self.num_layers - 2)):
                    if (not inside_outside):
                        init.gauss_(lin.weight, mean=(np.sqrt(np.pi) / np.sqrt(dims[l])), std=0.0001)
                        init.constant_(lin.bias, value=(- bias))
                    else:
                        init.gauss_(lin.weight, mean=((- np.sqrt(np.pi)) / np.sqrt(dims[l])), std=0.0001)
                        init.constant_(lin.bias, value=bias)
                elif ((multires > 0) and (l == 0)):
                    init.constant_(lin.bias, value=0.0)
                    init.constant_(lin.weight[:, 3:], value=0.0)
                    init.gauss_(lin.weight[:, :3], mean=0.0, std=(np.sqrt(2) / np.sqrt(out_dim)))
                elif ((multires > 0) and (l in self.skip_in)):
                    init.constant_(lin.bias, value=0.0)
                    init.gauss_(lin.weight, mean=0.0, std=(np.sqrt(2) / np.sqrt(out_dim)))
                    init.constant_(lin.weight[:, (- (dims[0] - 3)):], value=0.0)
                else:
                    init.constant_(lin.bias, value=0.0)
                    init.gauss_(lin.weight, mean=0.0, std=(np.sqrt(2) / np.sqrt(out_dim)))
            if weight_norm:
                lin = weightnorm.weight_norm(module=lin, name='weight', dim=0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def execute(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = jt.concat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        # return torch.cat([x[:, :1] / self.scale, x[:, 1:2] / self.scale, x[:, 2:]], dim=-1)
        return jt.concat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)
    def sdf(self, x):
        return self.execute(x)[:, :1]

    def sdfM(self, x):
        return self.execute(x)[:, 1:2]

    def sdf_hidden_appearance(self, x):
        return self.execute(x)

    def gradient(self, x):
        x.requires_grad = True
        y = self.sdf(x)
        gradients = jt.grad(
            loss=y,
            targets=x,
            retain_graph=True)
        return gradients.unsqueeze(1)

    def gradientM(self, x):
        x.requires_grad = True
        y = self.sdfM(x)
        gradients = jt.grad(
            loss=y,
            targets=x,
            retain_graph=True)
        return gradients.unsqueeze(1)

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def execute(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return jt.sin(30 * input)

def sine_init(m):
    with jt.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with jt.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class BatchLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True):
        super(BatchLinear, self).__init__(in_features,out_features,bias)
        # self.stas = nn.AdaptiveAvgPool(1)
        self.variance2 = nn.Parameter(jt.array(0.5))
    def execute(self, input, params):
        bias = params.bias
        weight = params.weight

        wc = jt.ones([len(input), 1]) * self.variance2
        input += jt.squeeze(jt.mean(input.unsqueeze(0), dim=2, keepdims=True), 0)*(wc[:, :1].clamp(1e-6, 1e6))
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))

        output += bias.unsqueeze(-2)
        return output

# # This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature - 3] + [d_hidden for _ in range(n_layers)] + [d_out]

        # self.embedview_fn = None
        # if multires_view > 0:
        #     embedview_fn, input_ch = get_embedder(multires_view)
        #     self.embedview_fn = embedview_fn
        #     dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = weightnorm.weight_norm(module=lin, name='weight', dim=0)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def execute(self, points, normals, view_dirs, feature_vectors):
        # if self.embedview_fn is not None:
        #     view_dirs = self.embedview_fn(view_dirs)

        rendering_input = jt.concat([points, normals, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = jt.sigmoid(x)
        return x

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetworkM(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch*2 - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = BatchLinear(dims[l], out_dim)

            if l == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin)

            if weight_norm:
                lin = weightnorm.weight_norm(module=lin, name='weight', dim=0)

            setattr(self, "lin" + str(l), lin)

        self.relu = Sine()

    def execute(self, points, normals, view_dirs, lds, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            lds = self.embedview_fn(lds)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = jt.concat([jt.array(points), jt.array(view_dirs), jt.array(lds), jt.array(normals), jt.array(feature_vectors)], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = jt.concat([jt.array(points), jt.array(normals), jt.array(feature_vectors)], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = jt.concat([jt.array(points), jt.array(view_dirs), jt.array(feature_vectors)], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x,lin)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = jt.sigmoid(x)

        return x

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h =nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = nn.Parameter(jt.array(init_val))

    def execute(self, x):
        return jt.ones([len(x), 1]) * jt.exp(self.variance * 10.0)

class Pts_Bias(nn.Module):
    def __init__(self, d_hidden=256, multires=10, d_in=3):
        super(Pts_Bias, self).__init__()
        embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        self.embed_fn_fine = embed_fn

        self.pts_fea = nn.Sequential(nn.Linear(input_ch, d_hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_hidden, d_hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_hidden, d_hidden),
                                      nn.ReLU(inplace=True))

        self.pts_bias = nn.Sequential(nn.Linear(d_hidden, 3),
                                      nn.Tanh())


    def execute(self, x):
        x = self.embed_fn_fine(x)
        pts_fea = self.pts_fea(x)
        pts_bias = self.pts_bias(pts_fea)

        return pts_bias,pts_fea
