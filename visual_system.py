import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PartialConv(nn.Module):

    def __init__(self, dim, n_div, forward):  # n_div 卷积的通道率
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.bn(x)
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PEREAttention(nn.Module):
    def __init__(self, c1, num_heads, head=True):
        super(PEREAttention, self).__init__()
        self.nums_head = num_heads
        self.head = head
        self.scaling = (c1 // self.nums_head) ** -0.5
        # self.Conv1 = PartialConv(c1, n_div=5, forward='split_cat')  # embedding
        # self.Conv2 = nn.Conv2d(c1, c1, 3, 2, 1, bias=False)
        self.norm = nn.LayerNorm(c1)
        self.q_Linear = nn.Linear(c1, c1, bias=False)  # 获取 Query , Key, Value
        self.K_Linear = nn.Linear(c1, c1, bias=False)
        self.V_Linear = nn.Linear(c1, c1, bias=False)
        # self.pos_block = PosCNN(c1, c1)
        self.proj = nn.Linear(c1, c1)
        # self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, c1 // num_heads, num_heads=num_heads)
        self.dropout = nn.Dropout(0.2)
        self.projout = nn.Dropout(0.2)
        # self.Linear = nn.Linear(2 * c1, c1, bias=False)
        # self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, H=None, W=None):
        # # x = self.qkv(x)
        # x = self.Conv1(x)
        # x = self.Conv2(x)  # b x h/2 x w/2 x c
        x = torch.flatten(x, start_dim=2).permute(0, 1, 2)  # b seq_len dim
        # print('input x shape', x.shape)
        x1 = x
        B, N, C = x.shape
        x = self.norm(x.permute(0, 1, 2))
        Q = self.q_Linear(x).reshape(B, self.nums_head, N, C // self.nums_head)
        K = self.K_Linear(x).reshape(B, self.nums_head, N, C // self.nums_head)
        V = self.V_Linear(x).reshape(B, self.nums_head, N, C // self.nums_head)
        attention = (Q @ K.transpose(-2, -1))
        # print('Q shape', Q.shape)
        # if self.rpe_k is not None:
        # pe_k = self.rpe_k(Q)
        # print('pe_k is ', pe_k)
        #   attention += self.rpe_k(Q)
        # if self.rpe_q is not None:
        # pe_q = self.rpe_q(K*self.scaling).transpose(2, 3)
        # print('pe_q is ', pe_q)
        #  attention += self.rpe_q(K * self.scaling).transpose(2, 3)
        attention = attention.softmax(dim=-1)
        # attention = self.dropout(attention)
        out = attention @ V
        # if self.rpe_v is not None:
        # pe_v = self.rpe_v(attention)
        # print('pe_v is ', pe_v)
        #    out += self.rpe_v(attention)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.projout(x)
        # x_relate = x1 * torch.softmax(self.Linear(torch.cat((x, x1), dim=-1)))
        # x_relate = x1 * torch.sigmoid((self.Linear(torch.cat((x, x1), dim=-1))))
        # x_relate = self.act(x_relate)
        x = x + x1
        # if self.head:
        #     x = self.pos_block(x, H, W)
        return x


class PereTransformer(nn.Module):
    def __init__(self, c1, nums_head, nums_attention, c2, upsample=True):
        super().__init__()
        self.nums_head = nums_head
        self.nums_attention = nums_attention
        self.c = c2
        self.upsample = upsample
        self.head_attention = PEREAttention(c1, nums_head, head=True)
        self.attention = nn.Sequential(*[PEREAttention(c1, nums_head, head=False) for _ in range(nums_attention - 1)])
        self.Conv1 = PartialConv(c1, n_div=4, forward='split_cat')
        self.Conv2 = Conv(c1, self.c // 2, 1, 1)
        # self.upConv = nn.ConvTranspose2d(c1, c2//2, 4, 2, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.Conv3 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = self.Conv1(x)
        x = self.Conv3(x)  # embedding

        x = torch.flatten(x, start_dim=2).permute(0, 2, 1)
        x = self.head_attention(x, H // 2, W // 2)

        output = self.attention(x).reshape(B, C, H // 2, W // 2)

        output = F.interpolate(output, size=(7, 7), mode='nearest')
        output = self.Conv2(output)
        output = output.reshape(B, self.c // 2, H, W)
        return output


class C2f_PERE(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.Transformer = PereTransformer(c1, 1, 1, c2=c2)
        self.cv2 = Conv((2 + n) * self.c, c2 // 2, 1)  # optional act=FReLU(c2)
        # self.cv3 = nn.Conv2d(c1, c2 // 2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.Transformer(x)
        return torch.cat((x, self.cv2(torch.cat(y, 1))), dim=1)

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Conv_LiRT(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.conv = nn.Conv2d(c1, c2 // 2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.transformer = PereTransformer(c1, 1, 1, c2=c2, upsample=False)
        self.bn = nn.BatchNorm2d(c2)
        self.conv1 = nn.Conv2d(c2 // 2, c2 // 2, 3, 2, padding=1)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(torch.cat((self.conv1(self.transformer(x)), self.conv(x)), dim=1)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class FSBGM_Datasets(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.data = pd.read_csv(annotations_file)
        self.image_root = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取当前索引的数据行
        row = self.data.iloc[idx]

        # 获取图像路径
        img_name = row['frame_name']
        img_path = os.path.join(self.image_root, "images", img_name)

        # 打开图像并应用变换
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # 获取测量值和误差
        measured_distance = row['measured_distance']
        label = row['error']

        # 返回图像及标签数据
        return image, measured_distance, label


class FSBGM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.to(device)
        self.conv2 = Conv_LiRT(512, 64, 3, 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, 20)  # 动态设置线性层
        self.output = nn.Linear(20 + 1, 1)

    def forward(self, x, row_distance):
        row_distance = row_distance.float()
        x = self.backbone(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        # 根据实际输入尺寸初始化线性层（仅初始化一次）
        row_distance = row_distance.unsqueeze(1)
        x = torch.cat((x, row_distance), dim=-1)

        x = self.output(x)

        return x


if __name__ == "__main__":
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-2]  # 去掉最后的全连接层和全局平均池化层
    resnet50_modified = nn.Sequential(*modules)
    for param in resnet50_modified.parameters():
        param.requires_grad = False
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FSBGM(resnet50_modified)
    model.to(device)
    epochs = 50
    training_datasets = FSBGM_Datasets('I:\\visual_system_data\\latastes\\FSGBM\\train\\distance_data.csv',
                                       'I:\\visual_system_data\\latastes\\FSGBM\\train', transform=transform)
    testing_datasets = FSBGM_Datasets('I:\\visual_system_data\\latastes\\FSGBM\\val\\distance_data.csv',
                                      'I:\\visual_system_data\\latastes\\FSGBM\\val', transform=transform)
    training_dataloader = DataLoader(training_datasets, batch_size=16)
    testing_dataloader = DataLoader(testing_datasets, batch_size=16)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00025)

    # 模拟输入
    train_losses = []
    test_losses = []
    for _ in range(epochs):
        model.train()
        for i, [data, raw_distance, labels] in enumerate(training_dataloader):
            data = data.to(device)
            raw_distance = raw_distance.to(device)
            labels = labels.to(device)
            outputs = model(data, raw_distance)
            loss = loss_fn(outputs.float(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = len(training_dataloader) * _ + i + 1
            if train_step % 100 == 0:
                print("train time: {}, loss: {}".format(train_step, loss.item()))
                train_losses.append(loss.item())

        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for datas, raw_distance, labels in testing_dataloader:
                datas = datas.to(device)
                raw_distance = raw_distance.to(device)
                labels = labels.to(device)
                outputs = model(datas, raw_distance)
                loss = loss_fn(outputs.float(), labels.float())
                total_test_loss += loss.item()
        avg_loss = total_test_loss / len(testing_dataloader)
        print('test avg loss: {}'.format(avg_loss))
        test_losses.append(avg_loss)
        if avg_loss <= torch.min(torch.Tensor(test_losses)):
            torch.save(model.state_dict(), "fsgbm.pth")
            print('save ')
