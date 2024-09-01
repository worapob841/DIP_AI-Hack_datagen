# Ref. https://www.youtube.com/watch?v=nkQHASviYac
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# For generator kernel size
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# Weighted Scaled Convolution


class WSConv2d(nn.Module):
    # gain : for the initialize constant square root of 2 divided by kernel size square time in_channels
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.scale = (gain/(in_channels*(kernel_size ** 2)))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale)+self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3,num_class=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(num_class+1, num_class)
        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(
            len(factors) - 1
        ):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, labels, alpha, steps):
        # x = x.view(-1,self.z_dim)
        c = self.label_emb(labels)
        # print('c',c.shape,c.min(),c.max(),c.dtype)
        c = c.view(c.shape[0], c.shape[1], 1, 1)
        # print('c',c.shape,c.min(),c.max(),c.dtype)
        x = torch.cat((x, c), dim=1)
        # print('x',x.shape,x.min(),x.max(),x.dtype)
        # print()


        out = self.initial(x)
        # # for imagesize 64
        # steps = 4
        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)



# class Generator(nn.Module):
#     def __init__(self, z_dim, in_channels, img_channels=3):
#         super().__init__()
#         self.initial = nn.Sequential(
#             PixelNorm(),
#             nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1x1 -> 4x4
#             nn.LeakyReLU(0.2),
#             WSConv2d(in_channels, in_channels,
#                      kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.2),
#             PixelNorm(),
#         )

#         # to RGB
#         self.initial_rgb = WSConv2d(
#             in_channels, img_channels, kernel_size=1, stride=1, padding=0)

#         # progressive block
#         self.prog_blocks, self.rgb_layers = (
#             nn.ModuleList([]), nn.ModuleList([self.initial_rgb])
#         )

#         for i in range(len(factors)-1):
#             # The reason why len(factors) -1
#             # factor of i -> factor[i+1]

#             # conv in channel
#             conv_in_c = int(in_channels * factors[i])
#             # conv out channel
#             conv_out_c = int(in_channels * factors[i+1])
#             self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
#             self.rgb_layers.append(
#                 WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

#     def fade_in(self, alpha, upscaled, generated):
#         return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

#     def forward(self, x, alpha, steps):  # steps=0 (4x4), steps=1 (8x8), ...
#         out = self.initial(x)  # 4x4

#         if steps == 0:
#             return self.initial_rgb(out)

#         for step in range(steps):
#             upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
#             out = self.prog_blocks[step](upscaled)

#         final_upscaled = self.rgb_layers[steps - 1](upscaled)
#         final_out = self.rgb_layers[steps](out)
#         return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3,num_class=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_class, num_class)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(
            []), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1, 0, -1):
            conv_in_c = int(in_channels*factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(
                ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        # From RGB
        # This is for 4x4 img resolution
        self.inital_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.inital_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # block for 4x4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels,
                     kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha)*downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(
            x.shape[0], 1, x.shape[2], x.shape[3])  # N x C x H x W - > N
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):  # steps=0 (4x4), steps=1 (8x8), ...
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step+1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3, num_class=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_class, num_class)

        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Modify the initialization to account for additional channels due to label embeddings
        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(
                ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels + num_class, conv_in_c, kernel_size=1, stride=1, padding=0)
            )

        # Initial RGB layer for 4x4 image resolution
        self.initial_rgb = WSConv2d(
            img_channels + num_class, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block for 4x4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels,
                     kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(
            x.shape[0], 1, x.shape[2], x.shape[3])  # N x C x H x W - > N
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, labels, alpha, steps):
        # Embed and reshape labels to concatenate with images
        c = self.label_emb(labels)
        # print('c',c.shape,c.min(),c.max(),c.dtype)
        c = c.view(c.shape[0], c.shape[1], 1, 1)
        c = c.repeat(1, 1, x.shape[2], x.shape[3])
        # print('c',c.shape)
        # print('x',x.shape)
        # c = self.label_emb(labels).view(labels.shape[0], labels.shape[1], 1, 1)
        # c = c.repeat(1, 1, x.shape[2], x.shape[3])  # Match the height and width of the image
        x = torch.cat((x, c), dim=1)
        # Concatenate the label embedding with the image
        # x = torch.cat([x, c], dim=1)

        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    z_dim = 100
    in_ch = 256
    gen = Generator(z_dim, in_ch, img_channels=3)
    critic = Discriminator(z_dim, in_ch, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size/4))
        x = torch.randn((1, z_dim, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success ! At image size : {img_size}")
