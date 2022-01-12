import torch.nn as nn
import torch.nn.functional as F
import torch
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
        
class Attention_block(nn.Module):
    """
    refer from attention u-net (https://arxiv.org/abs/1804.03999)
    """
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x):
        # down-sampling g conv used as gate signal
        g1 = self.W_g(g)
        # up-sampling l conv
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        result = x*psi*2
        # return re-weigted output
        return result

class Generator(nn.Module):
    """
    normal_cyclegan_generator
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Initial convolution block
        self.initial_block = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features*2
        # for _ in range(2):
        self.down_sampling1=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2
        self.down_sampling2=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2

        # Residual blocks
        tmp=[]
        for _ in range(n_residual_blocks):
            tmp += [ResidualBlock(in_features)]
        self.ResidualBlock = nn.Sequential(*tmp)
        # Upsampling
        out_features = in_features//2
        self.up_sampling1=nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        self.up_sampling2=nn.Sequential(nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        # Output layer
        self.Output_layer=nn.Sequential( nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() )

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('x',x.shape)
        initial=self.initial_block(x)
        # print('initial',initial.shape)
        down_sampling1=self.down_sampling1(initial)
        # print('down_sampling1',down_sampling1.shape)
        down_sampling2=self.down_sampling2(down_sampling1)
        # print('down_sampling2',down_sampling2.shape)
        res_out = self.ResidualBlock(down_sampling2)
        # print('res_out',res_out.shape)
        # sum_level3 = torch.cat((down_sampling2,res_out),di1m=1)
        up_sampling1 =self.up_sampling1(res_out)
        # print('up_sampling1',up_sampling1.shape)
        # sum_level2 = torch.cat((down_sampling1,up_sampling1),di1m=1)
        up_sampling2 =self.up_sampling2(up_sampling1)
        # print('up_sampling2',up_sampling2.shape)
        # sum_level1 = torch.cat((initial,up_sampling2),dim=1)
        output = self.Output_layer(up_sampling2)
        # print('output',output.shape)
        return output

class Generator3(nn.Module):
    """
    skip_connected_cyclegan_generator
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator3, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Initial convolution block
        self.initial_block = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features*2
        # for _ in range(2):
        self.down_sampling1=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2
        self.down_sampling2=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2

        # Residual blocks
        tmp=[]
        for _ in range(n_residual_blocks):
            tmp += [ResidualBlock(in_features)]
        self.ResidualBlock = nn.Sequential(*tmp)
        # Upsampling
        out_features = in_features//2
        self.up_sampling1=nn.Sequential(nn.ConvTranspose2d(in_features*2, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        self.up_sampling2=nn.Sequential(nn.ConvTranspose2d(in_features*2, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        # Output layer
        self.Output_layer=nn.Sequential( nn.ReflectionPad2d(3),
                    nn.Conv2d(64*2, output_nc, 7),
                    nn.Tanh() )


        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('x',x.shape)
        initial=self.initial_block(x)
        # print('initial',initial.shape)
        down_sampling1=self.down_sampling1(initial)
        # print('down_sampling1',down_sampling1.shape)
        down_sampling2=self.down_sampling2(down_sampling1)
        # print('down_sampling2',down_sampling2.shape)
        res_out = self.ResidualBlock(down_sampling2)
        # print('res_out',res_out.shape)
        sum_level3 = torch.cat((down_sampling2,res_out),dim=1)
        up_sampling1 =self.up_sampling1(sum_level3)
        # print('up_sampling1',up_sampling1.shape)
        sum_level2 = torch.cat((down_sampling1,up_sampling1),dim=1)
        up_sampling2 =self.up_sampling2(sum_level2)
        # print('up_sampling2',up_sampling2.shape)
        sum_level1 = torch.cat((initial,up_sampling2),dim=1)
        output = self.Output_layer(sum_level1)
        # print('output',output.shape)
        return output

class Generator4(nn.Module):
    """
    attention_cyclegan_generator
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator4, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Initial convolution block
        self.initial_block = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features*2
        # for _ in range(2):
        self.down_sampling1=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2
        self.down_sampling2=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2

        # Residual blocks
        tmp=[]
        for _ in range(n_residual_blocks):
            tmp += [ResidualBlock(in_features)]
        self.ResidualBlock = nn.Sequential(*tmp)
        # Upsampling
        out_features = in_features//2
        self.up_sampling1=nn.Sequential(
                    nn.ConvTranspose2d(in_features*2, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        self.up_sampling2=nn.Sequential(
                    nn.ConvTranspose2d(in_features*2, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        # Output layer
        self.Output_layer=nn.Sequential( nn.ReflectionPad2d(3),
                    nn.Conv2d(64*2, output_nc, 7),
                    nn.Tanh() )


        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('x',x.shape)
        initial=self.initial_block(x)
        # print('initial',initial.shape)
        down_sampling1=self.down_sampling1(initial)
        # print('down_sampling1',down_sampling1.shape)
        down_sampling2=self.down_sampling2(down_sampling1)
        # print('down_sampling2',down_sampling2.shape)
        res_out = self.ResidualBlock(down_sampling2)
        # print('res_out',res_out.shape)
        att3=self.Att3(g=res_out,x=down_sampling2)
        sum_level3 = torch.cat((att3,res_out),dim=1)
        up_sampling1 =self.up_sampling1(sum_level3)
        # print('up_sampling1',up_sampling1.shape)
        att2=self.Att2(g=up_sampling1,x=down_sampling1)
        sum_level2 = torch.cat((att2,up_sampling1),dim=1)
        up_sampling2 =self.up_sampling2(sum_level2)
        # print('up_sampling2',up_sampling2.shape)
        att1=self.Att1(g=up_sampling2,x=initial)
        sum_level1 = torch.cat((att1,up_sampling2),dim=1)
        output = self.Output_layer(sum_level1)
        # print('output',output.shape)
        return output

class Generator5(nn.Module):
    """
    attention_cyclegan_generator
    replace ConvTranspose2d with Up+Conv
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator5, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Initial convolution block
        self.initial_block = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features*2
        # for _ in range(2):
        self.down_sampling1=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2
        self.down_sampling2=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2

        # Residual blocks
        tmp=[]
        for _ in range(n_residual_blocks):
            tmp += [ResidualBlock(in_features)]
        self.ResidualBlock = nn.Sequential(*tmp)
        # Upsampling
        out_features = in_features//2
        self.up_sampling1=nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_features*2, out_features,  1, 1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        self.up_sampling2=nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#nearest
                    nn.Conv2d(in_features*2, out_features, 1, 1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        # Output layer
        self.Output_layer=nn.Sequential( nn.ReflectionPad2d(3),
                    nn.Conv2d(64*2, output_nc, 7),
                    nn.Tanh() )


        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('x',x.shape)
        initial=self.initial_block(x)
        # print('initial',initial.shape)
        down_sampling1=self.down_sampling1(initial)
        # print('down_sampling1',down_sampling1.shape)
        down_sampling2=self.down_sampling2(down_sampling1)
        # print('down_sampling2',down_sampling2.shape)
        res_out = self.ResidualBlock(down_sampling2)
        # print('res_out',res_out.shape)
        att3=self.Att3(g=res_out,x=down_sampling2)
        sum_level3 = torch.cat((att3,res_out),dim=1)
        up_sampling1 =self.up_sampling1(sum_level3)
        # print('up_sampling1',up_sampling1.shape)
        att2=self.Att2(g=up_sampling1,x=down_sampling1)
        sum_level2 = torch.cat((att2,up_sampling1),dim=1)
        up_sampling2 =self.up_sampling2(sum_level2)
        # print('up_sampling2',up_sampling2.shape)
        att1=self.Att1(g=up_sampling2,x=initial)
        sum_level1 = torch.cat((att1,up_sampling2),dim=1)
        output = self.Output_layer(sum_level1)
        # print('output',output.shape)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Conv_Block(nn.Module):
    def __init__(self, in_channels,  out_channels, act_func=nn.ReLU(inplace=True)):
        super(Conv_Block, self).__init__()
        self.act_func1 = act_func
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act_func2 = act_func
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.dp = nn.Dropout(0.6)

    def forward(self, x):
        out = self.conv1(x)
        out2 = self.bn1(out)
        # out = self.dp(out)
        out1 = self.act_func1(out2)

        out3 = self.conv2(out1)
        out4 = self.bn2(out3)

        out5 = self.act_func2(out4)

        return out5

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),    #尺寸扩大2倍
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),    #通道数改变
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        # self.args = args
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv_Block(input_nc, filters[0])
        self.Conv2 = Conv_Block(filters[0], filters[1])
        self.Conv3 = Conv_Block(filters[1], filters[2])
        self.Conv4 = Conv_Block(filters[2], filters[3])
        self.Conv5 = Conv_Block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = Conv_Block(filters[3]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = Conv_Block(filters[2]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = Conv_Block(filters[1]+filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = Conv_Block(filters[0]+filters[0], filters[0])

        self.final_map = nn.Sequential(
            nn.Conv2d(filters[0], output_nc,7,padding=3)
            # nn.Tanh()
            # nn.ConvTranspose2d(1, 1, 2, 2),
            #nn.Upsample(scale_factor=2,mode='nearest'),
            # nn.Sigmoid()
        )
        # self.v0 = nn.Conv2d(filters[0], 1, 1, 1)
        # self.v4 = nn.Conv2d(filters[0], 1, 1, 1)

    def forward(self, x):

        e1 = self.Conv1(x)
        # print(e1.shape)
        e2 = self.Maxpool1(e1)
        # print(e2.shape)
        # c1 = self.final_map0(e2).cpu().detach().numpy()
        # print(c1[0,0,:,:])
        # imsave('output/c1.png', (c1[0,0,:,:]))

        #print(e2.shape)
        e2 = self.Conv2(e2)

        # print(e2.shape)
        e3 = self.Maxpool2(e2)
        # c2 = self.final_map1(e3).cpu().detach().numpy()
        # imsave('output/c2.png', (c2[0,0,:,:]))
        # print(e3.shape)
        e3 = self.Conv3(e3)
        # print(e3.shape)
        e4 = self.Maxpool3(e3)
        # c3 = self.final_map2(e4).cpu().detach().numpy()
        # imsave('output/c3.png', (c3[0,0,:,:]))
        # print(e4.shape)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        # c4 = self.final_map3(e5).cpu().detach().numpy()
        # imsave('output/c4.png', (c4[0,0,:,:]))
        # for i in range(16):
        #     print(c4[0,0,i,:])

        # print(e5.shape)
        e5 = self.Conv5(e5)
        # print(e5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)
        # print(d5.shape)
        d4 = self.Up4(d5)
        # print(d4.shape)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # print(d4.shape)
        d3 = self.Up3(d4)
        # print(d3.shape)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # print(d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print(d2.shape)
        out = self.final_map(d2)

        # fv0= self.v0(e1)
        # fv4= self.v4(d2)
        # imsave('feature visualization/x00.png', (fv0[0,0,:,:].cpu()))
        # imsave('feature visualization/x01.png', (fv1[0,0,:,:].cpu()))
        # imsave('feature visualization/x02.png', (fv2[0,0,:,:].cpu()))
        # imsave('feature visualization/x03.png', (fv3[0,0,:,:].cpu()))
        # imsave('feature visualization/x04.png', (fv4[0,0,:,:].cpu()))
        # print(out.shape)
        # imsave('output/c55.png', (out[0,0,:,:].cpu().detach().numpy()))
        #d1 = self.active(out)

        return out
