import torch
import torch.nn as nn


class BCNN(nn.Module):
    def __init__(self, in_channels=8, n_class=6):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)

        # conv
        self.conv0_1 = nn.Conv2d(
            in_channels, 24, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(
            24, 24, kernel_size=3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(
            24, 48, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            48, 48, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(
            48, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(
            64, 96, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(
            96, 128, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(
            128, 192, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1)

        # deconv
        self.deconv5_1 = nn.ConvTranspose2d(
            192, 192, kernel_size=3, stride=1, padding=1)

        self.deconv4 = nn.ConvTranspose2d(
            192, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4_1 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=1, padding=1)

        self.deconv3 = nn.ConvTranspose2d(
            128, 96, kernel_size=4, stride=2, padding=1)
        self.deconv3_1 = nn.ConvTranspose2d(
            192, 96, kernel_size=3, stride=1, padding=1)

        self.deconv2 = nn.ConvTranspose2d(
            96, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2_1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(
            64, 48, kernel_size=4, stride=2, padding=1)
        self.deconv1_1 = nn.ConvTranspose2d(
            96, 48, kernel_size=3, stride=1, padding=1)

        self.deconv0 = nn.ConvTranspose2d(
            48, n_class + 6, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # conv
        conv0 = self.relu(self.conv0(
            self.relu(self.conv0_1(x))))

        conv1 = self.relu(self.conv1(
            self.relu(self.conv1_1(conv0))))

        conv2 = self.relu(self.conv2(
            self.relu(self.conv2_2(
                self.relu(self.conv2_1(conv1))))))

        conv3 = self.relu(self.conv3(
            self.relu(self.conv3_2(
                self.relu(self.conv3_1(conv2))))))

        conv4 = self.relu(self.conv4(
            self.relu(self.conv4_2(
                self.relu(self.conv4_1(conv3))))))

        conv5 = self.relu(self.conv5(
            self.relu(self.conv5_2(
                self.relu(self.conv5_1(conv4))))))

        # deconv
        deconv5_1 = self.relu(self.deconv5_1(conv5))

        deconv4 = self.relu(self.deconv4(deconv5_1))
        concat4 = torch.cat([conv4, deconv4], dim=1)
        deconv4_1 = self.relu(self.deconv4_1(concat4))

        deconv3 = self.relu(self.deconv3(deconv4_1))
        concat3 = torch.cat([conv3, deconv3], dim=1)
        deconv3_1 = self.relu(self.deconv3_1(concat3))

        deconv2 = self.relu(self.deconv2(deconv3_1))
        concat2 = torch.cat([conv2, deconv2], dim=1)
        deconv2_1 = self.relu(self.deconv2_1(concat2))

        deconv1 = self.relu(self.deconv1(deconv2_1))
        concat1 = torch.cat([conv1, deconv1], dim=1)
        deconv1_1 = self.relu(self.deconv1_1(concat1))

        deconv0 = self.deconv0(deconv1_1)

        category = deconv0[:, 0, :, :]
        instance = deconv0[:, 1:3, :, :]
        confidence = deconv0[:, 3, :, :]
        classify = deconv0[:, 4:10, :, :]
        heading = deconv0[:, 10, :, :]
        height = deconv0[:, 11, :, :]

        return category, instance, confidence, classify, heading, height
