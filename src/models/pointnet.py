"""
PointNet model for semantic segmentation
Original implementation with feature transformation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def feature_transform_reguliarzer(trans):
    """
    Regularization loss for feature transformation matrix
    
    Args:
        trans: transformation matrix [B, K, K]
    
    Returns:
        regularization loss
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class STN3d(nn.Module):
    """
    Spatial Transformer Network for 3D coordinates
    """
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Safety check for batch size 1 during training
        if batchsize == 1 and self.training:
            # Use eval mode for BatchNorm when batch size is 1
            self.bn4.eval()
            self.bn5.eval()
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            self.bn4.train()
            self.bn5.train()
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """
    Spatial Transformer Network for k-dimensional features
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Safety check for batch size 1 during training
        if batchsize == 1 and self.training:
            # Use eval mode for BatchNorm when batch size is 1
            self.bn4.eval()
            self.bn5.eval()
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            self.bn4.train()
            self.bn5.train()
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet feature encoder
    """
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetObjectClassifier(nn.Module):
    """
    PointNet model for object classification
    """
    def __init__(self, num_classes, feature_transform=False, channel=3):
        super(PointNetObjectClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        
        # Feature extraction backbone
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform, channel=channel)
        
        # Classification head for object classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass for object classification"""
        batchsize = x.size()[0]
        
        # Extract global features using PointNet encoder
        x, trans, trans_feat = self.feat(x)  # x: [B, 1024]
        
        # Classification head
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # [B, num_classes]
        
        return x, trans_feat


class PointNetSemanticSegmentation(nn.Module):
    """
    PointNet model for semantic segmentation (per-point classification)
    """
    def __init__(self, num_classes, feature_transform=False, channel=3):
        super(PointNetSemanticSegmentation, self).__init__()
        self.num_classes = num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform=feature_transform, channel=channel)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batchsize, n_pts, self.num_classes)
        return x, trans_feat


class ObjectClassificationLoss(torch.nn.Module):
    """
    Loss function for object classification with optional feature transform regularization
    """
    def __init__(self, mat_diff_loss_scale=0.001):
        super(ObjectClassificationLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):
        # Classification loss
        loss = F.cross_entropy(pred, target)
        
        # Feature transform regularization (if enabled)
        if trans_feat is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        else:
            total_loss = loss
            
        return total_loss


class SemanticSegmentationLoss(torch.nn.Module):
    """
    Loss function for semantic segmentation with optional feature transform regularization
    """
    def __init__(self, mat_diff_loss_scale=0.001):
        super(SemanticSegmentationLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight=None):
        loss = F.nll_loss(pred, target, weight=weight)
        if trans_feat is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        else:
            total_loss = loss
        return total_loss


def create_pointnet_classifier(num_classes, feature_transform=True, channel=3):
    """
    Factory function to create a PointNet object classifier
    
    Args:
        num_classes: int - number of object classes
        feature_transform: bool - whether to use feature transformation
        channel: int - input channel dimension (3 for XYZ)
    
    Returns:
        model: PointNetObjectClassifier
        criterion: ObjectClassificationLoss
    """
    model = PointNetObjectClassifier(
        num_classes=num_classes,
        feature_transform=feature_transform,
        channel=channel
    )
    
    criterion = ObjectClassificationLoss()
    
    return model, criterion


def create_pointnet_segmentation(num_classes, feature_transform=True, channel=3):
    """
    Factory function to create a PointNet semantic segmentation model
    
    Args:
        num_classes: int - number of semantic classes
        feature_transform: bool - whether to use feature transformation
        channel: int - input channel dimension (3 for XYZ)
    
    Returns:
        model: PointNetSemanticSegmentation
        criterion: SemanticSegmentationLoss
    """
    model = PointNetSemanticSegmentation(
        num_classes=num_classes,
        feature_transform=feature_transform,
        channel=channel
    )
    
    criterion = SemanticSegmentationLoss()
    
    return model, criterion




