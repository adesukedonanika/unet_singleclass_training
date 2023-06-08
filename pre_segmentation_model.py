
# (from versions: 0.0.1, 0.0.2, 0.0.3, 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.2.0, 0.2.1, 0.3.0, 0.3.1, 0.3.2)
# pip install segmentation-models-pytorch==0.2.1

import segmentation_models_pytorch as smp
import torch
import numpy as np
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.losses import DiceLoss


smp.DeepLabV3Plus()
# Unet
# Unet++
# MAnet
# Linknet
# FPN
# PSPNet
# PAN
# DeepLabV3
# DeepLabV3+

def UnetModel(encoder_name, encoder_weights, in_ch:int, outClass:int, activationName:str="softmax2d"):
    # 1クラスマスク＝2クラス分類になるため、　outClass=2が正しい
    model = smp.Unet(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_ch,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=outClass,                      # model output channels (number of classes in your dataset)
        # decoder_channels = [256, 128, 64, 32, 16] # default is [256, 128, 64, 32, 16]
        activation=activationName
    )
    return model

def UnetPlusPlusModel(encoder_name, encoder_weights, in_ch:int, outClass:int):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_ch,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=outClass,                      # model output channels (number of classes in your dataset)
        
    )
    return model


def calculate_iou(output, target):
    # 出力とターゲットのバイナリマスクを作成
    output_mask = (output > 0.5).float()
    target_mask = (target > 0.5).float()

    # 共通部分のピクセル数を計算
    intersection = torch.sum(output_mask * target_mask)

    # 合計ピクセル数を計算
    total_pixels = torch.sum(output_mask) + torch.sum(target_mask) - intersection

    # IoUを計算
    iou = intersection / (total_pixels + 1e-7)  # 0での除算を防ぐために小さな値を加算

    return iou

import segment_model_training

def validateModel(model,dataloader,workDir:str, epoch:int, means, stds, modelID):
    
    # 一般的には、2クラス分類の場合はバイナリクロスエントロピー損失関数BCEWithLogitsLossを使用します。
    # 損失関数と評価指標の設定
    loss = torch.nn.BCEWithLogitsLoss()#DiceLoss(mode='binary')
    metrics = IoU(threshold=0.5)

    # モデルの評価
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_iou = 0
        total_samples = 0

        for images, masks in dataloader:
            if torch.cuda.is_available():
                images = torch.autograd.Variable(images).cuda()
                masks = torch.autograd.Variable(masks).cuda()                            
            predicts = torch.sigmoid(model(images))#0~1の確率でPredを出力
            # print("predicts.shape",predicts.shape)
            # print("masks.shape", masks.shape, np.unique(masks.data.cpu()))
            segment_model_training.savePred(img_show=images[0],
                                            msk_show=masks[0],
                                            pred=predicts[0],
                                            mean_values=means,
                                            std_deviation=stds,
                                            workDir=workDir,
                                            epoch=epoch,
                                            modelID=modelID)

            batch_loss = loss(predicts, masks)
            batch_iou = metrics(predicts, masks)

            total_loss += batch_loss.item() * images.size(0)
            total_iou += batch_iou.item() * images.size(0)
            total_samples += images.size(0)

        average_loss = total_loss / total_samples
        average_iou = total_iou / total_samples

        print(f'Average Loss: {average_loss:.4f}')
        print(f'Average IoU: {average_iou:.4f}')
        
        
    return average_loss, average_iou