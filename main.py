import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
import open3d as o3d
import torch

from modules.dataset import ModelNetDataset
from modules import visualizer
import Pointnet_Pointnet2_pytorch.models.pointnet2_cls_ssg as pointnet


def get_accuracy(model, test_dataset, class_map, draw):
    data_num, true_num = 0, 0
    for dataset_idx in range(1, len(test_dataset) + 1):
        # パッチサイズ分だけ取得
        data = test_dataset.take(dataset_idx)
        point_clouds, labels = list(data)[0]

        # 各ラベルに対する予測確率を取得
        predictions = model.predict(point_clouds)
        # 確率が最も高いラベルを取得
        predictions = tf.math.argmax(predictions, -1)
        is_correct_list = predictions.numpy() == labels.numpy()
        false_indexes = np.arange(0, len(labels))[~is_correct_list]

        if draw:
            for false_index in false_indexes:
                points = np.array(point_clouds[false_index])
                points = points[points[:, 3] == 1][:, :3]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                predict_label = class_map[str(predictions[false_index].numpy())]
                true_label = class_map[str(labels.numpy()[false_index])]
                window_name = f"prediction: {predict_label}, label: {true_label}"
                o3d.visualization.draw_geometries([pcd], window_name=window_name)

        data_num += len(labels)
        true_num += np.count_nonzero(is_correct_list)

    accuracy = true_num / data_num
    print(f"Prediction accuracy: {accuracy}")


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main():
    # ------- #
    # Dataset #
    # ------- #

    max_num, max_diff = 1024, 128
    class_num = 40
    dataset = ModelNetDataset(class_num, cache=True)
    train_dataloader, test_dataloader, class_map = dataset.get_torch_dataloader(max_num, max_diff, mask=False)
    dataloaders_dict = {
        'train': train_dataloader,
        'valid': test_dataloader
    }

    # ---------- #
    # Load Model #
    # ---------- #

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = pointnet.get_model(class_num, normal_channel=False)
    model.apply(inplace_relu)
    criterion = pointnet.get_loss()
    model, criterion = model.cuda(), criterion.cuda()

    log_file_list = glob.glob("./output/fit/*")
    visualizer.open_tensorboard(log_file_list[-1])
    checkpoint = torch.load(
        './Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # ----------------------------- #
    # Configs for Transfer Learning #
    # ----------------------------- #

    # print(model)
    model.fc3 = torch.nn.Linear(in_features=256, out_features=class_num).cuda()

    # 転移学習で学習させるパラメータを、変数params_to_updateに格納
    params_to_update = []
    # 学習させるパラメータ名
    update_param_names = ['fc3.weight', 'fc3.bias']
    # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print('learning parameter name : ', name)
        else:
            param.requires_grad = False

    # ----------------- #
    # Transfer Learning #
    # ----------------- #

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # エポック数
    num_epochs = 20
    for epoch in range(num_epochs):
        mean_correct = []
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        scheduler.step()
        for phase in ['train', 'valid']:
            if phase == 'train':
                # 学習モードに設定
                model.train()
            else:
                # 訓練モードに設定
                model.eval()

            for batch_id, (inputs, labels) in enumerate(dataloaders_dict[phase], 0):
                # optimizerを初期化
                optimizer.zero_grad()

                inputs = inputs.transpose(2, 1)
                inputs, labels = inputs.cuda(), labels.cuda()

                # 学習時のみ勾配を計算させる設定にする
                with torch.set_grad_enabled(phase == 'train'):
                    pred, trans_feat = model(inputs.float())
                    # 損失を計算
                    loss = criterion(pred, labels.long(), trans_feat)
                    # ラベルを予測
                    pred_choice = pred.data.max(1)[1]

                    correct = pred_choice.eq(labels.long().data).cpu().sum()
                    mean_correct.append(correct.item() / float(inputs.size()[0]))
                    # 訓練時は逆伝搬の計算
                    if phase == 'train':
                        # 逆伝搬の計算
                        loss.backward()
                        # パラメータ更新
                        optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)

    # get_accuracy(model, test_dataset, class_map, draw=False)


if __name__ == "__main__":
    main()
