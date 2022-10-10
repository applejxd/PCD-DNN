import glob
import json
import os
import re
import shutil
import urllib.request

import numpy as np
import tensorflow as tf
import torch
from torch.utils import data
import trimesh

from modules import utils


def _get_pcd_from_mesh(file, max_num, max_diff):
    # 一様分布でサンプル点数を設定
    num_points = int(np.random.uniform(max_num - max_diff, max_num))
    padding_size = max_num - num_points

    # サンプリング
    points = np.array(trimesh.load(file).sample(num_points))
    points += np.random.uniform(-0.005, 0.005, size=(num_points, 3))
    points = np.insert(points, 3, 1, axis=1)

    # パディング
    padding_points = np.repeat([[0, 0, 0, 0]], padding_size, axis=0)
    points = np.concatenate([points, padding_points])

    np.random.shuffle(points)
    return points


class ModelNetTorchDataSet(torch.utils.data.Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.points = points
        self.labels = labels

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        return self.points[index], self.labels[index]


class ModelNetDataset:
    def __init__(self, class_num, cache=True):
        self.cache = cache
        self.class_num = class_num
        if class_num not in [10, 40]:
            raise RuntimeError("invalid class_num")

        # パス設定
        if self.class_num == 10:
            self.zip_url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        elif self.class_num == 40:
            self.zip_url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
        else:
            raise RuntimeError("invalid class_num")

        self._download_dataset()

    def _download_dataset(self):
        # zip ダウンロード
        target_path = f"./data/ModelNet{self.class_num}.zip"
        if not os.path.exists(target_path):
            # About 125 sec
            with utils.timer("zip download"):
                urllib.request.urlretrieve(self.zip_url, target_path)

        # zip 展開
        target_path = f"./data/ModelNet{self.class_num}"
        if not os.path.exists(target_path):
            # About 19 sec
            with utils.timer("unzip"):
                shutil.unpack_archive(f"{target_path}.zip", "./data")

    def get_data(self, max_num, max_diff):
        pcd_path = f"./data/ModelNet{self.class_num}_points_{max_num}_{max_diff}"
        map_path = f"./data/ModelNet{self.class_num}_classes.json"
        # README 以外のフォルダ取得
        folders = [p for p in glob.glob('./data/ModelNet10/*') if re.search('^(?!.*txt$).*$', p)]

        train_points, train_labels = [], []
        test_points, test_labels = [], []
        class_map = {}
        if self.cache and os.path.exists(f"{pcd_path}.npz") and os.path.exists(map_path):
            sampled_pcd = np.load(f"{pcd_path}.npz")
            train_points, train_labels = sampled_pcd["train_points"], sampled_pcd["train_labels"]
            test_points, test_labels = sampled_pcd["test_points"], sampled_pcd["test_labels"]

            with open(map_path) as fp:
                class_map = json.load(fp)
        else:
            for idx, folder in enumerate(folders):
                print(f"processing class: {os.path.basename(folder)}")
                # フォルダ名取得
                class_map[idx] = folder.split("/")[-1]
                # ファイル取得
                train_files = glob.glob(os.path.join(folder, "train/*"))
                test_files = glob.glob(os.path.join(folder, "test/*"))

                train_points += [_get_pcd_from_mesh(train_file, max_num, max_diff) for train_file in train_files]
                train_labels += [idx for _ in train_files]
                test_points += [_get_pcd_from_mesh(test_file, max_num, max_diff) for test_file in test_files]
                test_labels += [idx for _ in test_files]

            train_points, test_points = np.array(train_points), np.array(test_points)
            train_labels, test_labels = np.array(train_labels), np.array(test_labels)

            np.savez(pcd_path, train_points=train_points, test_points=test_points,
                     train_labels=train_labels, test_labels=test_labels)
            with open(map_path, "w") as fp:
                json.dump(class_map, fp)

        return train_points, test_points, train_labels, test_labels, class_map

    def get_tf_dataset(self, max_num, max_diff, batch_size=32, mask=True):
        train_points, test_points, train_labels, test_labels, class_map = self.get_data(max_num, max_diff)
        if not mask:
            train_points = train_points[:, :, :3]
            test_points = test_points[:, :, :3]

        # np.ndarray からデータセット (generator) 生成
        train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

        # データ順のシャッフル・(ノイズ追加・点のシャッフル)・バッチ化
        train_dataset = train_dataset.shuffle(len(train_points)).batch(batch_size)
        test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

        return train_dataset, test_dataset, class_map

    def get_torch_dataloader(self, max_num, max_diff, batch_size=32, mask=True):
        train_points, test_points, train_labels, test_labels, class_map = self.get_data(max_num, max_diff)
        if not mask:
            train_points = train_points[:, :, :3]
            test_points = test_points[:, :, :3]

        train_dataset = ModelNetTorchDataSet(train_points, train_labels)
        test_dataset = ModelNetTorchDataSet(test_points, test_labels)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader, class_map
