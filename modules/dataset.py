import glob
import json
import os
import shutil
import urllib.request

import numpy as np
import trimesh

from modules import utils


def _get_model_net_data():
    zip_url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = "./data/model_net.zip"
    if not os.path.exists(zip_path):
        # About 125 sec
        with utils.timer("zip download"):
            urllib.request.urlretrieve(zip_url, zip_path)

    if not os.path.exists("./data/ModelNet10"):
        # About 19 sec
        with utils.timer("unzip"):
            shutil.unpack_archive("./data/model_net.zip", "./data")


def _get_pcd_from_mesh(file, mean, diff):
    # 一様分布でサンプル点数を設定
    num_min = int(mean - diff / 2)
    num_max = int(mean + diff / 2)
    num_points = int(np.random.uniform(num_min, num_max))
    padding_size = num_max - num_points

    # サンプリング
    points = np.array(trimesh.load(file).sample(num_points))
    points += np.random.uniform(-0.005, 0.005, size=(num_points, 3))
    points = np.insert(points, 3, 1, axis=1)

    # パディング
    padding_points = np.repeat([[0, 0, 0, 0]], padding_size, axis=0)
    points = np.concatenate([points, padding_points])

    np.random.shuffle(points)
    return points


def get_np_dataset(mean, diff):
    _get_model_net_data()

    # README 以外のフォルダ取得
    file_path = "./data/ModelNet10"
    folders = glob.glob(os.path.join(file_path, "[!README]*"))

    train_points, train_labels = [], []
    test_points, test_labels = [], []
    class_map = {}
    pcd_path, map_path = "./data/sampled_pcd", "./data/class_map.json"
    if os.path.exists(f"{pcd_path}.npz") and os.path.exists(map_path):
        sampled_pcd = np.load(f"{pcd_path}.npz")
        train_points = sampled_pcd["train_points"]
        train_labels = sampled_pcd["train_labels"]
        test_points = sampled_pcd["test_points"]
        test_labels = sampled_pcd["test_labels"]

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

            for train_file in train_files:
                points = _get_pcd_from_mesh(train_file, mean=mean, diff=diff)
                train_points.append(points)
                train_labels.append(idx)

            for test_file in test_files:
                points = _get_pcd_from_mesh(test_file, mean=mean, diff=diff)
                test_points.append(points)
                test_labels.append(idx)

        train_points = np.array(train_points)
        test_points = np.array(test_points)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        np.savez(pcd_path, train_points=train_points, test_points=test_points,
                 train_labels=train_labels, test_labels=test_labels)
        with open(map_path, "w") as fp:
            json.dump(class_map, fp)

    return train_points, test_points, train_labels, test_labels, class_map
