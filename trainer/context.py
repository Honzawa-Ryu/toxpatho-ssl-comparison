# -*- coding: utf-8 -*-
"""
# 学習の実行コンテキスト

旧 scripts/train/train_tggate.py のモジュールグローバルを集約したもの
（REFACTOR_PLAN.md §6-1 / §6-2 / Phase 1-a-b）。

"""
import os
import json
import datetime
import argparse
from dataclasses import dataclass
from typing import Any

import torch

import lib.sslmodel as sslmodel
import lib.model.zoo as zoo


@dataclass
class RunContext:
    """1回の学習実行が必要とする状態をまとめたもの。

    以前はこれらがすべてモジュールグローバル（`args` / `PROJECT_PATH` / `ssl_class` /
    `DEVICE` / `DIR_NAME` / `LOGGER` / `file_log`）で、各関数が引数ではなく
    グローバルを直接読んでいた。そのままではこのファイルから関数を切り出せないため、
    明示的に引き回す形に変更した（REFACTOR_PLAN.md §6-1 / Phase 1-a）。

    ファイル分割そのものは Phase 1-b で行う。ここでは配線だけを変える。
    """
    args: argparse.Namespace
    project_path: str
    dir_name: str          # 出力先ディレクトリ（旧 DIR_NAME）
    file_log: str          # logger.pkl のパス（旧 file_log）
    logger: Any            # sslmodel.utils.logger_save インスタンス（旧 LOGGER）
    device: torch.device   # 旧 DEVICE
    ssl_class: Any         # 旧 ssl_class

    @property
    def log(self):
        """`ctx.log.info(...)` と書けるようにするショートカット（旧 LOGGER.logger）。"""
        return self.logger.logger


def build_context(args, filename: str = 'train_tggate') -> RunContext:
    """引数から RunContext を組み立てる。

    以前 `if __name__ == '__main__':` ブロックが直接行っていた初期化
    （シード固定・スレッド数・出力ディレクトリ作成・ロガー初期化・config.json 保存・
    DEVICE 決定・SSLクラス構築・multicrop 設定）をここに集約した。
    `import` しただけでは何も起きず、この関数を呼んだときだけ初期化される
    （REFACTOR_PLAN.md §6-2 / Phase 1-a）。
    """
    sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control
    # Per-image CPU augmentation in DataLoader workers must not oversubscribe threads:
    # torch defaults to all cores, so N workers x 32 threads thrash the allocated CPUs
    # (all-cores 100%, ~16x slower aug). Pin intra-op parallelism to 1 (workers inherit).
    torch.set_num_threads(1)

    # path setting
    project_path = args.project_path
    # --dir_result は run_slurm.sh の世代によって絶対パス（新: ${OUTPUT_DIR} =
    # ${PROJECT_ROOT}/outputs/${EXP_NAME}）と相対パス（旧: outputs/${EXP_NAME}）の
    # 両方が渡される。素の文字列連結だと絶対パスのときに
    # {project_path}/result/{project_path}/outputs/... という二重ネストになるため
    # os.path.join を使う（第2引数以降が絶対パスならそれ以前を捨てる仕様）。
    #   絶対パス -> そのまま {OUTPUT_DIR}
    #   相対パス -> 従来通り {project_path}/result/{dir_result}
    dir_name = os.path.join(project_path, 'result', args.dir_result) # for output
    file_log = f'{dir_name}/logger.pkl'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    now = datetime.datetime.now().strftime('%H%M%S')
    logger = sslmodel.utils.logger_save()
    logger.init_logger(filename, dir_name, now, level_console='debug')

    # 実行条件（引数）を JSON ファイルに保存し、ログ（標準出力）にもダンプする
    config_json_path = os.path.join(dir_name, 'config.json')
    try:
        with open(config_json_path, 'w') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)
        logger.logger.info(f"Saved run configuration to {config_json_path}")
    except Exception as e:
        logger.logger.warning(f"Failed to save run configuration: {e}")

    logger.logger.info(f"Execution Arguments:\n{json.dumps(vars(args), indent=4)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    # Set SSL class
    ssl_class = zoo.DICT_SSL[args.ssl_name](DEVICE=device)
    # multicrop config (read by SwaV/DINO prepare_transform + prepare_model)
    ssl_class.n_global_crops = args.n_global_crops
    ssl_class.n_local_crops = args.n_local_crops
    ssl_class.local_crop_size = args.local_crop_size

    return RunContext(
        args=args, project_path=project_path, dir_name=dir_name, file_log=file_log,
        logger=logger, device=device, ssl_class=ssl_class,
    )
