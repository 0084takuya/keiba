#!/usr/bin/env python3
"""
PPOモデルの特徴量重要度をヒートマップで可視化するスクリプト
"""
import argparse
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from keiba_sanshutsu.ppo.horse_racing_env import HorseRacingEnv


def main():
    parser = argparse.ArgumentParser(description="PPOモデルの特徴量重要度をヒートマップで表示")
    parser.add_argument("--model_path", type=str, default="horse_racing_ppo", help="モデルファイル（拡張子なし）のパス")
    parser.add_argument("--host", type=str, default="localhost", help="MySQLホスト")
    parser.add_argument("--user", type=str, default="root", help="MySQLユーザ")
    parser.add_argument("--password", type=str, default="", help="MySQLパスワード")
    parser.add_argument("--database", type=str, default="mykeibadb", help="データベース名")
    parser.add_argument("--entry_table", type=str, default="umagoto_race_joho", help="出馬情報テーブル名")
    parser.add_argument("--result_table", type=str, default="record_master", help="結果情報テーブル名")
    parser.add_argument("--port", type=int, default=3306, help="MySQLポート番号")
    parser.add_argument("--max_samples", type=int, default=1000, help="サンプル数上限（features取得用）")
    args = parser.parse_args()

    # 環境生成（features取得のみ）
    print(f"[PI] 環境生成開始: {datetime.datetime.now()}")
    env_start = time.time()
    env = DummyVecEnv([
        lambda: HorseRacingEnv(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database,
            entry_table=args.entry_table,
            result_table=args.result_table,
            port=args.port,
            max_samples=args.max_samples
        )
    ])
    env_end = time.time()
    print(f"[PI] 環境生成終了: {datetime.datetime.now()} (所要 {env_end-env_start:.2f} 秒)")

    # 学習済みモデルのロード（環境チェックをスキップ）
    print(f"[PI] モデルロード開始: {datetime.datetime.now()}")
    load_start = time.time()
    model = PPO.load(args.model_path)
    load_end = time.time()
    print(f"[PI] モデルロード終了: {datetime.datetime.now()} (所要 {load_end-load_start:.2f} 秒)")

    # 重み抽出と算出
    print(f"[PI] 重み算出開始: {datetime.datetime.now()}")
    wt_start = time.time()
    first_layer = model.policy.mlp_extractor.policy_net[0]
    weights = first_layer.weight.data.cpu().numpy()  # shape: (hidden_units, input_features)
    importances = np.abs(weights).sum(axis=0)
    wt_end = time.time()
    print(f"[PI] 重み算出終了: {datetime.datetime.now()} (所要 {wt_end-wt_start:.2f} 秒)")

    # 特徴量名を取得
    print(f"[PI] 特徴量取得開始: {datetime.datetime.now()}")
    feat_start = time.time()
    feature_names = env.envs[0].features  # DummyVecEnvの内部環境
    feat_end = time.time()
    print(f"[PI] 特徴量取得終了: {datetime.datetime.now()} (所要 {feat_end-feat_start:.2f} 秒)")

    # importancesとfeature_namesの長さを合わせる
    if len(feature_names) != len(importances):
        print(f"[PI] 警告: feature_names の長さ({len(feature_names)})と importances の長さ({len(importances)})が異なります。最小長に揃えます。")
    min_len = min(len(feature_names), len(importances))
    feature_names = feature_names[:min_len]
    importances = importances[:min_len]

    # DataFrame化して並び替え
    df_start = time.time()
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df_end = time.time()
    print(f"[PI] DataFrame構築所要: {df_end-df_start:.2f} 秒")

    # ヒートマップ描画
    plot_start = time.time()
    print(f"[PI] ヒートマップ描画開始: {datetime.datetime.now()}")
    # 色バー（カラーバー）付きで重要度を可視化
    plt.figure(figsize=(max(12, len(feature_names) * 0.3), 6))
    ax = sns.heatmap(
        df.set_index("feature").T,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Importance"},
        linewidths=0.5,
        linecolor="gray"
    )
    ax.set_title("Feature Importances Heatmap")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Feature")
    # X軸ラベルを90度回転して見やすく
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plot_end = time.time()
    print(f"[PI] ヒートマップ描画終了: {datetime.datetime.now()} (所要 {plot_end-plot_start:.2f} 秒)")


if __name__ == "__main__":
    main() 