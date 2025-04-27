#!/usr/bin/env python3
"""
特徴量間および特徴量とラベルの相関係数をヒートマップで可視化するスクリプト
"""
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
# プロジェクトルートからppoディレクトリをパスに追加してHorseRacingEnvをインポート
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from horse_racing_env import HorseRacingEnv


def main():
    parser = argparse.ArgumentParser(description="特徴量とラベルの相関係数をヒートマップで表示")
    parser.add_argument("--host", type=str, default="localhost", help="MySQLホスト")
    parser.add_argument("--user", type=str, default="root", help="MySQLユーザ")
    parser.add_argument("--password", type=str, default="", help="MySQLパスワード")
    parser.add_argument("--database", type=str, default="mykeibadb", help="データベース名")
    parser.add_argument("--entry_table", type=str, default="umagoto_race_joho", help="出馬情報テーブル名")
    parser.add_argument("--result_table", type=str, default="record_master", help="結果情報テーブル名")
    parser.add_argument("--port", type=int, default=3306, help="MySQLポート番号")
    parser.add_argument("--max_samples", type=int, default=1000, help="サンプル数上限")
    parser.add_argument("--lookback_number", type=int, default=1, help="取得期間の長さ（数値）")
    parser.add_argument("--lookback_unit", type=str, choices=["MONTH","YEAR"], default="MONTH", help="取得期間の単位（MONTHまたはYEAR）")
    args = parser.parse_args()

    # 環境を生成してデータをロード
    env = DummyVecEnv([
        lambda: HorseRacingEnv(
            host=args.host,
            user=args.user,
            password=args.password,
            database=args.database,
            entry_table=args.entry_table,
            result_table=args.result_table,
            port=args.port,
            max_samples=args.max_samples,
            lookback_number=args.lookback_number,
            lookback_unit=args.lookback_unit
        )
    ])
    # 内部環境のDataFrameを取得
    df = env.envs[0].data.copy()
    env.close()

    # 相関行列を算出（数値列のみ）
    corr_mat = df.corr()

    # 特徴量をグループ化（列名を最後の"_"で分割）
    groups = {}
    for col in corr_mat.columns:
        grp = col.rsplit('_', 1)[0] if '_' in col else col
        groups.setdefault(grp, []).append(col)

    # グループ間平均相関行列を生成
    group_names = list(groups.keys())
    group_corr = pd.DataFrame(index=group_names, columns=group_names, dtype=float)
    for gi in group_names:
        for gj in group_names:
            if gi == gj:
                group_corr.loc[gi, gj] = 1.0
            else:
                vals = corr_mat.loc[groups[gi], groups[gj]].values.flatten()
                group_corr.loc[gi, gj] = float(vals.mean()) if len(vals) > 0 else 0.0

    # 相関係数の高いグループ間関係を要約して出力
    print("[Summary] 強い相関関係を持つグループペア（|相関| >= 0.7）:")
    threshold = 0.7
    for i, gi in enumerate(group_names):
        for j, gj in enumerate(group_names):
            if i < j:
                corr_val = group_corr.loc[gi, gj]
                if abs(corr_val) >= threshold:
                    relation = "正の相関" if corr_val > 0 else "負の相関"
                    print(f" - {gi} と {gj}: {relation} (相関係数 {corr_val:.2f})")

    # グループ化相関行列を描画
    plt.figure(figsize=(len(group_names)*1.2, len(group_names)*1.2))
    sns.heatmap(
        group_corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Group Correlation'}
    )
    plt.title('Group-level Correlation Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"group_correlation_{args.lookback_number}{args.lookback_unit}.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main() 