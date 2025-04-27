#!/usr/bin/env python3
"""
競馬データを用いて深層強化学習エージェント(DQN)を訓練するスクリプト。
各馬ごとにTop3予測を強化学習形式で学習します。
"""
import argparse
import time
import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from keiba_sanshutsu.ppo.horse_racing_env import HorseRacingEnv


def main():
    parser = argparse.ArgumentParser(description="競馬環境で深層強化学習エージェントを訓練する(DQN)")
    parser.add_argument("--host", type=str, default="localhost", help="MySQLホスト")
    parser.add_argument("--user", type=str, default="root", help="MySQLユーザ")
    parser.add_argument("--password", type=str, default="", help="MySQLパスワード")
    parser.add_argument("--database", type=str, default="mykeibadb", help="データベース名")
    parser.add_argument("--entry_table", type=str, default="umagoto_race_joho", help="出馬情報テーブル名")
    parser.add_argument("--result_table", type=str, default="record_master", help="結果情報テーブル名")
    parser.add_argument("--port", type=int, default=3306, help="MySQLポート番号")
    parser.add_argument("--timesteps", type=int, default=200000, help="学習ステップ数")
    parser.add_argument("--max_samples", type=int, default=None, help="使用するサンプル数上限")
    parser.add_argument("--lookback_number", type=int, default=1, help="取得期間の長さ（数値）")
    parser.add_argument("--lookback_unit", type=str, choices=["MONTH","YEAR"], default="MONTH", help="取得期間の単位（MONTHまたはYEAR）")
    args = parser.parse_args()

    # 環境生成
    print(f"[DQN] 環境生成開始: {datetime.datetime.now()}")
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
            max_samples=args.max_samples,
            lookback_number=args.lookback_number,
            lookback_unit=args.lookback_unit
        )
    ])
    env_end = time.time()
    print(f"[DQN] 環境生成終了: {datetime.datetime.now()} (所要 {env_end-env_start:.2f} 秒)")

    # DQNモデル生成
    print(f"[DQN] モデル生成開始: {datetime.datetime.now()}")
    model_start = time.time()
    model = DQN("MlpPolicy", env, verbose=1)
    model_end = time.time()
    print(f"[DQN] モデル生成終了: {datetime.datetime.now()} (所要 {model_end-model_start:.2f} 秒)")

    # 学習開始
    print(f"[DQN] 学習開始: {datetime.datetime.now()}")
    learn_start = time.time()
    model.learn(total_timesteps=args.timesteps)
    learn_end = time.time()
    print(f"[DQN] 学習終了: {datetime.datetime.now()} (所要 {learn_end-learn_start:.2f} 秒)")

    # モデル保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"dqn_racing_{timestamp}"
    model.save(save_path)
    print(f"[DQN] モデルを{save_path}.zipとして保存しました。")

    # 環境クローズ
    env.close()
    print(f"[DQN] 環境をクローズしました。")


if __name__ == "__main__":
    main() 