#!/usr/bin/env python3
"""
競馬データを用いて深層強化学習エージェント(PPO)を訓練するスクリプト。
各馬ごとにTop3予測を強化学習形式で学習します。
"""
import argparse
import time
import datetime
import os
import sys
import io
import contextlib
import openai
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
# patch_gymによるGym→Gymnasium互換警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.vec_env.patch_gym")
# プロジェクトルートをパスに追加してHorseRacingEnvをインポート
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from horse_racing_env import HorseRacingEnv
from dotenv import load_dotenv
import re, json

# .envファイルから環境変数を読み込む
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path)


def main():
    parser = argparse.ArgumentParser(description="競馬環境で深層強化学習エージェントを訓練する")
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

    # lookback期間をわかりやすく表示
    unit_str = "ヶ月" if args.lookback_unit == "MONTH" else "年"
    print(f"[Main] データ取得期間: 過去{args.lookback_number}{unit_str}")

    # GPT API で総評を取得
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY が設定されていません")

    # ベクトル化環境の生成
    print("[Main] ベクトル化環境生成開始")
    env_start = time.time()
    env_start_dt = datetime.datetime.now()
    print(f"[Main] 環境生成開始時刻: {env_start_dt}")
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
    env_end_dt = datetime.datetime.now()
    print(f"[Main] 環境生成終了時刻: {env_end_dt}")
    print(f"[Main] 環境生成時間: {env_end - env_start:.2f} 秒")
    print("[Main] ベクトル化環境生成完了")

    # PPOエージェントの作成（開始・終了時刻と時間計測）
    print("[Main] PPOモデル生成開始")
    model_start = time.time()
    model_start_dt = datetime.datetime.now()
    print(f"[Main] PPOモデル生成開始時刻: {model_start_dt}")
    model = PPO("MlpPolicy", env, verbose=1)
    model_end = time.time()
    model_end_dt = datetime.datetime.now()
    print(f"[Main] PPOモデル生成終了時刻: {model_end_dt}")
    print(f"[Main] PPOモデル生成時間: {model_end - model_start:.2f} 秒")
    print("[Main] PPOモデル生成完了")

    # 学習開始
    print("[Main] 学習開始")
    start_time = time.time()
    print(f"[Main] 学習開始時刻: {datetime.datetime.now()}")
    # 学習実行とログキャプチャ
    print("[Main] 学習処理中(ログをキャプチャ)...")
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        model.learn(total_timesteps=args.timesteps)
    # キャプチャしたログを表示
    log_content = log_buffer.getvalue()
    print(log_content)
    # テーブル部分を抽出してJSONに変換
    table_dict = {}
    current_group = None
    for line in log_content.splitlines():
        # グループヘッダ行を検出
        m = re.match(r"^\|\s*([^|]+/)\s*\|", line)
        if m:
            grp = m.group(1).strip()
            if grp.endswith('/'):
                current_group = grp[:-1]
            continue
        # メトリクス行を検出
        if line.strip().startswith('|'):
            parts = line.strip().strip('|').split('|')
            if len(parts) >= 2:
                k = parts[0].strip()
                v = parts[1].strip()
                if k and v:
                    key = f"{current_group}/{k}" if current_group else k
                    # 値を数値化
                    try:
                        val = int(v) if v.isdigit() else float(v)
                    except:
                        val = v
                    table_dict[key] = val
    # JSON文字列作成
    json_table = json.dumps(table_dict, ensure_ascii=False)
    print("[Main] 抽出したメトリクステーブルをJSON形式で出力")
    print(json_table)
    # GPT API で評価
    print("[Main] GPTによる総評を取得中...")
    prompt = f"以下は学習結果メトリクスをJSON形式で表したものです。学習の良し悪しを簡潔に説明してください。\n{json_table}\n評価:"
    print(f"[Main] GPT評価開始: {prompt}")
    response = openai.ChatCompletion.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "あなたは強化学習(RL)モデル評価の専門家です。"},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000,
    )
    evaluation = response.choices[0].message.content.strip()
    print("--- GPT評価結果 ---")
    print(evaluation)
    end_time = time.time()
    print(f"[Main] 学習終了時刻: {datetime.datetime.now()}")
    elapsed = end_time - start_time
    print(f"[Main] 総学習時間: {elapsed:.2f} 秒 ({elapsed/3600:.2f} 時間)")
    print("[Main] 学習完了")

    # モデル保存（末尾に日時を付与）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/horse_racing_ppo_{args.lookback_number}{args.lookback_unit}_{timestamp}"
    model.save(save_path)
    print(f"[Main] モデルを{save_path}.zipとして保存しました。")
    print("[Main] モデル保存完了")

    print("[Main] 環境クローズ開始")
    env.close()
    print("[Main] 環境クローズ完了")


if __name__ == "__main__":
    main() 