import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pymysql
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import base64
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import font_manager
import time
from sanshutsu_kun.modules.plot_utils import plot_and_save_graphs
from sanshutsu_kun.modules.gpt_utils import ask_gpt41
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# --- .envからAPIキーをロード ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# --- DB接続設定 ---
DB_HOST = os.getenv('KEIBA_DB_HOST', 'localhost')
DB_USER = os.getenv('KEIBA_DB_USER', 'root')
DB_PASS = os.getenv('KEIBA_DB_PASS', '')
DB_NAME = os.getenv('KEIBA_DB_NAME', 'mykeibadb')
DB_PORT = int(os.getenv('KEIBA_DB_PORT', 3306))

def fetch_data():
    print('=== fetch_data関数 実行開始 ===')
    start_time = time.time()
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    today = datetime.today()
    three_years_ago = today - timedelta(days=365*3)
    date_min = three_years_ago.strftime('%Y%m%d')
    # --- 騎手3着率 ---
    cursor.execute('''SELECT KISHU_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ("1","2","3") THEN 1 ELSE 0 END) FROM umagoto_race_joho WHERE KISHU_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL GROUP BY KISHU_CODE''')
    kishu_3in = {row[0]: row[2]/row[1] if row[1]>0 else 0 for row in cursor.fetchall()}
    # --- 調教師3着率 ---
    cursor.execute('''SELECT CHOKYOSHI_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ("1","2","3") THEN 1 ELSE 0 END) FROM umagoto_race_joho WHERE CHOKYOSHI_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL GROUP BY CHOKYOSHI_CODE''')
    chokyoshi_3in = {row[0]: row[2]/row[1] if row[1]>0 else 0 for row in cursor.fetchall()}
    # --- race_shosaiから距離・芝/ダート・頭数 ---
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai = {row[0]: (float(row[1]) if row[1] else 0, int(row[2]) if row[2] else 0, int(row[3]) if row[3] else 0) for row in cursor.fetchall()}
    # --- メインデータ ---
    query = '''SELECT BAREI, SEIBETSU_CODE, BATAIJU, FUTAN_JURYO, TANSHO_NINKIJUN, KISHU_CODE, CHOKYOSHI_CODE, KETTO_TOROKU_BANGO, RACE_CODE, KAISAI_NEN, KAISAI_GAPPI, KAKUTEI_CHAKUJUN FROM umagoto_race_joho WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KAISAI_NEN IS NOT NULL AND KAISAI_GAPPI IS NOT NULL'''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN']
    df = pd.DataFrame(rows, columns=columns)
    # --- 馬体重変動 ---
    df['BATAIJU'] = pd.to_numeric(df['BATAIJU'], errors='coerce')
    df['BATAIJU_DIFF'] = df.groupby('KETTO_TOROKU_BANGO')['BATAIJU'].diff().fillna(0)
    # --- 馬の過去5走3着以内率 ---
    df['KAKUTEI_CHAKUJUN_norm'] = df['KAKUTEI_CHAKUJUN'].apply(lambda x: str(x).strip().replace('　','').replace(' ','').translate(str.maketrans('０１２３４５６７８９', '0123456789')).lstrip('0'))
    df['target'] = df['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
    uma_5r_3in = {}
    for uma, group in df.groupby('KETTO_TOROKU_BANGO'):
        arr = group['target'].rolling(5, min_periods=1).mean()
        uma_5r_3in.update(dict(zip(group.index, arr)))
    df['UMA_5R_3IN_RATE'] = df.index.map(uma_5r_3in)
    # --- 騎手3着率・調教師3着率 ---
    df['KISHU_3IN_RATE'] = df['KISHU_CODE'].map(kishu_3in)
    df['CHOKYOSHI_3IN_RATE'] = df['CHOKYOSHI_CODE'].map(chokyoshi_3in)
    # --- 距離・芝/ダート・頭数 ---
    df['KYORI'] = df['RACE_CODE'].map(lambda x: race_shosai.get(x, (0,0,0))[0])
    df['TRACK_CODE'] = df['RACE_CODE'].map(lambda x: race_shosai.get(x, (0,0,0))[1])
    df['SHUSSO_TOSU'] = df['RACE_CODE'].map(lambda x: race_shosai.get(x, (0,0,0))[2])
    # --- レース日付・間隔 ---
    df['RACE_DATE'] = df['KAISAI_NEN'].astype(str) + df['KAISAI_GAPPI'].astype(str).str.zfill(4)
    df['RACE_DATE'] = pd.to_datetime(df['RACE_DATE'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['KETTO_TOROKU_BANGO', 'RACE_DATE'])
    df['RACE_KANKAKU'] = df.groupby('KETTO_TOROKU_BANGO')['RACE_DATE'].diff().dt.days.fillna(0)
    # --- ターゲットエンコーディング ---
    for col in ['KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE']:
        te = df.groupby(col)['target'].transform('mean')
        df[f'{col}_TE'] = te
    # --- カテゴリ変数をLabelEncoder ---
    for col in ['SEIBETSU_CODE','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    # --- 数値系カラムをfloat型に変換 ---
    for col in ['BAREI','BATAIJU','BATAIJU_DIFF','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_3IN_RATE','CHOKYOSHI_3IN_RATE','UMA_5R_3IN_RATE','KYORI','TRACK_CODE','SHUSSO_TOSU','RACE_KANKAKU','KISHU_CODE_TE','CHOKYOSHI_CODE_TE','KETTO_TOROKU_BANGO_TE','RACE_CODE_TE']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 欠損値補完
    df = df.fillna(0)
    # target=1/0の件数をprint
    n1 = (df['target'] == 1).sum()
    n0 = (df['target'] == 0).sum()
    print(f'target=1(3着以内): {n1}件, target=0: {n0}件')
    if n1 < 100:
        print('警告: 3着以内データが極端に少ないです。抽出期間や条件を見直してください。')
    # 特徴量リスト
    feature_cols = [
        'BAREI','SEIBETSU_CODE','BATAIJU','BATAIJU_DIFF','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE',
        'KYORI','TRACK_CODE','SHUSSO_TOSU','KISHU_3IN_RATE','CHOKYOSHI_3IN_RATE','UMA_5R_3IN_RATE','RACE_KANKAKU',
        'KISHU_CODE_TE','CHOKYOSHI_CODE_TE','KETTO_TOROKU_BANGO_TE','RACE_CODE_TE'
    ]
    X = df[feature_cols]
    y = df['target']
    end_time = time.time()
    print(f'【fetch_data関数 実行所要時間】{end_time - start_time:.2f}秒')
    return X, y, df, n0, n1

def main():
    print('=== LightGBM/XGBoost競馬3着以内予測 実行開始 ===')
    X, y, df, n0, n1 = fetch_data()
    if len(X) == 0:
        print('データがありません')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # --- SMOTEでオーバーサンプリング ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f'SMOTE後: {np.bincount(y_train_res)}')
    run_dir = f'sanshutsu_kun/1_predict_models/a1_gbdt_lightbgm/results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(run_dir, exist_ok=True)
    # --- LightGBM学習（パラメータ最適化） ---
    scale_pos_weight = n0 / n1 if n1 > 0 else 1.0
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 7,
        'learning_rate': 0.03,
        'feature_fraction': 0.9,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'scale_pos_weight': scale_pos_weight
    }
    lgb_model = lgb.train(
        lgb_params, lgb.Dataset(X_train_res, y_train_res), valid_sets=[lgb.Dataset(X_test, y_test)], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    lgb_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    # --- XGBoost学習（同じ特徴量） ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.03, scale_pos_weight=scale_pos_weight,
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    # --- アンサンブル（平均） ---
    probs = (lgb_probs + xgb_probs) / 2
    # --- PRカーブからF1最大化閾値を探索 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'最適閾値（F1最大）: {best_threshold:.4f}, F1={f1s[best_idx]:.4f}')
    preds = (probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds)
    print('--- 評価結果（最適閾値適用/アンサンブル） ---')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(report)
    # グラフ
    graph_paths = plot_and_save_graphs(y_test, preds, probs, [], [], run_dir)
    # 画像base64化
    images_b64 = []
    for path in graph_paths:
        with open(path, 'rb') as imgf:
            b64img = base64.b64encode(imgf.read()).decode('utf-8')
            images_b64.append((os.path.basename(path), b64img))
    # GPT評価
    prompt = f"""
競馬の3着以内に入る馬の特徴をLightGBM+XGBoostアンサンブルで学習したモデルの評価結果です。\n\n最適閾値: {best_threshold:.4f}\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    # 結果保存
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write('--- 評価結果 ---\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(report + '\n')
        f.write('\n--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    # スクリプト自身をrun_dirにコピー
    import sys
    script_path = os.path.abspath(sys.argv[0])
    shutil.copyfile(script_path, os.path.join(run_dir, os.path.basename(script_path)))

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')