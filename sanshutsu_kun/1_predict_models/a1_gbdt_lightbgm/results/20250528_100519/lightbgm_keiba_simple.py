import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
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
import shap
from catboost import CatBoostClassifier

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
    # --- tenko_baba_jotaiから天候・馬場状態 ---
    cursor.execute('SELECT RACE_CODE, TENKO_JOTAI_GENZAI, BABA_JOTAI_SHIBA_GENZAI, BABA_JOTAI_DIRT_GENZAI FROM tenko_baba_jotai')
    tenko_baba = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}
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
    # --- 追加特徴量: 直近成績・前走着順・前走人気・前走馬体重・前走馬体重変動 ---
    # 前走着順
    df['ZEN_SO_CHAKUJUN'] = df.groupby('KETTO_TOROKU_BANGO')['KAKUTEI_CHAKUJUN_norm'].shift(1).fillna('')
    df['ZEN_SO_CHAKUJUN'] = pd.to_numeric(df['ZEN_SO_CHAKUJUN'], errors='coerce').fillna(0)
    # 前走人気
    df['ZEN_SO_NINKI'] = df.groupby('KETTO_TOROKU_BANGO')['TANSHO_NINKIJUN'].shift(1).fillna(0)
    # 前走馬体重
    df['ZEN_SO_BATAIJU'] = df.groupby('KETTO_TOROKU_BANGO')['BATAIJU'].shift(1).fillna(0)
    # 前走馬体重変動
    df['ZEN_SO_BATAIJU_DIFF'] = df['BATAIJU'] - df['ZEN_SO_BATAIJU']
    # 出走回数
    df['SHUSSO_KAISU'] = df.groupby('KETTO_TOROKU_BANGO').cumcount()
    # 斤量変化
    df['FUTAN_JURYO_DIFF'] = df.groupby('KETTO_TOROKU_BANGO')['FUTAN_JURYO'].diff().fillna(0)
    # --- 数値型変換 ---
    for col in ['ZEN_SO_CHAKUJUN','ZEN_SO_NINKI','ZEN_SO_BATAIJU','ZEN_SO_BATAIJU_DIFF','SHUSSO_KAISU','FUTAN_JURYO_DIFF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # --- 天候・馬場状態を特徴量として追加 ---
    df['TENKO_JOTAI_GENZAI'] = df['RACE_CODE'].map(lambda x: tenko_baba.get(x, (None, None, None))[0])
    df['BABA_JOTAI_SHIBA_GENZAI'] = df['RACE_CODE'].map(lambda x: tenko_baba.get(x, (None, None, None))[1])
    df['BABA_JOTAI_DIRT_GENZAI'] = df['RACE_CODE'].map(lambda x: tenko_baba.get(x, (None, None, None))[2])
    # カテゴリ変数をLabelEncoder
    for col in ['TENKO_JOTAI_GENZAI','BABA_JOTAI_SHIBA_GENZAI','BABA_JOTAI_DIRT_GENZAI']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # --- 騎手・調教師ペア3着率 ---
    df['KISHU_CHOKYOSHI_PAIR'] = df['KISHU_CODE'].astype(str) + '_' + df['CHOKYOSHI_CODE'].astype(str)
    pair_3in = df.groupby('KISHU_CHOKYOSHI_PAIR')['target'].mean()
    df['KISHU_CHOKYOSHI_PAIR_3IN_RATE'] = df['KISHU_CHOKYOSHI_PAIR'].map(pair_3in)

    # --- 直近5走の着順平均・着順変化率 ---
    df['ZEN_5R_CHAKUJUN_MEAN'] = df.groupby('KETTO_TOROKU_BANGO')['ZEN_SO_CHAKUJUN'].rolling(5, min_periods=1).mean().reset_index(0,drop=True)
    df['ZEN_5R_CHAKUJUN_DIFF'] = df.groupby('KETTO_TOROKU_BANGO')['ZEN_SO_CHAKUJUN'].diff().fillna(0)

    # --- 数値型変換 ---
    for col in ['ZEN_SO_CHAKUJUN','ZEN_SO_NINKI','ZEN_SO_BATAIJU','ZEN_SO_BATAIJU_DIFF','SHUSSO_KAISU','FUTAN_JURYO_DIFF','TENKO_JOTAI_GENZAI','BABA_JOTAI_SHIBA_GENZAI','BABA_JOTAI_DIRT_GENZAI','KISHU_CHOKYOSHI_PAIR_3IN_RATE','ZEN_5R_CHAKUJUN_MEAN','ZEN_5R_CHAKUJUN_DIFF']:
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
    # --- 追加特徴量: 直近成績・前走着順・前走人気・前走馬体重・前走馬体重変動 ---
    # 前走着順
    df['ZEN_SO_CHAKUJUN'] = df.groupby('KETTO_TOROKU_BANGO')['KAKUTEI_CHAKUJUN_norm'].shift(1).fillna('')
    df['ZEN_SO_CHAKUJUN'] = pd.to_numeric(df['ZEN_SO_CHAKUJUN'], errors='coerce').fillna(0)
    # 前走人気
    df['ZEN_SO_NINKI'] = df.groupby('KETTO_TOROKU_BANGO')['TANSHO_NINKIJUN'].shift(1).fillna(0)
    # 前走馬体重
    df['ZEN_SO_BATAIJU'] = df.groupby('KETTO_TOROKU_BANGO')['BATAIJU'].shift(1).fillna(0)
    # 前走馬体重変動
    df['ZEN_SO_BATAIJU_DIFF'] = df['BATAIJU'] - df['ZEN_SO_BATAIJU']
    # 出走回数
    df['SHUSSO_KAISU'] = df.groupby('KETTO_TOROKU_BANGO').cumcount()
    # 斤量変化
    df['FUTAN_JURYO_DIFF'] = df.groupby('KETTO_TOROKU_BANGO')['FUTAN_JURYO'].diff().fillna(0)
    # --- 数値型変換 ---
    for col in ['ZEN_SO_CHAKUJUN','ZEN_SO_NINKI','ZEN_SO_BATAIJU','ZEN_SO_BATAIJU_DIFF','SHUSSO_KAISU','FUTAN_JURYO_DIFF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # --- 特徴量リスト拡張 ---
    feature_cols = [
        'BAREI','SEIBETSU_CODE','BATAIJU','BATAIJU_DIFF','FUTAN_JURYO','FUTAN_JURYO_DIFF','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE',
        'KYORI','TRACK_CODE','SHUSSO_TOSU','KISHU_3IN_RATE','CHOKYOSHI_3IN_RATE','UMA_5R_3IN_RATE','RACE_KANKAKU',
        'KISHU_CODE_TE','CHOKYOSHI_CODE_TE','KETTO_TOROKU_BANGO_TE','RACE_CODE_TE',
        'ZEN_SO_CHAKUJUN','ZEN_SO_NINKI','ZEN_SO_BATAIJU','ZEN_SO_BATAIJU_DIFF','SHUSSO_KAISU',
        'TENKO_JOTAI_GENZAI','BABA_JOTAI_SHIBA_GENZAI','BABA_JOTAI_DIRT_GENZAI',
        'KISHU_CHOKYOSHI_PAIR_3IN_RATE','ZEN_5R_CHAKUJUN_MEAN','ZEN_5R_CHAKUJUN_DIFF'
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
    # --- CatBoost学習（同じ特徴量） ---
    cat_model = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.03, scale_pos_weight=scale_pos_weight, l2_leaf_reg=1.0, verbose=0, random_seed=42)
    cat_model.fit(X_train_res, y_train_res)
    cat_probs = cat_model.predict_proba(X_test)[:,1]
    # --- アンサンブル（平均） ---
    probs = (lgb_probs + xgb_probs + cat_probs) / 3
    # --- PRカーブからprecision重視（0.8以上）かつF1最大の閾値を探索 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    mask = precisions >= 0.8
    if mask.sum() > 0:
        best_idx = (f1s * mask).argmax()
    else:
        best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'最適閾値（precision>=0.8,F1最大）: {best_threshold:.4f}, F1={f1s[best_idx]:.4f}, precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f}')
    # --- 複数閾値で自動評価 ---
    thresholds_to_eval = [0.5, best_threshold, 0.7, 0.8]
    for th in thresholds_to_eval:
        preds = (probs >= th).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        report = classification_report(y_test, preds)
        print(f'--- 評価結果（閾値={th:.2f}） ---')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(report)
    # --- SHAP値で特徴量重要度グラフ ---
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False, plot_type='bar')
    shap_path = os.path.join(run_dir, 'shap_feature_importance.png')
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
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
競馬の3着以内に入る馬の特徴をLightGBM+XGBoost+CatBoostアンサンブルで学習したモデルの評価結果です。\n\n最適閾値: {best_threshold:.4f}\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
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
    # モデル保存
    lgb_model.save_model(os.path.join(run_dir, 'lgb_model.txt'))
    xgb_model.save_model(os.path.join(run_dir, 'xgb_model.json'))
    cat_model.save_model(os.path.join(run_dir, 'cat_model.cbm'))
    # --- SHAP値で寄与の低い特徴量を自動除外し再学習 ---
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    threshold = np.percentile(shap_abs_mean, 20)  # 下位20%を除外
    selected_features = [col for col, val in zip(X_test.columns, shap_abs_mean) if val > threshold]
    print(f'SHAP値で選択された特徴量数: {len(selected_features)} / {len(X_test.columns)}')
    # 再学習
    X_train_sel = X_train_res[selected_features]
    X_test_sel = X_test[selected_features]
    lgb_model_sel = lgb.train(
        lgb_params, lgb.Dataset(X_train_sel, y_train_res), valid_sets=[lgb.Dataset(X_test_sel, y_test)], num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]
    )
    lgb_probs_sel = lgb_model_sel.predict(X_test_sel, num_iteration=lgb_model_sel.best_iteration)
    # アンサンブルも再計算
    xgb_model_sel = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.03, scale_pos_weight=scale_pos_weight,
        reg_alpha=1.0, reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    xgb_model_sel.fit(X_train_sel, y_train_res)
    xgb_probs_sel = xgb_model_sel.predict_proba(X_test_sel)[:,1]
    cat_model_sel = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.03, scale_pos_weight=scale_pos_weight, l2_leaf_reg=1.0, verbose=0, random_seed=42)
    cat_model_sel.fit(X_train_sel, y_train_res)
    cat_probs_sel = cat_model_sel.predict_proba(X_test_sel)[:,1]
    probs_sel = (lgb_probs_sel + xgb_probs_sel + cat_probs_sel) / 3
    # 再評価
    precisions_sel, recalls_sel, thresholds_sel = precision_recall_curve(y_test, probs_sel)
    f1s_sel = 2 * precisions_sel * recalls_sel / (precisions_sel + recalls_sel + 1e-8)
    mask_sel = precisions_sel >= 0.8
    if mask_sel.sum() > 0:
        best_idx_sel = (f1s_sel * mask_sel).argmax()
    else:
        best_idx_sel = f1s_sel.argmax()
    best_threshold_sel = thresholds_sel[best_idx_sel] if best_idx_sel < len(thresholds_sel) else 0.5
    preds_sel = (probs_sel >= best_threshold_sel).astype(int)
    acc_sel = accuracy_score(y_test, preds_sel)
    f1_sel = f1_score(y_test, preds_sel)
    report_sel = classification_report(y_test, preds_sel)
    print(f'--- SHAP特徴量選択後の評価（閾値={best_threshold_sel:.2f}）---')
    print(f'Accuracy: {acc_sel:.4f}')
    print(f'F1 Score: {f1_sel:.4f}')
    print(report_sel)
    # 保存
    with open(os.path.join(run_dir, "eval_and_gpt_advice_shap_selected.txt"), 'w', encoding='utf-8') as f:
        f.write('--- SHAP特徴量選択後の評価 ---\n')
        f.write(f'Accuracy: {acc_sel:.4f}\n')
        f.write(f'F1 Score: {f1_sel:.4f}\n')
        f.write(report_sel + '\n')

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    print(f'【実行所要時間】{end_time - start_time:.2f}秒')