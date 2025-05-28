import os
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
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    today = datetime.today()
    three_years_ago = today - timedelta(days=365*3)
    date_min = three_years_ago.strftime('%Y%m%d')
    # 主要特徴量を抽出
    query = '''
        SELECT BAREI, SEIBETSU_CODE, BATAIJU, FUTAN_JURYO, TANSHO_NINKIJUN, KISHU_CODE, CHOKYOSHI_CODE, KETTO_TOROKU_BANGO, RACE_CODE, KAISAI_NEN, KAISAI_GAPPI,
               KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho
        WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s
        AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KAISAI_NEN IS NOT NULL AND KAISAI_GAPPI IS NOT NULL
    '''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()
    columns = ['BAREI','SEIBETSU_CODE','BATAIJU','FUTAN_JURYO','TANSHO_NINKIJUN','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE','KAISAI_NEN','KAISAI_GAPPI','KAKUTEI_CHAKUJUN']
    df = pd.DataFrame(rows, columns=columns)
    # 数値系カラムをfloat型に変換
    for col in ['BAREI','BATAIJU','FUTAN_JURYO','TANSHO_NINKIJUN']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # KAKUTEI_CHAKUJUNの値をprint
    print('KAKUTEI_CHAKUJUNの値サンプル:', df['KAKUTEI_CHAKUJUN'].value_counts().head(20))
    # 全角数字・空白を半角数字に正規化
    def normalize_chakujun(val):
        if val is None:
            return ''
        s = str(val).strip().replace('　','').replace(' ', '')
        # 全角数字→半角
        s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        return s
    df['KAKUTEI_CHAKUJUN_norm'] = df['KAKUTEI_CHAKUJUN'].apply(normalize_chakujun)
    print('正規化後KAKUTEI_CHAKUJUNの値サンプル:', df['KAKUTEI_CHAKUJUN_norm'].value_counts().head(20))
    # ラベル作成
    df['target'] = df['KAKUTEI_CHAKUJUN_norm'].isin(['1','2','3']).astype(int)
    # カテゴリ変数をLabelEncoder
    for col in ['SEIBETSU_CODE','KISHU_CODE','CHOKYOSHI_CODE','KETTO_TOROKU_BANGO','RACE_CODE']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    # 欠損値補完
    df = df.fillna(0)
    # target=1/0の件数をprint
    n1 = (df['target'] == 1).sum()
    n0 = (df['target'] == 0).sum()
    print(f'target=1(3着以内): {n1}件, target=0: {n0}件')
    if n1 < 100:
        print('警告: 3着以内データが極端に少ないです。抽出期間や条件を見直してください。')
    X = df.drop(['KAKUTEI_CHAKUJUN','KAKUTEI_CHAKUJUN_norm','target','KAISAI_NEN','KAISAI_GAPPI'], axis=1)
    y = df['target']
    return X, y, df

def plot_and_save_graphs(y_test, preds, probs, run_dir):
    # 日本語フォント設定
    try:
        font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
        if not os.path.exists(font_path):
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
        if not os.path.exists(font_path):
            font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    except Exception as e:
        pass
    paths = []
    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc_score:.3f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'roc_curve.png'))
    paths.append(os.path.join(run_dir, 'roc_curve.png'))
    plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.savefig(os.path.join(run_dir, 'pr_curve.png'))
    paths.append(os.path.join(run_dir, 'pr_curve.png'))
    plt.close()
    # 混同行列
    cm = confusion_matrix(y_test, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
    paths.append(os.path.join(run_dir, 'confusion_matrix.png'))
    plt.close()
    return paths

def ask_gpt41(prompt, images_b64):
    # GPT-4.1 API呼び出し（画像はbase64で渡す）
    # ここではテキストのみで簡易実装
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT-4.1 API Error] {e}"

def main():
    print('=== LightGBM競馬3着以内予測 実行開始 ===')
    X, y, df = fetch_data()
    if len(X) == 0:
        print('データがありません')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    run_dir = f'sanshutsu_kun/1_predict_models/a1_gbdt_lightbgm/results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(run_dir, exist_ok=True)
    # LightGBM学習
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model = lgb.train(
        params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )
    # 推論
    probs = model.predict(X_test, num_iteration=model.best_iteration)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds)
    print('--- 評価結果 ---')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(report)
    # グラフ
    graph_paths = plot_and_save_graphs(y_test, preds, probs, run_dir)
    # 画像base64化
    images_b64 = []
    for path in graph_paths:
        with open(path, 'rb') as imgf:
            b64img = base64.b64encode(imgf.read()).decode('utf-8')
            images_b64.append((os.path.basename(path), b64img))
    # GPT評価
    prompt = f"""
競馬の3着以内に入る馬の特徴をLightGBMで学習したモデルの評価結果です。\n\n精度: {acc:.4f}\nF1スコア: {f1:.4f}\n詳細:\n{report}\n\n以下のグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。\n"""
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
    print(f'【実行所要時間】{datetime.now()}')

if __name__ == '__main__':
    main() 