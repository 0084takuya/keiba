import os
import pymysql
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import base64
from modules.plot_utils import plot_and_save_graphs
from modules.gpt_utils import ask_gpt41
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List

# --- .envからAPIキーをロード ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# --- DB接続設定（必要に応じて修正） ---
DB_HOST = os.getenv('KEIBA_DB_HOST', 'localhost')
DB_USER = os.getenv('KEIBA_DB_USER', 'root')
DB_PASS = os.getenv('KEIBA_DB_PASS', '')
DB_NAME = os.getenv('KEIBA_DB_NAME', 'mykeibadb')
DB_PORT = int(os.getenv('KEIBA_DB_PORT', 3306))

@dataclass
class Feature:
    name: str
    db_column: Optional[str] = None
    description: str = ""
    compute_func: Optional[Callable] = None
    is_categorical: bool = False
    depends_on: Optional[str] = None  # 参照する辞書やSQL名

# 特徴量定義
FEATURES: List[Feature] = [
    Feature("BAREI", "BAREI", "馬齢"),
    Feature("SEIBETSU_CODE", "SEIBETSU_CODE", "性別コード", is_categorical=True),
    Feature("BATAIJU", "BATAIJU", "馬体重"),
    Feature("FUTAN_JURYO", "FUTAN_JURYO", "負担重量"),
    Feature("TANSHO_NINKIJUN", "TANSHO_NINKIJUN", "単勝人気順"),
    Feature("KYORI", None, "距離", depends_on="race_shosai"),
    Feature("TRACK_CODE", None, "芝/ダート（トラックコード）", is_categorical=True, depends_on="race_shosai"),
    Feature("SHUSSO_TOSU", None, "頭数", depends_on="race_shosai"),
    # Feature("FUKUSHO_NINKIJUN", None, "複勝人気順", depends_on="umagoto_race_joho"),  # 一時的に除外
    Feature("KISHU_3IN_RATE", None, "騎手3着以内率", depends_on="kishu_stats"),
    Feature("TENKO_CODE", None, "馬場状態", is_categorical=True, depends_on="race_tenko"),
    Feature("CHOKYOSHI_3IN_RATE", None, "調教師3着以内率", depends_on="chokyoshi_stats"),
    Feature("FATHER_LINEAGE", None, "父系統ID", is_categorical=True, depends_on="uma_father"),
    Feature("UMA_5R_3IN_RATE", None, "馬過去5走3着以内率", depends_on="uma_race"),
    Feature("COURSE_3IN_RATE", None, "コース適性3着以内率", depends_on="uma_course"),
    Feature("COURSE_UMA_5R_3IN_RATE", None, "コース×馬過去5走3着以内率", depends_on="uma_course_5r"),
]

# --- 特徴量とラベルの抽出 ---
def fetch_data():
    print('=== fetch_data関数 実行開始 ===')
    start_time = time.time()
    conn = pymysql.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASS,
        db=DB_NAME, port=DB_PORT, charset='utf8mb4')
    cursor = conn.cursor()
    # 直近3年分の日付を計算
    today = datetime.today()
    three_years_ago = today - timedelta(days=365*3)
    date_min = three_years_ago.strftime('%Y%m%d')

    # 1. 騎手ごとの3着以内率を計算
    cursor.execute('''
        SELECT KISHU_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ('1','2','3') THEN 1 ELSE 0 END)
        FROM umagoto_race_joho
        WHERE KISHU_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        GROUP BY KISHU_CODE
    ''')
    kishu_stats = {row[0]: (row[2]/row[1] if row[1]>0 else 0) for row in cursor.fetchall()}

    # 2. 調教師ごとの3着以内率
    cursor.execute('''
        SELECT CHOKYOSHI_CODE, COUNT(*), SUM(CASE WHEN KAKUTEI_CHAKUJUN IN ('1','2','3') THEN 1 ELSE 0 END)
        FROM umagoto_race_joho
        WHERE CHOKYOSHI_CODE IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        GROUP BY CHOKYOSHI_CODE
    ''')
    chokyoshi_stats = {row[0]: (row[2]/row[1] if row[1]>0 else 0) for row in cursor.fetchall()}

    # 3. 馬場状態（TENKO_CODE）をRACE_CODEで取得
    cursor.execute('''
        SELECT RACE_CODE, TENKO_CODE FROM race_shosai WHERE TENKO_CODE IS NOT NULL
    ''')
    race_tenko = {row[0]: row[1] for row in cursor.fetchall()}
    tenko_labels = {code: i for i, code in enumerate(sorted(set(race_tenko.values())))}

    # 4. 父系統IDをUMAごとに取得
    cursor.execute('''
        SELECT KETTO_TOROKU_BANGO, KETTO1_HANSHOKU_TOROKU_BANGO FROM kyosoba_master2
    ''')
    uma_father = {row[0]: row[1] for row in cursor.fetchall()}
    father_ids = list(set(uma_father.values()))
    father_label = {code: i for i, code in enumerate(father_ids)}

    # 5. メインデータ取得
    db_columns = [f.db_column for f in FEATURES if f.db_column]
    query = f'''
        SELECT {', '.join(db_columns)}, KISHU_CODE, RACE_CODE, KETTO_TOROKU_BANGO, CHOKYOSHI_CODE, KEIBAJO_CODE
        FROM umagoto_race_joho
        WHERE CONCAT(KAISAI_NEN, LPAD(KAISAI_GAPPI, 4, '0')) >= %s
        AND {' AND '.join([f'{col} IS NOT NULL' for col in db_columns])}
        AND KISHU_CODE IS NOT NULL AND RACE_CODE IS NOT NULL AND KETTO_TOROKU_BANGO IS NOT NULL AND CHOKYOSHI_CODE IS NOT NULL AND KEIBAJO_CODE IS NOT NULL
    '''
    cursor.execute(query, (date_min,))
    rows = cursor.fetchall()

    # 6. 馬の過去5走3着以内率を計算
    cursor.execute('''
        SELECT KETTO_TOROKU_BANGO, RACE_CODE, KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho
        WHERE KETTO_TOROKU_BANGO IS NOT NULL AND KAKUTEI_CHAKUJUN IS NOT NULL
        ORDER BY KETTO_TOROKU_BANGO, RACE_CODE
    ''')
    uma_race = defaultdict(list)
    for row in cursor.fetchall():
        uma_race[row[0]].append((row[1], row[2]))

    # 7. コース適性（KEIBAJO_CODE, 距離帯ごとの3着以内率）
    cursor.execute('''
        SELECT u.KETTO_TOROKU_BANGO, u.KEIBAJO_CODE, s.KYORI, u.KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho u
        JOIN race_shosai s ON u.RACE_CODE = s.RACE_CODE
        WHERE u.KETTO_TOROKU_BANGO IS NOT NULL AND u.KEIBAJO_CODE IS NOT NULL AND s.KYORI IS NOT NULL AND u.KAKUTEI_CHAKUJUN IS NOT NULL
    ''')
    uma_course = defaultdict(list)
    for row in cursor.fetchall():
        try:
            kyori = int(row[2])
            if kyori < 1400:
                kyori_bin = 0
            elif kyori < 2000:
                kyori_bin = 1
            else:
                kyori_bin = 2
        except:
            kyori_bin = 1
        uma_course[(row[0], row[1], kyori_bin)].append(int(row[3]))

    # 8. コース×馬過去5走3着以内率を計算
    cursor.execute('''
        SELECT u.KETTO_TOROKU_BANGO, u.KEIBAJO_CODE, s.KYORI, u.KAKUTEI_CHAKUJUN
        FROM umagoto_race_joho u
        JOIN race_shosai s ON u.RACE_CODE = s.RACE_CODE
        WHERE u.KETTO_TOROKU_BANGO IS NOT NULL AND u.KEIBAJO_CODE IS NOT NULL AND s.KYORI IS NOT NULL AND u.KAKUTEI_CHAKUJUN IS NOT NULL
    ''')
    uma_course_5r = defaultdict(list)
    for row in cursor.fetchall():
        try:
            kyori = int(row[2])
            if kyori < 1400:
                kyori_bin = 0
            elif kyori < 2000:
                kyori_bin = 1
            else:
                kyori_bin = 2
        except:
            kyori_bin = 1
        uma_course_5r[(row[0], row[1], kyori_bin)].append(int(row[3]))

    # race_shosaiからKYORI, TRACK_CODE, SHUSSO_TOSUを全件取得
    cursor.execute('SELECT RACE_CODE, KYORI, TRACK_CODE, SHUSSO_TOSU FROM race_shosai')
    race_shosai_dict = {}
    for row in cursor.fetchall():
        race_shosai_dict[row[0]] = {
            'KYORI': float(row[1]) if row[1] and str(row[1]).isdigit() else 0,
            'TRACK_CODE': int(row[2]) if row[2] and str(row[2]).isdigit() else 0,
            'SHUSSO_TOSU': int(row[3]) if row[3] and str(row[3]).isdigit() else 0,
        }
    # umagoto_race_johoからFUKUSHO_NINKIJUNを全件取得
    # cursor.execute('SELECT RACE_CODE, KETTO_TOROKU_BANGO, FUKUSHO_NINKIJUN FROM umagoto_race_joho')
    # fukusho_ninki_dict = {}
    # for row in cursor.fetchall():
    #     key = (row[0], row[1])
    #     fukusho_ninki_dict[key] = int(row[2]) if row[2] and str(row[2]).isdigit() else 0

    conn.close()
    X, y = [], []
    for row in rows:
        try:
            feature_dict = {}
            idx = 0
            # DBカラムから直接取得
            for f in FEATURES:
                if f.db_column:
                    val = row[idx]
                    idx += 1
                    feature_dict[f.name] = float(val) if val is not None and str(val).replace('.', '', 1).isdigit() else 0
            # 追加特徴量
            kishu_code = row[idx]
            race_code = row[idx+1]
            ketto_toroku_bango = row[idx+2]
            chokyoshi_code = row[idx+3]
            keibajo_code = row[idx+4]
            # race_shosaiからKYORI, TRACK_CODE, SHUSSO_TOSU
            race_shosai_row = race_shosai_dict.get(race_code, None)
            if race_shosai_row:
                feature_dict['KYORI'] = race_shosai_row['KYORI']
                feature_dict['TRACK_CODE'] = race_shosai_row['TRACK_CODE']
                feature_dict['SHUSSO_TOSU'] = race_shosai_row['SHUSSO_TOSU']
            else:
                feature_dict['KYORI'] = 0
                feature_dict['TRACK_CODE'] = 0
                feature_dict['SHUSSO_TOSU'] = 0
            # FUKUSHO_NINKIJUN
            # feature_dict['FUKUSHO_NINKIJUN'] = fukusho_ninki_dict.get((race_code, ketto_toroku_bango), 0)
            feature_dict['KISHU_3IN_RATE'] = kishu_stats.get(kishu_code, 0)
            tenko_code = race_tenko.get(race_code, None)
            feature_dict['TENKO_CODE'] = tenko_labels[tenko_code] if tenko_code in tenko_labels else 0
            feature_dict['CHOKYOSHI_3IN_RATE'] = chokyoshi_stats.get(chokyoshi_code, 0)
            father_id = uma_father.get(ketto_toroku_bango, None)
            feature_dict['FATHER_LINEAGE'] = father_label[father_id] if father_id in father_label else 0
            past = uma_race.get(ketto_toroku_bango, [])
            past5 = [int(x[1]) for x in past[-6:-1]] if len(past) >= 6 else [int(x[1]) for x in past[:-1]]
            feature_dict['UMA_5R_3IN_RATE'] = sum([1 for c in past5 if c in [1,2,3]]) / len(past5) if past5 else 0
            try:
                kyori = float(feature_dict['KYORI'])
                if kyori < 1400:
                    kyori_bin = 0
                elif kyori < 2000:
                    kyori_bin = 1
                else:
                    kyori_bin = 2
            except:
                kyori_bin = 1
            course_key = (ketto_toroku_bango, keibajo_code, kyori_bin)
            course_results = uma_course.get(course_key, [])
            feature_dict['COURSE_3IN_RATE'] = sum([1 for c in course_results if c in [1,2,3]]) / len(course_results) if course_results else 0
            # コース×馬過去5走3着以内率
            course_uma5r = uma_course_5r.get(course_key, [])
            course_uma5r5 = course_uma5r[-5:] if len(course_uma5r) >= 5 else course_uma5r
            feature_dict['COURSE_UMA_5R_3IN_RATE'] = sum([1 for c in course_uma5r5 if c in [1,2,3]]) / len(course_uma5r5) if course_uma5r5 else 0
            # FEATURE順で格納
            X.append([feature_dict[f.name] for f in FEATURES])
            chaku = int(row[4]) if len(row) > 4 else 0  # TANSHO_NINKIJUNの次が着順想定
            y.append(1 if chaku in [1,2,3] else 0)
        except (ValueError, TypeError):
            continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # 標準化（連続値のみ）
    if len(X) > 0:
        scaler = StandardScaler()
        num_idx = [i for i, f in enumerate(FEATURES) if not f.is_categorical]
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
    fetch_time = time.time() - start_time
    print(f'【データ取得・前処理所要時間】{fetch_time:.2f}秒')
    return X, y

# --- PyTorch Dataset ---
class KeibaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- DQNネットワーク（簡易） ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 学習ループ（簡易） ---
def train():
    print('=== train関数 実行開始 ===')
    start_time = time.time()
    X, y = fetch_data()
    if len(X) == 0:
        print('データがありません')
        return
    # クラス不均衡対策：クラス重み計算
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # 実行時刻ディレクトリ作成
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f'sanshutsu_kun/1_predict_models/a_value_based_dqn/results/{now_str}'
    os.makedirs(run_dir, exist_ok=True)
    # データを8:2で分割
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dataset = KeibaDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = DQN(input_dim=X.shape[1], output_dim=2)  # 2クラス: 3着以内/それ以外
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_losses = []
    train_accuracies = []
    for epoch in range(5):  # エポック数は少なめ
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        train_losses.append(total_loss/len(loader))
        train_accuracies.append(correct/total)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, Accuracy: {correct/total:.4f}')
    torch.save(model.state_dict(), os.path.join(run_dir, 'dqn_keiba_simple.pth'))
    print('モデル保存済み')
    train_time = time.time() - start_time
    print(f'【学習所要時間】{train_time:.2f}秒')
    # 評価
    evaluate(model, X_test, y_test, train_losses, train_accuracies, run_dir)

def evaluate(model, X_test, y_test, train_losses, train_accuracies, run_dir):
    print('=== evaluate関数 実行開始 ===')
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds)
    print('--- 評価結果 ---')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(report)
    # PRカーブからF1最大化となる閾値を探索
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f'--- PRカーブからF1最大化となる最適閾値: {best_threshold:.3f} ---')
    # 最適閾値で再評価
    best_preds = (probs >= best_threshold).astype(int)
    best_acc = accuracy_score(y_test, best_preds)
    best_f1 = f1_score(y_test, best_preds)
    best_report = classification_report(y_test, best_preds)
    print('--- 最適閾値での再評価 ---')
    print(f'Accuracy: {best_acc:.4f}')
    print(f'F1 Score: {best_f1:.4f}')
    print(best_report)
    # グラフ出力
    graph_paths = plot_and_save_graphs(y_test, preds, probs, train_losses, train_accuracies, run_dir)
    # 画像base64化
    images_b64 = []
    for path in graph_paths:
        with open(path, 'rb') as imgf:
            b64img = base64.b64encode(imgf.read()).decode('utf-8')
            images_b64.append((os.path.basename(path), b64img))
    # プロンプト生成部分
    feature_desc = get_feature_description()
    prompt = f"""
競馬の3着以内に入る馬の特徴をDQNで学習したモデルの評価結果です。

【モデルで利用している特徴量（入力項目）】
{feature_desc}

精度: {acc:.4f}
F1スコア: {f1:.4f}
詳細:
{report}

PRカーブからF1最大化となる最適閾値: {best_threshold:.3f}
最適閾値での再評価:
精度: {best_acc:.4f}
F1スコア: {best_f1:.4f}
詳細:
{best_report}

以下の4つのグラフ画像（base64エンコード済み）も参考にして、モデル改善のためのアドバイスを日本語で簡潔に出力してください。
"""
    gpt_advice = ask_gpt41(prompt, images_b64)
    # 結果をファイル保存
    with open(os.path.join(run_dir, 'eval_and_gpt_advice.txt'), 'w', encoding='utf-8') as f:
        f.write('--- 評価結果 ---\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(report + '\n')
        f.write(f'--- PRカーブからF1最大化となる最適閾値: {best_threshold:.3f} ---\n')
        f.write('--- 最適閾値での再評価 ---\n')
        f.write(f'Accuracy: {best_acc:.4f}\n')
        f.write(f'F1 Score: {best_f1:.4f}\n')
        f.write(best_report + '\n')
        f.write('\n--- GPT-4.1からの改善アドバイス ---\n')
        f.write(gpt_advice + '\n')
    eval_time = time.time() - start_time
    print(f'【推論・評価所要時間】{eval_time:.2f}秒')

# --- 特徴量説明文の自動生成 ---
def get_feature_description():
    return '\n'.join([f'- {f.description}（{f.name}）' for f in FEATURES])

if __name__ == '__main__':
    train() 