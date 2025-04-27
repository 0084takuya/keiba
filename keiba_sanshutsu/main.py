#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import argparse
import mysql.connector
import openai
import os
import textwrap

def load_data_from_mysql(host, user, password, database, table):
    """
    MySQLからデータを直接取得します。
    接続に必要なパラメータはコマンドライン引数などから渡してください。
    """
    # MySQLへ接続
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor(dictionary=True)
    query = f"SELECT * FROM {table}"
    cursor.execute(query)
    rows = cursor.fetchall()
    # 取得した結果をDataFrameに変換
    df = pd.DataFrame(rows)
    cursor.close()
    conn.close()
    return df

def preprocess_data_schema(df):
    """
    record_masterテーブルのスキーマに基づいた前処理を実施します。
    対象の特徴量として ['KAISAI_NEN', 'KEIBAJO_CODE', 'KAISAI_KAIJI', 'RACE_BANGO', 'KYORI', 'TRACK_CODE'] を使用し、
    ターゲット変数として 'RECORD_SHIKIBETSU_KUBUN' を利用します。
    文字列の数値カラムは数値型に変換し、カテゴリ変数はダミー変数化します。
    """
    if 'RECORD_SHIKIBETSU_KUBUN' not in df.columns:
        raise ValueError("RECORD_SHIKIBETSU_KUBUN カラムが存在しません。")
    target = df['RECORD_SHIKIBETSU_KUBUN']
    feature_columns = ['KAISAI_NEN', 'KEIBAJO_CODE', 'KAISAI_KAIJI', 'RACE_BANGO', 'KYORI', 'TRACK_CODE']
    for col in ['KAISAI_NEN', 'KAISAI_KAIJI', 'RACE_BANGO', 'KYORI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = pd.get_dummies(df, columns=['KEIBAJO_CODE', 'TRACK_CODE'], drop_first=True)
    features = [col for col in df.columns if col in feature_columns or col.startswith("KEIBAJO_CODE_") or col.startswith("TRACK_CODE_")]
    return df, features, target

def train_random_forest_schema(df, features, target):
    """
    record_masterテーブルのスキーマに基づいたデータでランダムフォレストを学習・評価します。
    """
    X = df[features]
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    feature_importances = pd.Series(clf.feature_importances_, index=features)
    print("Feature Importances:")
    print(feature_importances.sort_values(ascending=False))
    return clf

def evaluate_output_gpt(output_text: str) -> str:
    """GPTのAPIを利用してランダムフォレストの出力結果を評価します。

    引数:
      output_text: 評価対象の出力結果文字列

    環境変数 OPENAI_API_KEY にOpenAIのAPIキーを設定してください。
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    prompt = f"次のランダムフォレストの出力結果について評価をしてください:\n{output_text}\n評価:" 
    response = openai.ChatCompletion.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "あなたは機械学習モデルの評価の専門家です。"},
            {"role": "user", "content": prompt}
        ],
        # max_completion_tokens=500
    )
    evaluation = response.choices[0].message.content.strip()
    return evaluation

def main():
    parser = argparse.ArgumentParser(description="ランダムフォレストでrecord_masterのスキーマを学習する")
    # MySQL接続情報とテーブル名を引数で指定
    parser.add_argument("--host", type=str, default="localhost", help="MySQLのホスト")
    parser.add_argument("--user", type=str, default="root", help="MySQLのユーザ名")
    parser.add_argument("--password", type=str, default="", help="MySQLのパスワード")
    parser.add_argument("--database", type=str, default="mykeibadb", help="使用するデータベース名")
    parser.add_argument("--table", type=str, default="record_master", help="データを取得するテーブル名")
    parser.add_argument("--gpt_eval", action="store_true", default=True, help="GPTのAPIでランダムフォレストの出力結果を評価する場合、このフラグを設定します。")
    args = parser.parse_args()

    print("record_masterテーブルのランダムフォレスト学習を開始します。")
    print(args)
    
    df = load_data_from_mysql(args.host, args.user, args.password, args.database, args.table)
    # スキーマに基づいたデータ前処理
    df, features, target = preprocess_data_schema(df)
    # モデルの学習と評価
    model = train_random_forest_schema(df, features, target)

    if args.gpt_eval:
        sample_output = textwrap.dedent("""\
        Accuracy: 0.8896882494004796
        Classification Report:
                      precision    recall  f1-score   support

                   1       0.79      0.71      0.75        86
                   2       0.92      0.95      0.93       327
                   3       0.00      0.00      0.00         4

           accuracy                           0.89       417
          macro avg       0.57      0.55      0.56       417
      weighted avg       0.88      0.89      0.89       417

        Feature Importances:
        KAISAI_NEN         0.296352
        KAISAI_KAIJI       0.186137
        RACE_BANGO         0.148804
        KYORI              0.140375
        KEIBAJO_CODE_03    0.019347
        TRACK_CODE_17      0.018865
        KEIBAJO_CODE_07    0.016103
        KEIBAJO_CODE_09    0.015957
        KEIBAJO_CODE_04    0.014685
        TRACK_CODE_24      0.014510
        KEIBAJO_CODE_02    0.014039
        KEIBAJO_CODE_10    0.013932
        KEIBAJO_CODE_05    0.013062
        TRACK_CODE_18      0.011829
        TRACK_CODE_11      0.010893
        KEIBAJO_CODE_08    0.010532
        TRACK_CODE_23      0.010197
        TRACK_CODE_54      0.008526
        KEIBAJO_CODE_06    0.007595
        TRACK_CODE_52      0.007082
        TRACK_CODE_12      0.006346
        TRACK_CODE_55      0.004855
        TRACK_CODE_53      0.002461
        TRACK_CODE_59      0.002212
        TRACK_CODE_51      0.001510
        TRACK_CODE_20      0.001490
        TRACK_CODE_56      0.001331
        TRACK_CODE_57      0.000838
        TRACK_CODE_21      0.000133
        dtype: float64
        """)
        evaluation = evaluate_output_gpt(sample_output)
        print("\nGPTによる評価:", evaluation)

if __name__ == "__main__":
    main()
