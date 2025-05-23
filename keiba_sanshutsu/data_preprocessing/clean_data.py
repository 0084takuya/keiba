import pandas as pd

def clean_data(df):
    # 欠損値を0で埋める例
    df = df.fillna(0)
    # 異常値処理例（必要に応じてカスタマイズ）
    # df = df[df['feature'] < threshold]
    print("前処理完了")
    return df

if __name__ == "__main__":
    # テスト用ダミーデータ
    df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
    clean_data(df) 