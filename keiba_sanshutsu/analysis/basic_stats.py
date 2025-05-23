import pandas as pd

def calc_win_rate(df_race, key):
    # key: 'horse_id', 'jockey_id', 'trainer_id' など
    total = df_race.groupby(key).size()
    win = df_race[df_race['着順'] == 1].groupby(key).size()
    win_rate = (win / total).fillna(0)
    print(f"{key}ごとの勝率:")
    print(win_rate)
    return win_rate

if __name__ == "__main__":
    # テスト用ダミーデータ
    df = pd.DataFrame({'horse_id': [1,1,2,2,3], '着順': [1,2,1,3,2]})
    calc_win_rate(df, 'horse_id') 