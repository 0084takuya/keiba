import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('特徴量の相関')
    plt.show()
    return corr

if __name__ == "__main__":
    # テスト用ダミーデータ
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]})
    plot_correlation(df) 