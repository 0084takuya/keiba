import matplotlib.pyplot as plt

def plot_bar(data, labels, title):
    plt.bar(labels, data)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # テスト用ダミーデータ
    plot_bar([10, 20, 15], ['A', 'B', 'C'], 'サンプル棒グラフ') 