import os, sys
sys.path.append(os.pardir)
import json
import pandas as pd
import matplotlib.pylab as plt

class SaveResult():
    def __init__(self) -> None:
        self.overWrite = False

    # ファイルがすでに存在しているか確認
    def confirmExistance(self, fname):
        if os.path.exists(fname):
            if not self.overWrite:
                if input('上書きしますか？(yes : no):') == "yes":
                    self.overWrite = True
            if self.overWrite:
                return fname
            else:
                fname = input('別のファイル名を入力:')
        return fname

    # オブジェクトの保存
    def saveObj(self, fname, obj):
        fname = self.confirmExistance(fname)
        with open(fname, 'r') as f:
            json.dump(obj, f, indent=2)

    # 画像データの保存
    def saveImage(self, fname):
        fname = self.confirmExistance(fname)
        plt.savefig(fname, transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

    # 学習過程の保存
    def saveHistory(self, fname, history):
        fname = self.confirmExistance(fname)
        df = pd.DataFrame(history.history)
        df.to_csv(fname)

    # モデルの保存
    def saveModel(self, fname, model):
        fname = self.confirmExistance(fname)
        model.save(fname)
