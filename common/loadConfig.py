import os
import yaml

class Config:
    def __init__(self):
        with open('common/config.yaml') as f:
            args = yaml.safe_load(f)
            # パス読み込み
            self.method = args['method']
            self.dataPath = args['dataPath']

            # 設定
            self.FIGSIZE = args['FIGSIZE']
            self.vectorTimes = args['vectorTimes']

            # 使用する指標
            self.aaIndex1 = args['aaindex1']
            self.aaIndex2 = args['aaindex2']

            # モデルの設定
            self.learningRate = args['learningRate']
            self.dropoutRatio = args['dropoutRatio']
            self.epochs = args['epochs']

        newDir = []

        # アミノ酸のデータのパス
        self.gpcrPath = os.path.join(self.dataPath, 'GPCR')
        self.cv0Path = os.path.join(self.gpcrPath, 'cv_0')
        self.trainSeqPath = os.path.join(self.cv0Path, 'train.txt')
        self.testSeqPath = os.path.join(self.cv0Path, 'test.txt')
        self.transFamilyPath = os.path.join(self.gpcrPath, 'trans.txt')

        # 画像データのパス
        self.methodImagePath = os.path.join(self.dataPath, self.method)
        newDir.append(self.methodImagePath)
        self.imageInfoPath = os.path.join(self.methodImagePath, 'imageInfo.csv')

        # 結果のパス
        currentDir = os.getcwd()
        self.resultDir = os.path.join(currentDir, 'result')
        newDir.append(self.resultDir)
        self.methodResultDir = os.path.join(self.resultDir, self.method + '_' + self.aaIndex1 + '_' + self.aaIndex2 + '_' + str(self.vectorTimes))
        newDir.append(self.methodResultDir)

        for dirName in newDir:
            if not os.path.exists(dirName):
                os.mkdir(dirName)

if __name__ == '__main__':
    config = Config()
