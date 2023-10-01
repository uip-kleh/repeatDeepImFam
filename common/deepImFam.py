import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append(os.pardir)
import matplotlib.pylab as plt
import seaborn as sns
from aaindex import aaindex1
import statistics
import numpy as np
import pandas as pd
import keras
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from common.loadConfig import Config
from common.imageDataFrameGenrator import ImageDataFrameGenerator
from common.saveResult import SaveResult

class DeepImFam(Config, ImageDataFrameGenerator, SaveResult):
    def __init__(self):
        Config.__init__(self)
        ImageDataFrameGenerator.__init__(self, imageDataDir=self.methodImagePath)
        SaveResult.__init__(self)

        self.logConfig()
        isContinue = input('if you wanna continue, enter:')

    # GPCRデータセットを読み込む
    def loadAASquences(self):
        sequences = []
        keys = []

        with open(self.trainSeqPath, 'r') as f:
            for l in f.readlines():
                key, sequence = l.split()
                sequences.append(sequence)
                keys.append(key)

        with open(self.testSeqPath, 'r') as f:
            for l in f.readlines():
                key, sequence = l.split()
                sequences.append(sequence)
                keys.append(key)

        return sequences, keys

    # アミノ酸ベクトルを定義する
    def defineAAVector(self):
        # AAindexの指標を使用
        aaIndex1 = aaindex1[self.aaIndex1].values
        aaIndex2 = aaindex1[self.aaIndex2].values
        aaIndex1Val = aaIndex1.values()
        aaIndex2Val = aaIndex2.values()


        aaIndex1Mean = statistics.mean(aaIndex1Val)
        aaIndex1Std = statistics.stdev(aaIndex1Val)
        aaIndex2Mean = statistics.mean(aaIndex2Val)
        aaIndex2Std = statistics.stdev(aaIndex2Val)

        self.aaVector = {}
        for key in aaIndex1.keys():
            if key == '-': continue
            self.aaVector[key] = np.array([
                (aaIndex1[key] - aaIndex1Mean) / aaIndex1Std * self.vectorTimes,
                (aaIndex2[key] - aaIndex2Mean) / aaIndex2Std * self.vectorTimes
            ])
        # print(self.aaVector)

    def translateFamily(self):
        self.familyDict = {}
        self.subFamilyDict = {}
        self.subSubFamilyDict = {}
        with open(self.transFamilyPath) as f:
            for l in f.readlines():
                key, subSubFamily, Family, subFamily = l.split()
                # key = int(key)
                self.familyDict[key] = Family
                self.subFamilyDict[key] = subFamily
                self.subSubFamilyDict[key] = subSubFamily

    # アミノ酸配列を用いた画像を生成する
    def generateImages(self):
        sequences, keys = self.loadAASquences()
        self.defineAAVector()

        imagesInfo = []
        self.translateFamily()
        # self.plotSequce(sequences[0])

        for findex, (sequence, key) in enumerate(zip(sequences, keys)):
            x = self.FIGSIZE / 2
            y = self.FIGSIZE / 2
            xPoints = [x]
            yPoints = [y]
            fig = plt.figure(figsize=(self.FIGSIZE/100, self.FIGSIZE/100))
            plt.style.use('classic')
            plt.axis('off')

            sequence = sequence.replace('_', '')
            # print(sequence)
            n = len(sequence)
            # print(n)
            for i, c in enumerate(sequence):
                if not c in self.aaVector: continue
                grayScale = str(1 - i/n)
                bufx = x
                bufy = y
                x += self.aaVector[c][0]
                y += self.aaVector[c][1]

                # 周期的境界の実装
                if x < 0:
                    bufx += self.FIGSIZE / 2
                    x += self.FIGSIZE / 2
                if x > self.FIGSIZE:
                    bufx -= self.FIGSIZE / 2
                    x -= self.FIGSIZE / 2
                if y < 0:
                    bufy += self.FIGSIZE / 2
                    y += self.FIGSIZE / 2
                if y > self.FIGSIZE:
                    bufy -= self.FIGSIZE / 2
                    y -= self.FIGSIZE / 2

                # print(i, n, grayScale)

                plt.plot([bufx, x], [bufy, y], color=grayScale, linewidth=.6)

                xPoints.append(x)
                yPoints.append(y)

            plt.xlim([0, self.FIGSIZE])
            plt.ylim([0, self.FIGSIZE])
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            # plt.show()
            fname = os.path.join(self.methodImagePath, str(findex) + '.png')

            imageInfo = {}
            imageInfo['family'] = self.familyDict[key]
            imageInfo['subFamily'] = self.subFamilyDict[key]
            imageInfo['subSubFamily'] = self.subSubFamilyDict[key]
            imageInfo['imagePath'] = fname
            imagesInfo.append(imageInfo)

            plt.savefig(fname)
            plt.cla()
            plt.clf()
            plt.close()
            # break
        df = pd.DataFrame(data=imagesInfo)
        df.to_csv(self.imageInfoPath)

    # 学習

    def drawProcess(self, history, isAccuracy=True, isLoss=False):
        key = 'accuracy'
        if isLoss:
            isAccuracy ^= True
            key = 'loss'

        pltTrain = history.history[key]
        pltTest = history.history['val_' + key]
        epochs = [i+1 for i in range(len(pltTrain))]

        plt.figure()
        plt.title(key)
        plt.xlabel('epochs')
        plt.ylabel(key)
        plt.plot(epochs, pltTrain, label='train')
        plt.plot(epochs, pltTest, label='test')
        plt.legend()

    def drawHaetMap(self, cm, norm=True):
        if norm: sns.heatmap(cm, annot=True, square=True, cbar=True, cmap='Blues')
        else: sns.heatmap(cm, annot=True, square=True, cbar=True, cmap='Blues', fmt='d')
        plt.yticks(rotation=0)
        plt.xlabel("Pre", fontsize=13, rotation=0)
        plt.ylabel("GT", fontsize=13)

    def generateDeepImFamModel(self, optimizer=optimizers.Adam, learningRate=-1):
        if learningRate == -1: learningRate = self.learningRate
        model = keras.models.Sequential([
            keras.layers.Conv2D(16, (2, 2), padding='valid', input_shape=(self.FIGSIZE, self.FIGSIZE, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(16, (2, 2), padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (2, 2), padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (2, 2), padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (2, 2), padding='valid'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(self.dropoutRatio),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(self.dropoutRatio),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.outputShape, activation='softmax')
        ])

        model.summary()

        model.compile(
            optimizer=optimizer(learning_rate=learningRate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def trainImages(self):
        trainDataFrameGenerator, testDataFrameGenerator = self.load()

        self.outputShape = len(trainDataFrameGenerator.class_indices.keys())

        # モデル
        early_stopping =  EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=20,
        )

        model = self.generateDeepImFamModel()
        model.fit(
            trainDataFrameGenerator,
            validation_data=testDataFrameGenerator,
            epochs=self.epochs,
            callbacks=[early_stopping]
        )

        history = model.history

        fname = os.path.join(self.methodResultDir, 'model.h5')
        self.saveModel(fname, model)
        fname = os.path.join(self.methodResultDir, 'history.csv')
        self.saveHistory(fname, history)
        # Save Accuracy
        self.drawProcess(history=history)
        fname = os.path.join(self.methodResultDir, 'accuracy.pdf')
        self.saveImage(fname)
        # Save Loss
        self.drawProcess(history=history, isLoss=True)
        fname = os.path.join(self.methodResultDir, 'loss.pdf')
        self.saveImage(fname)

        # draw heatmap
        labels = np.array(testDataFrameGenerator.classes)
        predict = np.argmax(model.predict(testDataFrameGenerator), axis=1)
        cm = confusion_matrix(labels, predict)
        self.drawHaetMap(cm, norm=False)
        fname = os.path.join(self.methodResultDir, 'confusionMatrix.pdf')
        self.saveImage(fname)
            # 正規化
        normcm = confusion_matrix(labels, predict, normalize='true')
        self.drawHaetMap(normcm, norm=True)
        fname = os.path.join(self.methodResultDir, 'normConfusionMatrix.pdf')
        self.saveImage(fname)

if __name__ == '__main__':
    deepimfam = DeepImFam()
    deepimfam.generateImages()
    # deepimfam.trainImages()
