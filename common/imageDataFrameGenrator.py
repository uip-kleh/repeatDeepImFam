import os, sys
sys.path.append(os.pardir)
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

class ImageDataFrameGenerator:
    def __init__(self, imageDataDir) -> None:
        self.imageDataDir = imageDataDir
        self.imageInfoPath = os.path.join(self.imageDataDir, 'imageInfo.csv')

    def load(self):
        dataFrame = pd.read_csv(self.imageInfoPath, index_col=0)
        trainDataFrame, testDataFrame = train_test_split(
            dataFrame,
            test_size=.2,
            shuffle=True,
            random_state=1
        )

        imageDataGenerator = ImageDataGenerator(
            preprocessing_function=lambda img: 1 - img,
            rescale=1/255
        )

        trainDataFrameGenerator = imageDataGenerator.flow_from_dataframe(
            dataframe=trainDataFrame,
            directory=self.imageDataDir,
            shuffle=True,
            seed=0,
            x_col='imagePath',
            y_col='family',
            target_size=(256, 256),
            batch_size=256,
            color_mode="grayscale",
            class_mode="categorical",
            subset="training"
        )

        testDataFrameGenerator = imageDataGenerator.flow_from_dataframe(
            dataframe=testDataFrame,
            directory=self.imageDataDir,
            shuffle=False,
            # seed=0,
            x_col='imagePath',
            y_col='family',
            target_size=(256, 256),
            batch_size=256,
            color_mode="grayscale",
            class_mode="categorical",
        )

        return trainDataFrameGenerator, testDataFrameGenerator

if __name__ == '__main__':
    imageDataFrameGenerator = ImageDataFrameGenerator('/home/mizuno/data/half')
    imageDataFrameGenerator.load()
