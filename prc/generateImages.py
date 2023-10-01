import os, sys
sys.path.append(os.pardir)
from common.deepImFam import DeepImFam

if __name__ == '__main__':
    deepimfam = DeepImFam()
    deepimfam.generateImages()
