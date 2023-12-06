import cv2
import preprocess
import segment


class DataPrepper:
    def prepare(self, data_index, horizontal_flip=False, vertical_flip=False):
        pre = preprocess.Preprocessor()
        preprocessed_image = pre.preprocess(f"../Data/data{data_index}.png")
        
        seg = segment.Segmenter()
        segments = seg.segment(preprocessed_image)
        
        lines = len(segments)
        cols = len(segments[0])
        
        for i in range(lines):
            for j in range(cols):
                name = f"../Data/characters/{self.__get_int(i, j, lines, cols, horizontal_flip, vertical_flip)}_{data_index}.png"
                cv2.imwrite(name, segments[i][j])
        
    def __get_int(self, i, j, lines, cols, horizontal_flip, vertical_flip):
        
        if not horizontal_flip and not vertical_flip:
            return cols*i + j
        if horizontal_flip and not vertical_flip:
            original = cols*i + cols - 1 - j
            return self.__horizontal_flip(original)
        if not horizontal_flip and vertical_flip:
            original = cols*(lines - 1 - i) + j
            return self.__vertical_flip(original)
        if horizontal_flip and vertical_flip:
            original = cols*(lines - 1 - i) + cols - 1 - j
            return self.__horizontal_flip(self.__vertical_flip(original))
            
    def __horizontal_flip(self, i):
        first = (i & 0b111) << 3
        second = (i & 0b111000) >> 3
        return first + second
        
    def __vertical_flip(self, i):
        new = i & 0b010010
        new = new | ((i & 0b001001) << 2)
        new = new | ((i & 0b100100) >> 2)
        return new
        
        
        
# Driver Code 
dp = DataPrepper()

dp.prepare(0)

dp.prepare(1, horizontal_flip=True)
dp.prepare(2, vertical_flip=True)
dp.prepare(3, horizontal_flip=True, vertical_flip=True)
dp.prepare(4)
dp.prepare(5, horizontal_flip=True)
dp.prepare(6, vertical_flip=True)
dp.prepare(7, horizontal_flip=True, vertical_flip=True)

dp.prepare(8)
dp.prepare(9, horizontal_flip=True)
dp.prepare(10, vertical_flip=True)
dp.prepare(11, horizontal_flip=True, vertical_flip=True)

dp.prepare(12)
dp.prepare(13, horizontal_flip=True)
dp.prepare(14, vertical_flip=True)
dp.prepare(15, horizontal_flip=True, vertical_flip=True)


