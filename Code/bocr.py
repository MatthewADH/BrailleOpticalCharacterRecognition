import char_recognition
import cv2
import postprocess
import preprocess
import segment


class BOCR:
    def __init__(self):
        self.pre = preprocess.Preprocessor()
        self.seg = segment.Segmenter()
        self.cr = char_recognition.CharRecognizer()
        self.post = postprocess.Postprocessor()


    # Run BOCR process for input image file 
    def bocr(self, image_name): 
        preprocessed_image = self.pre.preprocess(image_name)
        
        segments = self.seg.segment(preprocessed_image)
        
        int_characters = []
        for i in range(len(segments)):
            int_characters.append([])
            for seg in segments[i]:
                int_characters[i].append(self.cr.recognize(seg))
        
        brf_string = self.post.postprocess(int_characters)
        
        return brf_string