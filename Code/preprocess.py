import cv2
import math
import numpy as np


class Preprocessor:
    def preprocess(self, image_name):
        # Load the  image and convert to grayscale 
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

        #Filter  image using edge preserving filter
        image  = cv2.bilateralFilter(image, 10, 150, 150)

        # Apply   Thresholding to make image binary
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Deskew the image
        image = self.__deskew(image)

        return image



    # Finds the rotation angle that maximizes the combined variances of the row and column weights. 
    # Returns the deskewed image
    def __deskew(self, image):
        # Angle in degrees, used as  negative and positive distance from 0 degrees to search within for maximum 
        max_angle = 10
        
        # Inverts colors so background is black since black is added to borders when rotating 
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Exhaustive search on linear sampling of angles within the bounds
        # Finds angle that gives max variance in row and column weights.  
        angles = np.linspace(-max_angle, max_angle, num=101, endpoint=True)
        max_var = 0
        deskewed = image
        for angle in angles:
            rotated_image = self.__rotate(angle, image)     # Apply rotation to image 
            row_weights, col_weights = self.__get_weights(rotated_image)    # Calculate row and column weights
            var = np.var(row_weights) + np.var(col_weights)     # Calculate combined variance of weights
            if var > max_var:   # Keep the Image with the max variance 
                max_var = var
                deskewed = rotated_image;
        
        # Revert to white background and black foreground 
        _, image = cv2.threshold(deskewed, 127, 255, cv2.THRESH_BINARY_INV)
            
        return image
        

    #Returns the row and column weights of the image 
    def __get_weights(self, image):
        # Calculate row_weights
        row_weights = np.zeros(image.shape[0], dtype=int)
        for i in range(image.shape[0]):
            row_weights[i] = np.sum(image[i, :]) / image.shape[1]
        # Calculate col_weights 
        col_weights = np.zeros(image.shape[1], dtype=int)
        for i in range(image.shape[1]):
            col_weights[i] = np.sum(image[:, i]) / image.shape[0]
        return row_weights, col_weights

    # Applies given rotation angle to image
    # Rotation causes corners of the rectangle to go outside original bounding box, so image is scaled down to compensate 
    def __rotate(self, angle, image):
        height, width = image.shape[:2]
        
        # Calculate scaling factor to prevent corners being lost 
        c = math.sqrt(pow(width/2, 2) + pow(height/2, 2))   # Distance from rotation point to corner 
        new_h = 2 * c * math.sin(math.radians(45 + angle))  # Height of bounding box after rotation
        scale = height / new_h
        
        # Rotation matrix to rotate image around center point with given angle and calculated scale reduction 
        rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        
        # Returns image with rotation applied 
        return cv2.warpAffine(image, rot_mat, dsize=(width, height)) 
           
