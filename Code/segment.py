import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

class Segmenter:
    #Object used to collapse  neighboring lines of black or white space
    class Space:
        def __init__(self, start_index, width):
            self.start_index = start_index
            self.width = width
            
        def middle(self):
            return self.start_index + self.width // 2
        
        def end (self):
            return self.start_index + self.width - 1


    def __init__(self):
        self.image = None
        self.height = 0
        self.width = 0



    #Segment the image given as a parameter  
    # Assumes image has been preprocessed 
    def segment(self, input_image):
        self.image = input_image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        
        # Calculate row and column weights 
        row_weights, col_weights  = self.__get_weights()
        
        # Get lists of cuts to segment the image
        row_cuts = self.__get_cuts(row_weights, self.height)
        col_cuts = self.__get_cuts(col_weights, self.width)
        
        # Create 2D array of image segmetns from cuts that preserves location of characters
        segments = []
        for i in range(len(row_cuts)-1):
            segments.append([])
            for j in range(len(col_cuts)-1):
                # Cut out single character
                x1, y1 = col_cuts[j], row_cuts[i]
                x2, y2 = col_cuts[j+1], row_cuts[i+1]
                seg = self.image[y1:y2, x1:x2]
                
                # Shrink image to size for Neural Network 
                seg = self.__resize(seg)
                
                # Add context of line weights from surrounding image to preserve spacial information for dots 
                seg = self.__add_context(seg, x1, y1, x2, y2, row_weights, col_weights)
                
                # Add segmented image  to 2D array
                segments[i].append(seg)
                
        return segments
        
        
    #Returns the row and column weights of the image 
    def __get_weights(self):
        # Calculate row_weights
        row_weights = np.zeros(self.height, dtype=int)
        for i in range(self.height):
            row_weights[i] = np.sum(self.image[i, :]) / self.width
        # Calculate col_weights 
        col_weights = np.zeros(self.width, dtype=int)
        for i in range(self.width):
            col_weights[i] = np.sum(self.image[:, i]) / self.height
        return row_weights, col_weights

    #Returns list of indices to make cuts given the input weights
    def __get_cuts(self, weights, max_cut):
        img = weights.astype(np.uint8)
        
        # Apply Thresholding to weights to mark each line as white or black space 
        _, mask = cv2.threshold(img, 254.5, 255, cv2.THRESH_BINARY)
        
        # Combine neighboring whitespace into single space object
        whitespaces = self.__collapse_whitespace(mask)
        
        # Find whitespaces which mark seperation between characters  
        deliminators = self.__get_deliminators(whitespaces)
        
        # Make a list of the midpoints of whitespaces which seperate characters 
        midpoints = []
        for space in deliminators:
            midpoints.append(space.middle())
        
        # Use linear regression to make cuts at equal spacing 
        midpoints, slope = self.__fit_cuts(midpoints)
        
        # Add 2 outer cuts to list using the spacing from the linear model. Insures added cuts are inbounds
        cuts = []
        if midpoints[0] - 0.95*slope > 0:
            cuts.append(midpoints[0] - 0.95*slope)
        else:
            cuts.append(0)
        for point in midpoints:
            cuts.append(point)
        if midpoints[-1] + 0.95*slope < max_cut:
            cuts.append(midpoints[-1] + 0.95*slope)
        else:
            cuts.append(max_cut)
        
        return [int(cut) for cut in cuts]
        
    #Collapse neighboring lines of whitespace into a list of space objects
    def __collapse_whitespace(self, mask):
        whitespace_list = []
        current_space = None
        for i in range(len(mask)):
            if mask[i] > 128:  # Check if line is whitespace
                if current_space is None:
                    current_space = self.Space(i, 1)
                else:
                    current_space.width += 1
            else:
                if current_space is not None:
                    whitespace_list.append(current_space)
                    current_space = None
        if current_space is not None:
            whitespace_list.append(current_space)
        return whitespace_list

    # Take a list of whitespace objects and return a list of the whitespaces which seperate characters 
    def __get_deliminators(self, whitespaces):
        # Remove margins 
        whitespaces.pop(0)
        whitespaces.pop(-1)
        
        # Make an array of space sizes
        space_sizes = np.zeros(len(whitespaces), dtype=int)
        for i  in range(len(whitespaces)):
            space_sizes[i] = whitespaces[i].width
        
        # Apply Otsu's Thresholding to space_sizes to find threshold to mark each space as line seperation or not 
        threshold, _ = cv2.threshold(space_sizes.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply threshold value to get list of deliminators
        deliminators = []
        for space in whitespaces:
            if space.width > threshold:
                deliminators.append(space)
        
        return deliminators
    
    # Fit a linear model to cuts to give then equal spacing
    def __fit_cuts(self, cuts):
        # Linear Model
        line = np.polynomial.Polynomial.fit(range(len(cuts)), cuts, 1)
        
        # Cuts from linear model 
        _, new_cuts = line.linspace(len(cuts), [0, len(cuts)-1])
        return list(new_cuts), line.convert().coef[1]
    
    # Shrink character segment to proper size for neural network 
    def __resize(self, image):
        img = cv2.resize(image, (20,30), interpolation=cv2.INTER_AREA)
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bin_img

    # Add line weights to include spacial information from surrounding image in segmented image
    def __add_context(self, image, x1, y1, x2, y2, row_weights, col_weights):
        # cut out relevant section of row  weights  and srink it to 30 pixel length 
        row_weights = row_weights[y1:y2]
        row_weights = self.__shrink_weights(row_weights, 30)
        
        # Add row weights to right side of image 
        row_weights = np.asmatrix(row_weights)
        row_weights = np.transpose(row_weights)
        image = np.append(image, row_weights, axis=1)
        
        # cut out relevant section of column   weights  and srink it to 20 pixel length 
        col_weights = col_weights[x1:x2]
        col_weights = self.__shrink_weights(col_weights, 20)
        
        # Add column  weights to bottom  of image 
        col_weights = np.append(col_weights, 0)
        col_weights = np.asmatrix(col_weights)
        image = np.append(image, col_weights, axis=0)
        
        return image
        
    # Resize line weights to reduced size for neural network
    def __shrink_weights(self, weights, size):
        # starting index for bins
        cuts = np.linspace(0, weights.shape[0]-1, size+1, dtype='int')
        
        # Make each pixel be the average weight of the corresponding pixels in original image 
        shrunk = np.zeros(size)
        for i in range(size):
            shrunk[i] = np.average(weights[cuts[i]:cuts[i+1]])
        
        # Make the minimum weight be 0 and the maximum weight be 255
        shrunk = shrunk - np.min(shrunk)
        shrunk = (shrunk / np.max(shrunk)) * 255.0
        
        return shrunk
        
        
        
        
# Methods used for visualization during debugging   
    def __plot(self, data, title="", xlabel="Row Index", ylabel=""):
        plt.figure(figsize=(8, 6))
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def __show_cuts(self, row_cuts, col_cuts):
        # Set the color for the lines (gray color)
        line_color = (192, 192, 192)  # In BGR format

        # Copy the image to avoid modifying the original
        image_with_lines = self.image.copy()

        # Draw gray  lines on the image
        for cut  in row_cuts:
            cv2.line(image_with_lines, (0, cut), (self.width-1, cut), line_color, 2)
        for cut  in col_cuts:
            cv2.line(image_with_lines, (cut, 0), (cut, self.width-1), line_color, 2)

        # Display the image with the lines
        #image_with_lines = cv2.resize(image_with_lines, (960, 540))                # Resize image
        cv2.imshow("output", image_with_lines)                       # Show image
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __show_whitespace(self, row_weights, col_weights):
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        
        _, row_mask = cv2.threshold(row_weights.astype(np.uint8), 254.5, 255, cv2.THRESH_BINARY)
        row_spaces = self.__collapse_whitespace(row_mask)
        for space in row_spaces:
            img = cv2.rectangle(img, (0, space.start_index), (self.height-1, space.end()), 255, -1)
        
        _, col_mask = cv2.threshold(col_weights.astype(np.uint8), 254.5, 255, cv2.THRESH_BINARY)
        col_spaces = self.__collapse_whitespace(col_mask)
        for space in col_spaces:
            img = cv2.rectangle(img, (space.start_index, 0), (space.end(), self.width-1), 255, -1)
        
        cv2.imshow("Whitespaces", img)                       # Show image
        cv2.waitKey(0)
        cv2.destroyAllWindows()
