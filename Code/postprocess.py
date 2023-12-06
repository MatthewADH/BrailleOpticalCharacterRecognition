import brf

class Postprocessor:
    # Returns a string which is the contents of the BRF file from the given 2D array of predictions in integer form
    def postprocess(self, int_characters):
        # Remove trailing spaces 
        for row in int_characters:
            while row[-1] == 0:
                row.pop(-1)
        
        # Convert from integer representation to ascii character representation, preservinng character locations 
        brf_string = ""
        for row in int_characters:
            for char in row:
                brf_string += brf.to_ascii[char]
            brf_string += '\n'
        brf_string = brf_string[:-1]
        return brf_string