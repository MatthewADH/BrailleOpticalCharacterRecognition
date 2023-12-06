# BrailleOpticalCharacterRecognition
A project for University of South Carolina CSCE 580

This project presents a method for converting hardcopy braille to the electronic braille BRF file format. A braille page can be scanned using a standard scanner and this system will automatically convert the resulting image to a BRF file. A basic chatbot can be used to access the system. A detailed description of how the system works can be found in the report under the docs directory. 

## Braille OCR procedure
preprocessing -> character segmentation -> character recognition -> postprocessing 

## Code
* chatbot.py - Rule based chaptbot to interface with BOCR. Run this script as  entry point.
* bocr.py - Contains BOCR object to perform BOCR on an image.
* preprocess.py - Used by BOCR, filters, binarizes, and deskews input image.
* segment.py - Used by BOCR, takes preprocessed image and Seperates characters into individual images, re-adds spacial context to segmented images.
* char_recognition.py - Used by BOCR, loads CNN model and uses it to make predictions on segmented characters. 
* postprocess.py - Used by BOCR, converts integer predictions to BRF string using brf.py
* brf.py - Contains lookup table to convert integer representation of braille characters to BRF ascii characters  
* dataprep.py - Script used to convert datasets of scanned characters into preprocessed, segmented, and tagged datasets
* train.py - Script used to define and train CNN character recognition model

## Data
* Datasets used to train character recognition model. 

## Docs
* Project Report
* Project Presentation

## Model
* h5 file defining CNN character recognition model

## Test
* validation.png - Paragraph used for validation, achieved 99.2% accuracy
* preprocess/ - images showing stages of preprocessing
* segment/ - images showing stages of segmentation 