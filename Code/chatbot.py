import bocr
import cv2
import os
from datetime import datetime


# list of possible bot responses
bot_responses = {
    "greeting" : "Hello! I am a chat bot who can assist you in converting hard-copy braille into electronic braille.",
    "unknown" : "I don't understand. Try asking me what I can do.",
    "input_file" : "What is the file path to the photo you wish to BOCR.",
    "file_error" : "I encountered an error attempting to open the file. Maybe the file does not exist or is not a supported file type.",
    "OCR_begin" : "Thank you, I will begin the BOCR process.",
    "OCR_finished" : "The BOCR process has finished. You can find the BRF file at: ",
    "help" : "I am your interface to perform Braille Optical Character Recognition (BOCR) on an image file. Ask me to OCR a file for you."
    
}

greetings = [
    "hello",
    "hi",
    "hey",
    "greetings",
    "hola"
]

# Global Variables 
conversation_log = []

# Function to output Bot responses and add them to the conversation log
def printOutput(output):
    output = "Bot: " + output
    print(output)
    conversation_log.append(output)

# Function to get user input and save it to the conversation log
def getInput():
    user_input = input("You: ")
    conversation_log.append("User: " + user_input)
    return user_input     

# Function to save the conversation log
def save_log():
    filename = "../log/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_log.txt"
    with open(filename, "w") as conversation_file:
        conversation_file.write("\n".join(conversation_log))
    return filename

# Function to get the bot's response to a user input
def get_bot_response(input):
    for phrase in greetings:
        if phrase in input:
            return bot_responses["greeting"]
    if "ocr a file" in input:
        input_file = get_input_file()
        if input_file is not None:
            printOutput(bot_responses["OCR_begin"])
            output_file = do_BOCR(input_file)
            return bot_responses["OCR_finished"] + output_file
        else:
            return bot_responses["file_error"]
    elif "what can you do" in input:
        return bot_responses["help"]
    return bot_responses["unknown"]

# Function to get file from user
def get_input_file():
    printOutput(bot_responses["input_file"])
    
    file_path = getInput()
    if not os.path.exists(file_path):
        return None

    try:
        image = cv2.imread(file_path)
        if image is not None:
            return file_path
        else:
            return None
    except Exception as e:
        return None

# Function to perform BOCR on given file
def do_BOCR(input_file):
    ocr = bocr.BOCR()
    brf_string = ocr.bocr(input_file)

    filename = os.path.splitext(input_file)[0] + ".brf"
    with open(filename, "w") as file:
        file.write(brf_string)
    return filename

# Main chat loop
while True:
    user_input = getInput().lower()

    # Exit the loop if the user enters "quit" or "q"
    if user_input == "quit" or user_input == 'q':
        printOutput("Goodbye!")
        break

    bot_response = get_bot_response(user_input)
    printOutput(bot_response)

save_path = save_log()
print("Info: Chat log has been saved to: " + save_path)