# import gtts
# import playsound

# # text = input("i foung this object : " class_name)
# text = input("i foung this object : ")

# sound = gtts.gTTS(text,lang= "en")

# sound.save("obj_text.mp3")
# playsound.playsound("obj_text.mp3")

# import pyttsx3

# engine = pyttsx3.init()
# engine.say(class_name)
# engine.runAndWait()


# import pyttsx3

# engine = pyttsx3.init()

# while True:
#     text = input("Enter text to convert to speech: ")
#     engine.say(text)
#     engine.runAndWait()

import pyttsx3
import threading

class SpeechEngineThread(threading.Thread):
    def __init__(self, text):
        threading.Thread.__init__(self)
        self.text = text

    def run(self):
        engine = pyttsx3.init()
        engine.say(self.text)
        engine.runAndWait()

while True:
    text = input("Enter text to convert to speech: ")
    speech_thread = SpeechEngineThread(text)
    speech_thread.start()
