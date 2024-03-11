import os

import soundfile

import keystroke_extractor as ke

WINDOW_SIZE = 1023
HOP_SIZE = 225
BEFORE = int(0.2 * 14400)
AFTER = int(0.8 * 14400)

def create_recording(sentence, file_path):
    final_signal = []
    sentence_list = list(sentence.upper())
    for i in range(len(sentence_list)):
        FILE_PATH = "Recordings/" + sentence_list[i] + ".wav"
        signal, samplerate = ke.load_recording(FILE_PATH)
        energy = ke.process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
        peaks = ke.isolate_keystroke_peaks(energy)
        keystroke_boundaries = ke.find_keystroke_boundaries(peaks, signal, len(energy), BEFORE, AFTER)
        extracted_keystrokes = ke.isolate_keystrokes(keystroke_boundaries, signal)
        final_signal.append(extracted_keystrokes[0])

    # add silence between each keystroke
    silence = [0] * 14400
    for i in range(len(final_signal) - 1):
        final_signal.insert(i * 2 + 1, silence)
    
    final_signal = [item for sublist in final_signal for item in sublist]
    soundfile.write(os.path.join("Recordings", sentence + ".wav"), final_signal, samplerate)
    print("Recording created successfully at " + file_path)
    return final_signal, samplerate

if __name__ == "__main__":
    sentence = "ThisIsATestSentence"
    create_recording(sentence, os.path.join("Recordings", sentence + ".wav"))