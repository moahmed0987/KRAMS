def krams(training_recordings_dir, attack_recording_path, window_size, hop_size, before, after, num_epochs, epochs_per_checkpoint, batch_size, learning_rate, device, num_peaks):
    import os

    import model_evaluator as me
    import train_model as tm
    import training_data_processor as tdp
    from coatnet import CoAtNet
    base_dir = tm.run(training_recordings_dir, window_size, hop_size, before, after, num_epochs, epochs_per_checkpoint, batch_size, learning_rate, device)
    checkpoint_dir = os.path.join(base_dir, "Checkpoints")
    model_dir = os.path.join(base_dir, "Model")
    model_path = os.path.join(model_dir, "model.pth")
    data_dir = os.path.join(base_dir, "Data")
    mel_spectrograms = tdp.unaugmented_data_processing_pipeline(attack_recording_path, window_size, hop_size, before, after, num_peaks)
    from torch import Tensor
    mel_spectrograms = Tensor(mel_spectrograms)
    model = me.load_and_prepare_model(model_path, device)
    
    print("Predicted keystrokes:")
    me.predict_keystrokes(model, mel_spectrograms, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KRAMS: Keystroke Recognition using Augmented Mel-Spectrograms")
    parser.add_argument("TRAINING_RECORDINGS_DIR", help="The directory containing the training recordings")
    parser.add_argument("ATTACK_RECORDING_PATH", help="The path to the recording to attack")
    parser.add_argument("N_KEYSTROKES_IN_ATTACK", help="The number of keystrokes in the attack recording", type=int)
    args = parser.parse_args()
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    BEFORE = int(0.3 * 14400)
    AFTER = int(0.7 * 14400)
    NUM_EPOCHS = 1100
    EPOCHS_PER_CHECKPOINT = 10
    BATCH_SIZE = 130
    LEARNING_RATE = 0.0005
    from torch.cuda import is_available
    DEVICE = "cuda" if is_available() else "cpu"
    krams(args.TRAINING_RECORDINGS_DIR, args.ATTACK_RECORDING_PATH, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER, NUM_EPOCHS, EPOCHS_PER_CHECKPOINT, BATCH_SIZE, LEARNING_RATE, DEVICE, args.KEYSTROKES_IN_ATTACK)
