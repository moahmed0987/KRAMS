# KRAMS: Keystroke Recognition using Augmented Mel-Spectrograms

A command-line interface application that uses [SpecAugmented](https://arxiv.org/pdf/1904.08779) Mel-spectrograms to train a [CoAtNet](https://arxiv.org/pdf/2106.04803) deep-learning model that classifies keystrokes in an attack recording. 

The application takes a directory containing training recordings, an attack recording, and the number of keystrokes in that recording as input.

**WARNING**: It is highly recommended to use a GPU to run this application. The training process is computationally expensive and will take a long time on a CPU. Services such as [Google Colab](https://colab.research.google.com/) can be used to run the application on a GPU for free.

## Installation

To get started, follow these simple steps:

Clone repository:

```bash
git clone https://github.com/moahmed0987/KRAMS.git
```

Set up virtual environment:

- **Windows**:
    ```bash
    cd KRAMS
    python -m venv venv
    .\venv\Scripts\activate
    ```

- **MacOS/Linux**:
    ```bash
    cd KRAMS
    python3 -m venv venv
    source venv/bin/activate
    ```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the following command replacing the placeholders with your specified file paths:

```bash
python KRAMS.py <TRAINING_RECORDINGS_DIR> <ATTACK_RECORDING_PATH> <N_KEYSTROKES_IN_ATTACK>
```
TRAINING_RECORDINGS_DIR: Path to the directory containing the training recordings.

ATTACK_RECORDING_PATH: Path to the recording to be attacked.

N_KEYSTROKES_IN_ATTACK: Number of keystrokes in the attack recording.