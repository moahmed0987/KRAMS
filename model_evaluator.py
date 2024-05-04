import torch

from coatnet import CoAtNet


def predict_keystrokes(model, df_mel_spectrograms, device):
    output_labels = []
    with torch.no_grad():
        for ms in df_mel_spectrograms:
            ms = ms.to(device)
            ms = torch.unsqueeze(ms, 0)
            ms = torch.unsqueeze(ms, 0)
            output = model(ms)
            pred = output.argmax(dim=1, keepdim=True)
            output_labels.append(chr(pred.item()+65))
            print(chr(pred.item()+65), end=" ")
        print()
    return output_labels


def check_accuracy(target_labels, output_labels):
    if len(target_labels) != len(output_labels):
        print("Sample text and output text lengths do not match.")
        return 0
    correct = 0
    for i in range(len(target_labels)):
        if output_labels[i] == target_labels[i]:
            correct += 1
    print("Accuracy:", (correct / len(target_labels)) * 100, "%")
    return (correct / len(target_labels)) * 100