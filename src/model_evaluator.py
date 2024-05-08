import torch

from src.coatnet import CoAtNet


def load_and_prepare_model(model_path, device):
    num_blocks = [2, 2, 12, 28, 2]
    channels = [192, 192, 384, 768, 1536]
    model = CoAtNet((64, 64), 1, num_blocks, channels, num_classes=26)

    if model_path.endswith("model.pth"):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    else:
        model_checkpoint = torch.load(model_path, map_location=torch.device(device))
        model_state = model_checkpoint['model_state_dict']
        model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model

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