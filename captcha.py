import argparse
import sys
import torch

from data import ExtractImagePixelSimple
from model import SimpleCaptchaReader
from utils import LABELS, to_label_str

class Captcha(object):
    def __init__(self, model_path="model.pt", verbose=False):
        # Load file from a model file, default 'model.pt'
        loaded = torch.load(model_path)  # dictionary of loaded parameters
        args = loaded['args']

        # Initialize and load saved model
        self.model = SimpleCaptchaReader(len(LABELS), 
            args.window_size, args.hidden_size, args.dropout)
        self.model.load_state_dict(loaded['model_state'])
        self.model.eval()
        if verbose:
            print("| Loaded model parameters from {}".format(model_path), file=sys.stderr)

        # Initialize transform-to-tensor function
        self.transform = ExtractImagePixelSimple()
        self.verbose = verbose

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        im_tensor = self.transform(im_path).unsqueeze(0)  # add pseudo-batch dimension
        pred_output = self.model(im_tensor)
        max_score, argmax = pred_output.max(2)
        output_str = ''.join(to_label_str(argmax[0].tolist()))
        if self.verbose:
            print("| Prediction: {} | Scores = {}".format(output_str, max_score[0]), file=sys.stderr)
        with open(save_path, 'w', encoding='UTF-8') as f:
            print(output_str, file=f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Captcha reader inference routine")
    argparser.add_argument("im_path", help="image path of Captcha to predict")
    argparser.add_argument("save_path", help="output text file path")
    argparser.add_argument("--model-path", default="model.pt",
        help="Trained model file path (default: %(default)s)")
    args = argparser.parse_args()

    cr = Captcha(args.model_path)
    cr(args.im_path, args.save_path)
