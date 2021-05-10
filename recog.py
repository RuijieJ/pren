import os

from Nets.model import Model
from Utils.utils import *
from Configs.testConf import configs

import cv2
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])


def imread(imgpath):
    img = cv2.imread(imgpath)
    h, w, _ = img.shape

    x = transform(img)
    x.sub_(0.5).div_(0.5)
    x = x.unsqueeze(0)

    is_vert = True if h > w else False
    if is_vert:
        img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        x_clock = transform(img_clock)
        x_counter = transform(img_counter)
        x_clock.sub_(0.5).div_(0.5)
        x_counter.sub_(0.5).div_(0.5)
        x_clock = x_clock.unsqueeze(0)
        x_counter = x_counter.unsqueeze(0)
    else:
        x_clock, x_counter = 0, 0

    return x, x_clock, x_counter, is_vert


class Recognizer(object):

    def __init__(self, model):

        self.device = torch.device('cuda' if configs.cuda else 'cpu')

        self.model = model.to(self.device)
        self.model.eval()

        with open(configs.alphabet) as f:
            alphabet = f.readline().strip()
        self.converter = strLabelConverter(alphabet)

    def recog(self, imgpath):
        with torch.no_grad():

            x, x_clock, x_counter, is_vert = imread(imgpath)
            x = x.to(self.device)
            logits = self.model(x)  # [1, L, n_class]

            if is_vert:
                x_clock = x_clock.to(self.device)
                x_counter = x_counter.to(self.device)
                logits_clock = self.model(x_clock)
                logits_counter = self.model(x_counter)

                score, pred = logits[0].log_softmax(1).max(1)  # [L]
                pred = list(pred.cpu().numpy())
                score_clock, pred_clock = logits_clock[0].log_softmax(1).max(1)
                pred_clock = list(pred_clock.cpu().numpy())
                score_counter, pred_counter = logits_counter[0].log_softmax(1).max(1)
                pred_counter = list(pred_counter.cpu().numpy())

                scores = np.ones(3) * -np.inf

                if 1 in pred:
                    score = score[:pred.index(1)]
                    scores[0] = score.mean()
                if 1 in pred_clock:
                    score_clock = score_clock[:pred_clock.index(1)]
                    scores[1] = score_clock.mean()
                if 1 in pred_counter:
                    score_counter = score_counter[:pred_counter.index(1)]
                    scores[2] = score_counter.mean()

                c = scores.argmax()
                if c == 0:
                    pred = pred[:pred.index(1)]
                elif c == 1:
                    pred = pred_clock[:pred_clock.index(1)]
                else:
                    pred = pred_counter[:pred_counter.index(1)]

            else:
                pred = logits[0].argmax(1)
                pred = list(pred.cpu().numpy())
                if 1 in pred:
                    pred = pred[:pred.index(1)]

            pred = self.converter.decode(pred).replace('<unk>', '')

        return pred


if __name__== '__main__':
    checkpoint = torch.load(configs.model_path)
    model = Model(checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(configs.model_path))

    tester = Recognizer(model)

    imnames = os.listdir('samples')
    imnames.sort()
    impaths = [os.path.join('samples', imname) for imname in imnames]

    for impath in impaths:
        pred = tester.recog(impath)
        print('{}: {}'.format(impath, pred))
