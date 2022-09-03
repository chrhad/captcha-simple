import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import SimpleCaptchaReader
from utils import LABELS
from data import split_traindev

def collate_fn(batch):
    output = {}
    for k in batch[0].keys():
        output[k] = torch.stack([sample[k] for sample in batch], dim=0)
    return output

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Train model to read captcha")
    argparser.add_argument("input_dir", help="Input directory containing image files")
    argparser.add_argument("label_dir", help="Label directory containing text files corresponding to each image")
    argparser.add_argument("--output-model", default="model.pt",
        help="Model file ouptut path (default: %(default)s)")
    argparser.add_argument("--window-size", type=int, default=6,
        help="Sliding window size to extract area (default: %(default)s)")
    argparser.add_argument("--hidden-size", type=int, default=64,
        help="Hidden dimension size to represent features (default: %(default)s)")
    argparser.add_argument("--dropout", type=float, default=0.1, help="Dropout coefficient (default: %(default)s)")
    argparser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: %(default)s)")
    argparser.add_argument("--heldout", type=float, default=0.1,
        help="Proportion held for tuning (default: %(default)s)")
    argparser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: %(default)s)")
    argparser.add_argument("--max-epochs", type=int, default=100000,
        help="Maximum number of epochs (default: %(default)s")
    argparser.add_argument("--patience", type=int, default=10,
        help="Number of validation round to continue training after no improvement on heldout set (default: %(default)s)")
    argparser.add_argument("--valid-interval", type=int, default=1000,
        help="Number of epochs to perform one round of validation")
    args = argparser.parse_args()

    torch.manual_seed(123)  # fix random seed
    random.seed(123)

    train_dset, dev_dset = split_traindev(args.input_dir, args.label_dir, heldout=args.heldout)
    train_dloader = DataLoader(train_dset, args.batch_size, collate_fn=collate_fn)
    dev_dloader = DataLoader(dev_dset, 1, collate_fn=collate_fn)
    print("| Loaded {:d} instances of training data, {:d} instances of development data".format(len(train_dset), len(dev_dset)))

    # Model initialization and trainer parameters
    model = SimpleCaptchaReader(len(LABELS), args.window_size, args.hidden_size, args.dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=args.patience // 2)  # half the rounds of patience

    best = float('-inf')
    stall = 0  # number of times score does not improve on validation set
    for epoch in range(1, args.max_epochs+1):
        # Training iteration
        avg_loss = 0.
        model.train()
        for sample in train_dloader:
            optimizer.zero_grad()
            model.zero_grad()

            output = model(sample['image'], is_logits=True)  # do not convert to probability distribution first
            target = sample['label']
            loss = loss_fn(output.view(-1, len(LABELS)), target.view(-1))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(train_dloader)
        
        # Validation
        if epoch % args.valid_interval == 0:
            with torch.no_grad():
                model.eval()
                correct = 0  # number of correct predictions
                total = 0  # total number of characters
                for sample in dev_dloader:
                    preds = model(sample['image'])
                    _, argmax = preds.max(2)  # find argmax along the num_label dimension; batch_size x 5 x num_label
                    correct += (argmax == sample['label']).int().sum().item()
                    total += sample['label'].numel()
                accuracy = correct / total  # compute total accuracy in devset
                scheduler.step(accuracy)  # update score in scheduler, just in case lr needs to be reduced
                if best < accuracy:
                    best = accuracy
                    stall = 0  # reset
                    torch.save({
                        'epoch': epoch,
                        'args': args,
                        'model_state': model.state_dict(),
                    }, args.output_model)
                else:
                    stall += 1
                print("| Epoch {:d} | Learning rate {:.6f} | Training loss {:.4f} | Valid score {:3.2f}%".format(
                    epoch, optimizer.param_groups[0]['lr'], avg_loss, accuracy * 100.))
            
            if stall >= args.patience or optimizer.param_groups[0]['lr'] < 1e-7:
                print("| Early stopping due to plateau!")
                break
