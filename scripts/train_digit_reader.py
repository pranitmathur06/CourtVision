"""A digit classifier trained on SVHN, not on my own labels.

Every reader so far has been nearest-neighbour over the hand-labelled crops
themselves -- about 130 digit instances, which is a training set two orders of
magnitude too small. SVHN is 73k real-world digits under exactly the conditions
that break a jersey read: low resolution, motion blur, odd fonts, both
polarities, distractor digits crowding the sides.

Using it keeps the hand labels for EVALUATION ONLY, which also removes the
leave-one-out contortions the nearest-neighbour evaluation needed.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.datasets import SVHN

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SIZE = 32


def to_gray(images):
    # SVHN arrives as (N, 3, 32, 32) uint8.
    x = images.astype(np.float32) / 255.0
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1);  self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1); self.b2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1); self.b4 = nn.BatchNorm2d(64)
        self.c5 = nn.Conv2d(64, 128, 3, padding=1); self.b5 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = F.relu(self.b3(self.c3(x)))
        x = F.max_pool2d(F.relu(self.b4(self.c4(x))), 2)
        x = F.max_pool2d(F.relu(self.b5(self.c5(x))), 2)
        return self.fc(self.drop(x.flatten(1)))


def degrade(batch):
    """Make SVHN look like a jersey crop from a 720p broadcast.

    Three things separate the domains and each is simulated here: the number is
    read after an upscale from ~30 px so it is soft; the camera is moving so it
    smears; and a jersey is as often light-on-dark as dark-on-light.
    """
    n = batch.shape[0]
    # Blur, by downsampling and coming back up.
    # (MPS area-pooling needs divisible sizes, hence 8 and 16 rather than a
    # continuous blur radius.)
    for coarse in (8, 16):
        keep = torch.rand(n, device=batch.device) < 0.35
        small = F.interpolate(batch, size=coarse, mode="area")
        soft = F.interpolate(small, size=SIZE, mode="bilinear", align_corners=False)
        batch = torch.where(keep[:, None, None, None], soft, batch)
    # Polarity: a white number on a dark kit and the reverse are both common.
    flip = (torch.rand(n, 1, 1, 1, device=batch.device) < 0.5).float()
    batch = flip * (1 - batch) + (1 - flip) * batch
    # Contrast and brightness.
    gain = 0.5 + torch.rand(n, 1, 1, 1, device=batch.device)
    bias = (torch.rand(n, 1, 1, 1, device=batch.device) - 0.5) * 0.4
    batch = (batch - 0.5) * gain + 0.5 + bias
    # Per-sample standardisation, which is what inference will do too.
    m = batch.mean(dim=(1, 2, 3), keepdim=True)
    s = batch.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    return (batch - m) / s


def main():
    train = SVHN("svhn", split="train")
    test = SVHN("svhn", split="test")
    xtr = torch.from_numpy(to_gray(train.data)); ytr = torch.from_numpy(train.labels).long()
    xte = torch.from_numpy(to_gray(test.data));  yte = torch.from_numpy(test.labels).long()
    print(f"  SVHN train {len(xtr)}  test {len(xte)}")

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs, batch = 8, 256
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=epochs * (len(xtr) // batch + 1))
    torch.manual_seed(0)
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(xtr), batch):
            idx = order[i:i + batch]
            xb = degrade(xtr[idx].to(DEV)); yb = ytr[idx].to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb, label_smoothing=0.05)
            loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(idx)
        net.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(xte), 1024):
                xb = degrade(xte[i:i + 1024].to(DEV))
                correct += int((net(xb).argmax(1).cpu() == yte[i:i + 1024]).sum())
        print(f"  epoch {epoch}  loss {total/len(xtr):.3f}  degraded test acc {correct/len(xte):.3f}")
    torch.save(net.state_dict(), "models/jersey_digits.pt")

    # Pick the confidence floor HERE, on SVHN's own held-out digits under the
    # same degradation -- never on the jersey labels, which stay a clean test.
    probs = []
    with torch.no_grad():
        for i in range(0, len(xte), 1024):
            xb = degrade(xte[i:i + 1024].to(DEV))
            probs.append(F.softmax(net(xb), dim=1).cpu())
    probs = torch.cat(probs)
    conf, pred = probs.max(1)
    print("\n  floor  coverage  digit precision   (chosen on SVHN, not on jerseys)")
    for floor in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = conf >= floor
        if int(keep.sum()) == 0:
            continue
        print(f"  {floor:>5.2f}{float(keep.float().mean()):>10.0%}"
              f"{float((pred[keep] == yte[keep]).float().mean()):>17.3f}")


if __name__ == "__main__":
    main()
