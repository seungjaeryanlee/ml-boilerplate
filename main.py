import argparse
import datetime

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from networks import SimpleConvNet
from loggers import get_default_logger


def train(CONFIG, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % CONFIG.LOG.INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(CONFIG):
    # Initialize logger
    logger = get_default_logger()

    # Set device
    if CONFIG.USE_GPU is None: CONFIG.USE_GPU = torch.cuda.is_available()
    if CONFIG.USE_GPU: logger.info("Using GPU 💨")
    else: logger.warn("Using CPU 🐌")
    device = torch.device("cuda" if CONFIG.USE_GPU else "cpu")

    # Set random seed for reproducibility
    if CONFIG.SEED is None:
        CONFIG.SEED = datetime.datetime.now().strftime("%Y%m%d%H%M%S%z")
        logger.info(f"Random seed not specified: using seed {CONFIG.SEED} 🎲")
    torch.manual_seed(CONFIG.SEED)

    train_kwargs = {'batch_size': CONFIG.TRAIN.BATCH_SIZE}
    test_kwargs = {'batch_size': CONFIG.TEST.BATCH_SIZE}
    if CONFIG.USE_GPU:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = SimpleConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=CONFIG.TRAIN.LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=1, gamma=CONFIG.TRAIN.LEARNING_RATE_GAMMA)
    for epoch in range(1, CONFIG.TRAIN.EPOCHS + 1):
        train(CONFIG, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    # Load Configuration
    YAML_CONFIG = OmegaConf.load("configs/default.yaml")
    CLI_CONFIG = OmegaConf.from_cli()
    CONFIG = OmegaConf.merge(YAML_CONFIG, CLI_CONFIG)

    main(CONFIG)
