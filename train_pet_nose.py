import torch, torchsummary, argparse, datetime
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from pet_nose_model import PetNoseModel
from pet_nose_loader import PetNoseLoader
from torch.utils.data import DataLoader


def traintest(n_epochs, model, criterion, scheduler, optimizer, device, train_loader, test_loader, plt_name):
    print('training...')
    train_losses = []
    test_losses = []

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append((running_loss/len(train_loader)))

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, running_loss / len(train_loader)), end=', ')
        
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

            print(f"Test Loss: {test_loss/len(test_loader)}")
            test_losses.append(test_loss/len(test_loader))
        
        scheduler.step(running_loss)

    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plt_name)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help='Epochs')
    parser.add_argument('-b', type=int, help="Batch size")
    parser.add_argument('-lr', type=float, help="Learning rate")
    parser.add_argument('-p', type=str, help="Loss Plot")
    parser.add_argument('-s', type=str, help="Weight File")
    parser.add_argument('-cuda', type=str, help="Cuda? [Y/N]")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda == 'Y' else 'cpu')

    model = PetNoseModel()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = PetNoseLoader('../data/images', labels_file='../data/train_noses.3.txt', transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True)

    test_set = PetNoseLoader('../data/images', labels_file='../data/test_noses.txt', transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True)

    traintest(n_epochs=args.e, model=model, criterion=criterion, scheduler=scheduler, optimizer=optimizer, device=device, train_loader=train_loader, test_loader=test_loader, plt_name=args.p)

    torchsummary.summary(model, (3, 224, 224))
    torch.save(model.state_dict(), args.s)

if __name__ == "__main__":
    main()
