import os
import time
import torch
import argparse
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from som import SOM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--color', dest='dataset', action='store_const',
                        const='color', default=None,
                        help='use color')
    parser.add_argument('--mnist', dest='dataset', action='store_const',
                        const='mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--fashion_mnist', dest='dataset', action='store_const',
                        const='fashion_mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--train', action='store_const',
                        const=True, default=False,
                        help='train network')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.3, help='input learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='input total epoch')
    parser.add_argument('--data_dir', type=str, default='data', help='set a data directory')
    parser.add_argument('--res_dir', type=str, default='results', help='set a result directory')
    parser.add_argument('--model_dir', type=str, default='model', help='set a model directory')
    parser.add_argument('--row', type=int, default=20, help='set SOM row length')
    parser.add_argument('--col', type=int, default=20, help='set SOM col length')
    args = parser.parse_args()

    # Hyper parameters
    DATA_DIR = args.data_dir
    RES_DIR = args.res_dir + '/' + args.dataset
    MODEL_DIR = args.model_dir + '/' + args.dataset

    dataset = args.dataset
    batch_size = args.batch_size
    total_epoch = args.epoch
    row = args.row
    col = args.col
    train = args.train

    if train is True:
        if dataset == 'color':
            import color
            exit(0)
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        if dataset == 'mnist':
            train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        elif dataset == 'fashion_mnist':
            train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        else:
            print('Please set specify dataset. --mnist, --fashion_mnist')
            exit(0)
    
    train_data.train_data = train_data.train_data[:5000]
    train_data.train_labels = train_data.train_labels[:5000]

    print('Building Model...')
    som = SOM(input_size=28 * 28 * 1, out_size=(row, col))
    if os.path.exists('%s/som.pth' % MODEL_DIR):
        som.load_state_dict(torch.load('%s/som.pth' % MODEL_DIR))
        print('Model Loaded!')
    else:
        print('Create Model!')
    som = som.to(device)

    if train == True:
        losses = list()
        for epoch in range(total_epoch):
            running_loss = 0
            start_time = time.time()
            for idx, (X, Y) in enumerate(train_loader):
                X = X.view(-1, 28 * 28 * 1).to(device)    # flatten
                loss = som.self_organizing(X, epoch, total_epoch)    # train som
                running_loss += loss

            losses.append(running_loss)
            print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss, time.time() - start_time))

            if epoch % 5 == 0:
                # save
                som.save_result('%s/som_epoch_%d.png' % (RES_DIR, epoch), (1, 28, 28))
                torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)

        torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)
        plt.title('SOM loss')
        plt.plot(losses)
        plt.show()

    som.save_result('%s/som_result.png' % (RES_DIR), (1, 28, 28))
    torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)
