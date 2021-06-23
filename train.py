import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from models.generator import Generator
from models.dataset import GradientDataset

def pretrain(grad_data_loader, num_epochs=100, decay_factor=0.1, initial_lr=0.0001, save=True):
    generator = Generator()
    pretrain_criterion = F.mse_loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=initial_lr)

    # Push generator to gpu if it's available
    generator.to(device)
    generator.train()

    for epoch in range(num_epochs):
        if epoch == num_epochs // 2:
            for group in generator_optimizer.param_groups:
                group['lr'] = group['lr']*decay_factor

        running_loss = 0.0
        # Iterate through the dataloader
        for ii, (hr_grads, lr_grads) in enumerate(tqdm(grad_data_loader)):
            # Push data to gpu if available
            hr_grads, lr_grads = hr_grads.to(device), lr_grads.to(device)

            # Super-resolve low-resolution grads
            sr_grads = generator(lr_grads)

            # Compute loss, backpropagate, nad update generator
            loss = pretrain_criterion(sr_grads, hr_grads)
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()

            # Increment running loss
            running_loss += loss.item()
            del hr_grads, lr_grads, sr_grads

        print("Pretraining epoch {}, Average loss: {}".format(epoch, running_loss/len(grad_data_loader)))

    if save:
        # Save the final pretrained model if you're going to continue later
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'generator_optimizer': generator_optimizer},
                    'pretrained_grad_generator.pth.tar')


if __name__ == '__main__':
    data_dir = 'C:/Users/17175/Documents/Gradient_Refinement_GAN/Random_Surfaces'
    train_data = GradientDataset(data_dir, 'train', downsample_factor=4)
    # train_data.__getitem__(0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    pretrain(train_loader)
