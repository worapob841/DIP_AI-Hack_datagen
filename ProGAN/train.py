""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
    seed_everything
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import os
import torch.onnx
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=.5, hue=.3),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM,
                            1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha,
                                  step, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
            alpha=alpha
        )

    return tensorboard_step, alpha


def train_condition_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True, desc="🏋️ Training")
    for batch_idx, (real, labels) in enumerate(loop):
        real = real.to(config.DEVICE)
        labels = labels.to(config.DEVICE).long()
        # print(labels)
        # print(labels.shape)
        # print(real.shape)
        cur_batch_size = real.shape[0]
        # print(cur_batch_size)

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM -
                            config.NUM_CLASSES, 1, 1).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, labels, alpha, step)
            critic_real = critic(real, labels, alpha, step)
            critic_fake = critic(fake.detach(), labels, alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha,
                                  step, device=config.DEVICE,labels=labels)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, labels, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            # labels =
            # noise = torch.randn(8, config.Z_DIM-config.NUM_CLASSES, 1, 1).to(config.DEVICE)
            # labels = torch.randint(0,11,(8,1)).view(-1).to(config.DEVICE).long()
            # print('labels', labels.shape)
            img_cls = torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(config.DEVICE)
            noise = torch.randn(10, config.Z_DIM -
                            config.NUM_CLASSES, 1, 1).to(config.DEVICE)
            with torch.no_grad():
                fixed_fakes = gen(noise, img_cls, alpha, step) * 0.5 + 0.5
                # fixed_fakes = gen(config.FIXED_NOISE,labels, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
            alpha=alpha
        )

    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG, num_class=10
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
    print(gen)
    print(critic)

    # for tensorboard plotting
    writer = SummaryWriter(
        f"{config.ROOT_CHECKPOINT}/logs/gan_sz{config.START_TRAIN_AT_IMG_SIZE}")

    tensorboard_step = 0
    last_epoch = 0
    if config.LOAD_MODEL:
        tensorboard_step, last_epoch, alpha = load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        tensorboard_step, last_epoch, alpha = load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )
        # tensorboard_step,last_epoch,alpha = load_checkpoint(
        #     config.CHECKPOINT_GEN, gen.to(config.DEVICE), optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.99)), config.LEARNING_RATE,
        # )
        # tensorboard_step,last_epoch,alpha = load_checkpoint(
        #     config.CHECKPOINT_CRITIC, critic.to(config.DEVICE), optim.Adam(critic.parameters(),lr=1e-3, betas=(0.0, 0.99)), config.LEARNING_RATE,
        # )
        # print(gen.state_dict())
        # print(opt_gen.state_dict())

        # print(critic.state_dict())
        # print(opt_critic.state_dict())
        last_epoch += 1
        # load_checkpoint(
        #     config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        # )
        # load_checkpoint(
        #     config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        # )
        print(f'📄[INFO]\nALPHA = {alpha}\n' +
              f'\nTENSORBOARD_STEP = {tensorboard_step}\n'+f'LAST_EPOCH = {last_epoch}')
        tensorboard_step += 1
    print(str(os.system("tail -n +4 config.py"))+"\n")
    gen.train()
    critic.train()

    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    # print('step',step)
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        if not config.LOAD_MODEL:
            alpha = 1e-5  # start with very low alpha
        # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        loader, dataset = get_loader(4 * 2 ** step)
        print(f"📄[INFO] Current image size: {4 * 2 ** step}")

        for epoch in range(last_epoch, num_epochs):
            print(f"🚀 Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_condition_fn(
                # tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:

                # Check if the directory already exists
                gen_lastdir = f'{config.ROOT_CHECKPOINT}/generator_lastest'
                crit_lastdir = f'{config.ROOT_CHECKPOINT}/critic_lastest'
                if os.path.exists(gen_lastdir) and os.path.exists(crit_lastdir):
                    print(f"📄[INFO] Directory '{gen_lastdir}' already exists.")
                    print(f"📄[INFO] Directory '{crit_lastdir}' already exists.")
                gen_dir = f'{config.ROOT_CHECKPOINT}/generator_sz{4 * 2 ** step}'
                crit_dir = f'{config.ROOT_CHECKPOINT}/critic_sz{4 * 2 ** step}'
                if os.path.exists(gen_dir) and os.path.exists(crit_dir):
                    print(f"📄[INFO] Directory '{gen_dir}' already exists.")
                    print(f"📄[INFO] Directory '{crit_dir}' already exists.")
                else:
                    # Create the directory if it doesn't exist
                    try:
                        os.makedirs(gen_dir)
                        print(f"📄[INFO] Directory '{gen_dir}' created successfully.")
                        os.makedirs(crit_dir)
                        print(f"📄[INFO] Directory '{crit_dir}' created successfully.")
                        os.makedirs(gen_lastdir)
                        print(
                            f"📄[INFO] Directory '{gen_lastdir}' created successfully.")
                        os.makedirs(crit_lastdir)
                        print(
                            f"📄[INFO] Directory '{crit_lastdir}' created successfully.")
                    except OSError:
                        print(f"📄[INFO] Creation of directory failed.")

                save_checkpoint(gen, opt_gen, tensorboard_step=tensorboard_step, last_epoch=epoch, alpha=alpha,
                                filename=os.path.join(gen_dir, f'generator{epoch}_{step}_{tensorboard_step}.pth'))
                save_checkpoint(critic, opt_critic, tensorboard_step=tensorboard_step, last_epoch=epoch,
                                alpha=alpha, filename=os.path.join(crit_dir, f'critic{epoch}_{step}_{tensorboard_step}.pth'))
                save_checkpoint(gen, opt_gen, tensorboard_step=tensorboard_step, last_epoch=epoch, alpha=alpha,
                                filename=f'{config.ROOT_CHECKPOINT}/generator_lastest/generator_lastest.pth')
                save_checkpoint(critic, opt_critic, tensorboard_step=tensorboard_step, last_epoch=epoch,
                                alpha=alpha, filename=f'{config.ROOT_CHECKPOINT}/critic_lastest/critic_lastest.pth')

                # save_checkpoint(gen, opt_gen,tensorboard_step=tensorboard_step,last_epoch=epoch, filename=f'checkpoint_sz{config.START_TRAIN_AT_IMG_SIZE}/generator/generator_lastest.pth')
                # save_checkpoint(critic, opt_critic,tensorboard_step=tensorboard_step,last_epoch=epoch, filename=f'checkpoint_sz{config.START_TRAIN_AT_IMG_SIZE}/critic/critic_lastest.pth')
                # save_checkpoint(gen, opt_gen,tensorboard_step=tensorboard_step,last_epoch=epoch,alpha=alpha, filename=config.CHECKPOINT_GEN)
                # save_checkpoint(critic, opt_critic,tensorboard_step=tensorboard_step,last_epoch=epoch,alpha=alpha, filename=config.CHECKPOINT_CRITIC)

            print() 
        step += 1  # progress to the next img size
        writer = SummaryWriter(f"{config.ROOT_CHECKPOINT}/logs/gan_sz{4 * 2 ** step}")
        last_epoch = 0
        alpha = 1e-5



def export():

    print("EXPORTING MODEL...")
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    )
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()
    print(gen)
    print(critic)
    checkpoint = torch.load(
        '/mnt/c/Users/Worapob/Desktop/ML_playground/ProGAN/checkpoint/generator_sz64/generator29_4_842.pth', map_location="cuda")
    gen.load_state_dict(checkpoint["state_dict"])
    sample_input = torch.randn(1, config.Z_DIM, 1, 1)
    alpha = 0.5
    steps = 4
    onnx_filename = "generator.onnx"
    torch.onnx.export(gen, (sample_input, alpha, steps),
                      onnx_filename, verbose=True)


if __name__ == "__main__":
    seed_everything()
    main()
    # export()
