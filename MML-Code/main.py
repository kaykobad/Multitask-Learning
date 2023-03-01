import torch
import wandb
import json
from tqdm import tqdm
import torch.optim as optim
import torchnet as tnt
from torch import nn
import numpy as np
from torch.autograd import Variable
from dataloaders import load_dataset, Datasets, load_kinetics_dataset
from my_models import ModalitySpecificTransformer
from datetime import datetime
# from model_summary import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class Manager(object):
    def __init__(self, mm_model):
        self.model = mm_model
        self.batch_size = batch_size

        # self.train_dataset, self.train_loader, self.test_dataset, self.test_loader = load_dataset(dataset, batch_size=batch_size, frame_per_clip=frame_per_clip)
        self.train_dataset, self.train_loader, self.test_dataset, self.test_loader = load_kinetics_dataset(batch_size=batch_size, frame_per_clip=frame_per_clip, audio_sampling_rate=audio_sampling_rate)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        ts_running_loss = 0

        with torch.no_grad():
            # TODO: Add audio
            for video, audio, label in tqdm(self.test_loader, desc='Eval'):
                # for video, label in tqdm(self.test_loader, desc='Eval'):
                video = video.to(device)
                audio = audio.to(device)
                label = label.to(device)

                print("Eval:", audio.shape, video.shape, label)

                start_time = datetime.now()
                output = self.model(video, audio)
                diff = datetime.now() - start_time
                print("Eval Forward calculation per batch Took time(s):", diff.total_seconds())

                # print(output.shape)
                loss = self.criterion(output, label)
                ts_running_loss += loss.item()

                if error_meter is None:
                    topk = [1]
                    if output.size(1) > 5:
                        topk.append(5)
                    error_meter = tnt.meter.ClassErrorMeter(topk=topk)
                error_meter.add(output.data, label)

                # Detaching values
                video.detach()
                audio.detach()
                label.detach()
                output.detach()

        errors = error_meter.value()
        accuracy = [100-k for k in errors]
        print('Test Error: ' + ', '.join('@%s=%.2f' % t for t in zip(topk, errors)))
        print('Test Accuracy: ' + ', '.join('@%s=%.2f' % t for t in zip(topk, accuracy)))
        ts_final_loss = ts_running_loss / len(self.test_loader)

        return ts_final_loss, errors

    # def do_batch(self, optimizer, video, audio, label):
    def do_batch(self, optimizer, video, audio, label):
        video = video.to(device)
        audio = audio.to(device)
        label = label.to(device)
        video = Variable(video)
        audio = Variable(audio)
        label = Variable(label)

        optimizer.zero_grad()

        # Do forward-backward.
        # print(audio.shape)
        start_time = datetime.now()
        output = self.model(video, audio)
        diff = datetime.now() - start_time
        print("Train Forward calculation per batch Took time(s):", diff.total_seconds())
        loss = self.criterion(output, label)

        start_time = datetime.now()
        loss.backward()
        diff = datetime.now() - start_time
        print("Train Backward calculation per batch Took time(s):", diff.total_seconds())
        # print(self.model.classification_head.linear.weight)

        # Detaching values
        loss.detach()
        video.detach()
        audio.detach()
        label.detach()
        output.detach()
        del video
        del audio
        del label
        del output

        optimizer.step()
        return loss.item()

    def do_epoch(self, epoch_idx, optimizer):
        tr_running_loss = 0
        for video, audio, label in tqdm(self.train_loader, desc='Epoch: %d ' % epoch_idx):
            # for video, label in tqdm(self.train_loader, desc='Epoch: %d ' % epoch_idx):
            # loss = self.do_batch(optimizer, video, label)
            loss = self.do_batch(optimizer, video, audio, label)
            tr_running_loss += loss
            del loss

        tr_final_loss = tr_running_loss / len(self.train_loader)

        return tr_final_loss

    def save_model(self, epoch, best_accuracy, errors, savename):
        ckpt = {
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'model': self.model,
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, scheduler, save=True, savename='', best_accuracy=[0, 0]):
        best_accuracy = best_accuracy
        error_history = []

        self.model = self.model.to(device)

        self.eval()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % epoch_idx)

            self.model.train()

            tr_loss = self.do_epoch(epoch_idx, optimizer)

            optimizer.get_lr()

            for sch in scheduler:
                sch.step()
            ts_loss, errors = self.eval()
            error_history.append(errors)
            accuracy = [100-k for k in errors]

            wandb.log({
                "epoch": epoch_idx,
                "train loss": tr_loss,
                # "train accuracy": 1.0 - tr_loss,
                "test error": 1 - (accuracy[0] / 100.0),
                "test accuracy": accuracy[0] / 100.0,
            })

            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'training_loss': tr_loss,
                    # 'training_accuracy': 1.0 - tr_loss,
                    'test_error': 1 - (accuracy[0] / 100.0),
                    'test_accuracy': accuracy[0] / 100.0,
                }, fout)

            # Save best model, if required.
            if save and accuracy[0] > best_accuracy[0]:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' % (best_accuracy[0], accuracy[0]))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

                # Print Mask
                # print("Last mask:", self.model.module.layer4[2].mask3.mask.view(1, 1, 1, -1))
                # print("Last mask norm:", torch.norm(self.model.module.layer4[2].mask3.mask.data))
                # print("Last mask:", self.model.module.layer4[2].mask3.mask.data)
                # # print("First mask:", self.model.module.layer1[0].mask1.mask.view(1, 1, 1, -1))
                # print("First mask norm:", torch.norm(self.model.module.layer1[0].mask1.mask.data))
                # print("First mask:", self.model.module.layer1[0].mask1.mask.data)

        print('Finished finetuning...')
        print('Best error/accurac (Top-1): %0.2f%%, %0.2f%%' % (100 - best_accuracy[0], best_accuracy[0]))
        if len(best_accuracy) > 1:
            print('Best error/accuracy (Top-5): %0.2f%%, %0.2f%%' % (100 - best_accuracy[1], best_accuracy[1]))
        print('-' * 16)


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, lr_decay_factor=0.1):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.lr_decay_factor = lr_decay_factor

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def step_lr(self, epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
        """Handles step decay of learning rate."""
        factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
        new_lr = base_lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print('Set lr to ', new_lr)
        return optimizer

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = self.step_lr(
                epoch_idx, init_lr, decay_every,
                self.lr_decay_factor, optimizer)


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def get_lr(self):
        for i, op in enumerate(self.optimizers):
            # for param_group in op.param_groups:
            print(i, 'lr', op.param_groups[0]['lr'])


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total_params


def main():
    wandb.init(project="MTL-with-Transformer", entity="kaykobad", name=wandb_name)
    print('number of output layer and dataset: ', num_outputs, dataset)

    model = ModalitySpecificTransformer(num_classes=num_outputs, batch_dim=(batch_size, frame_per_clip, 3, 224, 224), audio_info=(audio_sampling_rate, audio_duration))
    model = nn.DataParallel(model)
    model = model.to(device)

    for name, param in model.named_parameters():
        # print(name, name.split("."), "classification_head" not in name, "classification_head" not in name.split("."))
        if optimize_ln and ('layernorm' in name):
            param.requires_grad = True
            print(name, "Not freezed")
        elif ('classification_head' not in name) and ('video_embedding' not in name) and ('patch_embeddings' not in name): # and ('layernorm' not in name):
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(name, "Not freezed")

    trainable_params, total_params = count_parameters(model)
    print('Total number of trainable parameters: ', trainable_params)
    print('Total number of parameters: ', total_params)
    params_to_optimize = model.parameters()
    optimizer = optim.Adam(params_to_optimize, lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    schedulers = [scheduler]
    optimizers = MultipleOptimizer(optimizer)

    # print("Model summary:")
    # total_params, trainable_params = summary(model, (8, 3, 224, 224))
    # print("Total params:", total_params, "Trainable params:", trainable_params)

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(model)

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # print(model)

    manager = Manager(model)
    manager.train(finetune_epochs, optimizers, schedulers, save=True, savename=save_name)

    # test_dataset, test_dataloader = load_dataset(Datasets.ucf101)
    # v, a, l = next(iter(test_dataloader))

    # v = torch.rand(4, 8, 3, 224, 224).cuda()
    # a = torch.rand(1, 8, 3, 224, 224)
    #
    # model = ModalitySpecificTransformer()
    # model.cuda()
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name)
    # model.eval()
    # out = model(v)
    # print(out["output"].shape)
    # print(out["output"])
    # v = torch.rand(768)
    #
    # model = ClassificationHead()
    # model.eval()
    # out = model(v)
    # print(out["out"])
    #
    # print(video.shape, video[0].shape, video[1].shape)
    # print(audio.shape, audio[0].shape, audio[1].shape)
    # print(label.shape, label[0].shape, label[1].shape)


if __name__ == '__main__':
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    NUM_OUTPUTS = {
        "ucf101": 2,
        "hmdb51": 51,
        "kinetics400": 3,
    }

    optimize_ln = False
    # modalities = [False, True]
    frame_per_clip = 32           # 4, 8, 16, 32
    audio_sampling_rate = 16000  # 16000, 44100, 48000
    audio_duration = 10
    dataset = 'kinetics400'
    checkpoint_suffix = '_fc-3-class-time'
    batch_size = 32
    lr = 5e-3
    finetune_epochs = 30
    save_name = 'checkpoints/' + dataset + checkpoint_suffix + '.pth'
    num_outputs = NUM_OUTPUTS[dataset]
    wandb_name = 'Kinetics-3C-A-fc16k-30'

    # Setting the seed
    torch.manual_seed(0)
    # random.seed(0)
    np.random.seed(0)

    main()
