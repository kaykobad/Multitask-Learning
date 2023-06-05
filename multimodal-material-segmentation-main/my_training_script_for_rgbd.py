import argparse
import os
import wandb
import numpy as np
from tqdm import tqdm
import random
from mypath import Path
from dataloaders.rgbd.prepare_data import prepare_data
from dataloaders import make_data_loader, make_data_loader2
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.my_deeplab_impl import *
# from modeling.my_deeplab_impl_3 import *
from modeling.my_deeplab_impl_for_rgbd import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels, calculate_weigths_labels_for_all
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, ConfusionMatrix


class TrainerMultimodalRGBD(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.test_loader = prepare_data(args, ckpt_dir=None)
        self.nclass = self.train_loader.dataset.n_classes_without_void + 1

        # calculate_weigths_labels_for_all(self.train_loader, self.test_loader, num_classes=self.nclass)

        # f

        # model = MMDeepLabSEMaskWithNormForRGBD(num_classes=self.train_loader.dataset.n_classes,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_rgb=args.use_rgb,
        #                 use_depth=args.use_depth,
        #                 norm=args.norm)

        # model = MMDeepLabSEMaskWithNormForRGBD(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_rgb=args.use_rgb,
        #                 use_depth=args.use_depth,
        #                 norm=args.norm)
        model = MMDeepLabSEMaskWithNormForRGBD(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        use_rgb=args.use_rgb,
                        use_depth=args.use_depth,
                        norm=args.norm)

        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total Parameters:", pytorch_total_params)
                        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr*10}]
        
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            print("Calculating Class Weights...")
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        # n_classes_without_void = self.train_loader.dataset.n_classes_without_void
        # if args.class_weighting != 'None':
        #     class_weighting = self.train_loader.dataset.compute_class_weights(weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting)
        # else:
        #     class_weighting = np.ones(self.train_loader.dataset.n_classes_without_void)
        # class_weighting = torch.tensor(class_weighting).float()
        # print(class_weighting.shape, class_weighting)

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        # self.criterion = nn.CrossEntropyLoss(weight=class_weighting, ignore_index=args.ignore_index).cuda()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index, reduction='mean').cuda()
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # self.confmat = ConfusionMatrix(num_classes=self.train_loader.dataset.n_classes, average=None)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            print("Using GPU: ", self.args.gpu_ids)
            print("Total GPU Available: ", torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_pred_2 = 0.0
        
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        scaler = torch.cuda.amp.GradScaler()
        self.evaluator.reset()
        # self.confmat.reset()
        for i, sample in enumerate(tbar):
            image, target, depth = sample['image'], sample['label'], sample['depth']

            if len(depth.shape) != 4:  # avoide automatic squeeze in later version of pytorch data loading
                depth = depth.unsqueeze(1)
                # print(depth.shape)

            if self.args.cuda:
                image, target, depth = image.cuda(), target.cuda(), depth.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            rgb = image if self.args.use_rgb else None
            depth = depth if self.args.use_depth else None
            
            with torch.cuda.amp.autocast():
                output = self.model(rgb=rgb, depth=depth)
                loss = self.criterion(output, target)
                # loss = self.criterion(output, target.long())
                # print(loss)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # -------------------- My Modification Start ----------------------
            pred = output.data.cpu().numpy()
            target_ = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_, pred)
            # self.confmat.update((output, target.long()))
            # -------------------- My Modification End ----------------------

            # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     self.summary.visualize_image(self.writer, self.args.dataset, image[0], target, output, global_step)

        # -------------------- My Modification Start ----------------------
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # mIoU = self.evaluator.Mean_Intersection_over_Union(ignore_index=args.ignore_index)
        # mIoU2 = self.confmat.miou(ignore_index=args.ignore_index)
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union(ignore_index=args.ignore_index)

        # print(">>>>>>> Old MIOU:", mIoU, "New MIOU:", mIoU2)

        log_data = {
            "Train Loss": train_loss,
            "Train mIoU": mIoU,
            "Train Pixel Acc": Acc,
            "Train Pixel Acc_class": Acc_class,
            "Train FWIoU": FWIoU,
        }
        # -------------------- My Modification End ----------------------

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image[0].data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % train_loss)

        # if self.args.no_val:
        #     # save checkpoint every epoch
        #     is_best = False
        #     self.saver.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.module.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, is_best, filename="checkpoint-latest-intermediate-pytorch-1.pth.tar")

        # -------------------- My Modification Start ----------------------
        return log_data
        # -------------------- My Modification End ----------------------


    def test(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        # self.confmat.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        output_all = None
        for i, sample in enumerate(tbar):
            image, target, depth = sample['image'], sample['label'], sample['depth']

            if len(depth.shape) != 4:  # avoide automatic squeeze in later version of pytorch data loading
                depth = depth.unsqueeze(1)
                # print(depth.shape)

            if self.args.cuda:
                image, target, depth = image.cuda(), target.cuda(), depth.cuda()

            rgb = image if self.args.use_rgb else None
            depth = depth if self.args.use_depth else None

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = self.model(rgb=rgb, depth=depth)
                    # loss = self.criterion(output, target.long())
                    loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target_ = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if output_all is None:
                output_all = output.cpu().clone()
            else:
                output_all = torch.cat((output_all,output.cpu().clone()),dim=0)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_, pred)
            # self.confmat.update((output, target.long()))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # mIoU = self.evaluator.Mean_Intersection_over_Union(ignore_index=args.ignore_index)
        # mIoU2 = self.confmat.miou(ignore_index=args.ignore_index)
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union(ignore_index=args.ignore_index)
        self.writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test/mIoU', mIoU, epoch)
        self.writer.add_scalar('test/Acc', Acc, epoch)
        self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        # print(">>>>>> Old MIOU:", mIoU, "New MIOU:", mIoU2)

        # -------------------- My Modification Start ----------------------
        log_data = {
            "Test Loss": test_loss,
            "Test mIoU": mIoU,
            "Test Pixel Acc": Acc,
            "Test Pixel Acc_class": Acc_class,
            "Test FWIoU": FWIoU,
        }
        # -------------------- My Modification End ----------------------

        global_step = epoch
        # self.summary.visualize_test_image(self.writer, self.args.dataset, image[0], target, output, global_step)
        
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image[0].data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred_2:
            is_best = True
            self.best_pred_2 = new_pred
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, filename=f"{args.model_name}_best_test.pth.tar")

        # -------------------- My Modification Start ----------------------
        return log_data
        # -------------------- My Modification End ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['nyudv2', 'pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo', 'multimodal_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--model-name', type=str, default='DeeplabV3Plus-Unnamed',
                        help='Modle name for wandb and checkpoint (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original','bce'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=float, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-aolp', action='store_true', default=False,
                        help='use aolp')
    parser.add_argument('--use-dolp', action='store_true', default=False,
                        help='use dolp')
    parser.add_argument('--use-nir', action='store_true', default=False,
                        help='use nir')
    parser.add_argument('--use-pol', action='store_true', default=False,
                        help='use pol')
    parser.add_argument('--use-rgb', action='store_true', default=False,
                        help='use rgb')
    parser.add_argument('--use-depth', action='store_true', default=False,
                        help='use depth')
    parser.add_argument('--use-segmap', action='store_true', default=False,
                        help='use segmap')
    parser.add_argument('--enable-se', action='store_true', default=False,
                        help='use se block on decoder')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')
    parser.add_argument('--is-multimodal', action='store_true', default=True,
                        help='use multihead architecture')
    parser.add_argument('--norm', type=str, default='avg',
                        help='avg, bn or bnr')
    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Path to dataset root.',)
    parser.add_argument('--modality', type=str, default='rgbd', choices=['rgbd', 'rgb', 'depth'])
    parser.add_argument('--raw-depth', action='store_true', default=False,
                        help='Whether to use the raw depth values instead of'
                        'the refined depth values')
    parser.add_argument('--aug-scale-min', default=1.0, type=float,
                        help='the minimum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--aug-scale-max', default=1.4, type=float,
                        help='the maximum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--height', type=int, default=480,
                        help='height of the training images. '
                        'Images will be resized to this height.')
    parser.add_argument('--width', type=int, default=640,
                        help='width of the training images. '
                        'Images will be resized to this width.')
    parser.add_argument('--class-weighting', type=str,
                        default='median_frequency',
                        choices=['median_frequency', 'logarithmic', 'None'],
                        help='which weighting mode to use for weighting the '
                        'classes of the unbalanced dataset'
                        'for the loss function during training.')
    parser.add_argument('--ignore-index', default=0, type=int, help='index to ignore during the evaluation (mIoU) of the experiment')

    # ------------------- Wandb -------------------
    args = parser.parse_args()
    args.valid_full_res = False
    args.batch_size_valid = args.batch_size
    args.c_for_logarithmic_weighting = 1.02
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    # if args.epochs is None:
    #     epoches = {
    #         'coco': 30,
    #         'cityscapes': 200,
    #         'pascal': 50,
    #         'kitti': 50,
    #         'kitti_advanced': 50
    #     }
    #     args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    # if args.lr is None:
    #     lrs = {
    #         'coco': 0.1,
    #         'cityscapes': 0.01,
    #         'pascal': 0.007,
    #         'kitti' : 0.01,
    #         'kitti_advanced' : 0.01
    #     }
    #     args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    # if args.checkname is None:
    #     args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # input('Check arguments! Press Enter...')
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    wandb.init(project="Material-Segmentation-MCubeS", entity="kaykobad", name=args.model_name)

    trainer = TrainerMultimodalRGBD(args)
    # if args.is_multimodal:
    #     print("USE Multimodal Model")
    #     trainer = TrainerMultimodal(args)
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        log_data = {
            "Epoch": epoch
        }
        log_data_2 = trainer.training(epoch)
        log_data.update(log_data_2)
        log_data_2 = trainer.test(epoch)
        log_data.update(log_data_2)
        wandb.log(log_data)

    trainer.writer.close()
    print(args)