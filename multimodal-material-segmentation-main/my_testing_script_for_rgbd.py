import argparse
import os
import wandb
import numpy as np
from tqdm import tqdm
import random
from mypath import Path
from dataloaders.rgbd.prepare_data import prepare_data
from torch.utils.data import DataLoader
from dataloaders.datasets import multimodal_dataset_2
from dataloaders import make_data_loader, make_data_loader2
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.my_deeplab_impl_3 import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import cv2

     
class TesterMultimodal(object):
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
        self.train_loader, self.test_loader = prepare_data(args, ckpt_dir=None, with_input_orig=True)
        self.nclass = self.train_loader.dataset.n_classes_without_void

        model_path = args.pth_path
        checkpoint = torch.load(model_path)
        
        # self.model = DeepLab(num_classes=20,
        #             backbone=args.backbone,
        #             output_stride=args.out_stride,
        #             sync_bn=args.sync_bn,
        #             freeze_bn=args.freeze_bn)

        # self.model = DeepFuseLab(num_classes=20,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_nir=args.use_nir,
        #                 use_aolp=args.use_aolp,
        #                 use_dolp=args.use_dolp,
        #                 use_segmap=args.use_segmap)

        # self.model = MMDeepLab(num_classes=20,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_nir=args.use_nir,
        #                 use_aolp=args.use_aolp,
        #                 use_dolp=args.use_dolp,
        #                 use_segmap=args.use_segmap,
        #                 enable_se=args.enable_se)

        # self.model = MMDeepLabSEMask(num_classes=20,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn,
        #                 use_nir=True,
        #                 use_aolp=args.use_aolp,
        #                 use_dolp=args.use_dolp,
        #                 use_pol=True,
        #                 use_segmap=args.use_segmap,
        #                 enable_se=args.enable_se)

        self.model = MMDeepLabSEMaskWithNormForRGBD(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        use_rgb=args.use_rgb,
                        use_depth=args.use_depth,
                        norm=args.norm)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(pytorch_total_params)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=0).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        
        # # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

    def test(self, epoch=0):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        image_all = None
        target_all = None
        output_all = None
        mIoUs = []
        FWIoUs = []
        for i, sample in enumerate(tbar):
            # if i==300:
            #     break
            image, target, depth = sample['image'], sample['label'], sample['depth']
            image_orig, target_orig, depth_orig = sample['image_orig'], sample['label_orig'], sample['depth_orig']
            if self.args.cuda:
                image, target, depth = image.cuda(), target.cuda(), depth.cuda()

            if len(depth.shape) != 4:  # avoide automatic squeeze in later version of pytorch data loading
                depth = depth.unsqueeze(1)
                # print(depth.shape)

            rgb = image if self.args.use_rgb else None
            depth = depth if self.args.use_depth else None

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = self.model(rgb=rgb, depth=depth)
                    loss = self.criterion(output, target)

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            target_ = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if image_all is None:
                image_all  = image.cpu().clone()
                target_all = target.cpu().clone()
                output_all = output.cpu().clone()
            else:
                image_all  = torch.cat(( image_all, image.cpu().clone()),dim=0)
                target_all = torch.cat((target_all, target.cpu().clone()),dim=0)
                output_all = torch.cat((output_all, output.cpu().clone()),dim=0)
                
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_, pred)

            # Uncomment if you want to save prediction
            # My evaluator
            ev = Evaluator(self.nclass)
            ev.add_batch(target_, pred)
            this_mIoU = ev.Mean_Intersection_over_Union()
            this_FWIoU = ev.Frequency_Weighted_Intersection_over_Union()
            print(f"Image {i} IoU: {this_mIoU} and FWIoU: {this_FWIoU}")
            mIoUs.append(this_mIoU)
            FWIoUs.append(this_FWIoU)

            # # Save the images
            # # print(f"Output shape: {output.shape}, Target Shape: {target.shape}")
            # img = image_orig.cpu().numpy()[0]
            # # print(img.shape)
            # # img = img.reshape(1024, 1024, 3)
            # t = target.cpu().numpy()
            # t = t.reshape(1024, 1024, 1)
            p = pred.reshape(480, 640, 1)
            # print(f"Image reShape: {img.shape}, Output reshape: {p.shape}, Target reShape: {t.shape}")
            # matplotlib.image.imsave(f'predictions/{i}-traget.png', t)
            # matplotlib.image.imsave(f'predictions/{i}-prediction.png', o)
            # cv2.imwrite(f'predictions/{i}-image.png', img)
            # cv2.imwrite(f'predictions/{i}-target.png', t)
            cv2.imwrite(f'predictions/nyudv2/test/{i}-prediction.png', p)
            # cv2.imwrite(f'predictions/nyudv2/test/{i}-rgb.png', image_orig.numpy())
            # cv2.imwrite(f'predictions/nyudv2/test/{i}-depth.png', depth_orig.numpy())
            # cv2.imwrite(f'predictions/nyudv2/test/{i}-target.png', target_orig.numpy())
            # cv2.imwrite(f'predictions/{i}-image.png', img)

            # out = output.data.cpu().numpy()[0]
            # # print(f"Out shape: {out.shape}, Type: {type(out)}")
            # for j in range(out.shape[0]):
            #     filter = out[j, :, :].astype(int)
            #     # print(f"Filter shape: {filter.shape}")
            #     filter = filter.reshape(1024, 1024, 1)
            #     cv2.imwrite(f'predictions/{i}-Filter-{j}.png', filter)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        confusion_matrix = self.evaluator.confusion_matrix
        # np.save(f'{os.path.dirname(args.pth_path)}/test/confusion_matrix.npy',confusion_matrix)
        np.save('nyud_test_confusion_matrix.npy', confusion_matrix)
        print("Confussion Matrix:", confusion_matrix)

        self.writer.add_scalar('test/mIoU', mIoU, epoch)
        self.writer.add_scalar('test/Acc', Acc, epoch)
        self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        # self.summary.visualize_test_image(self.writer, self.args.dataset, image_all, target_all, output_all, 0)
        
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print("mIoUs:", mIoUs)
        print("FWIoUs:", FWIoUs)
        print(np.argsort(np.array(mIoUs)))

        # print(f"Output shape: {output.shape}, Target Shape: {target.shape}")
        # matplotlib.image.imsave(f'predictions/{i}-traget.png', target.cpu().numpy())
        # matplotlib.image.imsave(f'predictions/{i}-prediction.png', output.data.cpu().numpy())
        #print('Loss: %.3f' % test_loss)



# def test_visualizer(args):
#     # Load the model
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model_path = "run/multimodal_dataset/MCubeSNet/experiment_10/checkpoint-latest-pytorch.pth.tar"
#     model = torch.load(model_path)
#     model.to(device)

#     # Prepare dataloader

#     # Take five images

#     # Fed to model

#     # Save the outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--use-rgb', action='store_true', default=False,
                        help='use rgb')
    parser.add_argument('--use-depth', action='store_true', default=False,
                        help='use depth')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='multimodal_dataset',
                        choices=['nyudv2', 'pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo', 'multimodal_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=float, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
    parser.add_argument('--use-segmap', action='store_true', default=False,
                        help='use segmap')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')
    parser.add_argument('--is-multimodal', action='store_true', default=False,
                        help='use multihead architecture')
    parser.add_argument('--enable-se', action='store_true', default=False,
                        help='use se block on decoder')
    parser.add_argument('--pth-path', type=str, default=None,
                        help='set the pth file path')
    parser.add_argument('--norm', type=str, default='avg',
                        help='avg, bn or bnr')
    parser.add_argument('--aug-scale-min', default=1.0, type=float,
                        help='the minimum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--aug-scale-max', default=1.4, type=float,
                        help='the maximum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--raw-depth', action='store_true', default=False,
                        help='Whether to use the raw depth values instead of'
                        'the refined depth values')

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

    # # default settings for epochs, batch_size and lr
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

    tester = TesterMultimodal(args)
    # exit

    # if args.is_multimodal:
    #     print("USE Multimodal Model")
    #     tester = TesterMultimodal(args)
    # else:
    #     tester = TesterAdv(args)
    # print('Starting Epoch:', tester.args.start_epoch)
    # print('Total Epoches:', tester.args.epochs)
    tester.test()
    tester.writer.close()
    print(args)