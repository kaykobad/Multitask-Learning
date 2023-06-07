# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import cv2
import numpy as np

from .dataset_base import DatasetBase

def _get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


class NYUv2Base:
    SPLITS = ['train', 'test']
    SPLIT_FILELIST_FILENAMES = {SPLITS[0]: 'train.txt', SPLITS[1]: 'test.txt'}
    SPLIT_DIRS = {SPLITS[0]: 'train', SPLITS[1]: 'test'}

    # number of classes without void
    N_CLASSES = [894, 40, 13]

    DEPTH_DIR = 'depth'
    DEPTH_RAW_DIR = 'depth_raw'
    RGB_DIR = 'rgb'

    LABELS_DIR_FMT = 'labels_{:d}'
    LABELS_COLORED_DIR_FMT = 'labels_{:d}_colored'

    CLASS_NAMES_13 = ['void',
                      'bed', 'books', 'ceiling', 'chair', 'floor', 'furniture',
                      'objects', 'picture', 'sofa', 'table', 'tv', 'wall',
                      'window']
    CLASS_NAMES_40 = ['void',
                      'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                      'table', 'door', 'window', 'bookshelf', 'picture',
                      'counter', 'blinds', 'desk', 'shelves', 'curtain',
                      'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
                      'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag',
                      'otherstructure', 'otherfurniture', 'otherprop']
    CLASS_NAMES_894 = ['void',
                       'book', 'bottle', 'cabinet', 'ceiling', 'chair', 'cone',
                       'counter', 'dishwasher', 'faucet', 'fire extinguisher',
                       'floor', 'garbage bin', 'microwave',
                       'paper towel dispenser', 'paper', 'pot',
                       'refridgerator', 'stove burner', 'table', 'unknown',
                       'wall', 'bowl', 'magnet', 'sink', 'air vent', 'box',
                       'door knob', 'door', 'scissor', 'tape dispenser',
                       'telephone cord', 'telephone', 'track light',
                       'cork board', 'cup', 'desk', 'laptop', 'air duct',
                       'basket', 'camera', 'pipe', 'shelves', 'stacked chairs',
                       'styrofoam object', 'whiteboard', 'computer',
                       'keyboard', 'ladder', 'monitor', 'stand', 'bar',
                       'motion camera', 'projector screen', 'speaker', 'bag',
                       'clock', 'green screen', 'mantel', 'window', 'ball',
                       'hole puncher', 'light', 'manilla envelope', 'picture',
                       'mail shelf', 'printer', 'stapler', 'fax machine',
                       'folder', 'jar', 'magazine', 'ruler', 'cable modem',
                       'fan', 'file', 'hand sanitizer', 'paper rack', 'vase',
                       'air conditioner', 'blinds', 'flower', 'plant', 'sofa',
                       'stereo', 'books', 'exit sign', 'room divider',
                       'bookshelf', 'curtain', 'projector', 'modem', 'wire',
                       'water purifier', 'column', 'hooks', 'hanging hooks',
                       'pen', 'electrical outlet', 'doll', 'eraser',
                       'pencil holder', 'water carboy', 'mouse', 'cable rack',
                       'wire rack', 'flipboard', 'map', 'paper cutter', 'tape',
                       'thermostat', 'heater', 'circuit breaker box',
                       'paper towel', 'stamp', 'duster', 'poster case',
                       'whiteboard marker', 'ethernet jack', 'pillow',
                       'hair brush', 'makeup brush', 'mirror',
                       'shower curtain', 'toilet', 'toiletries bag',
                       'toothbrush holder', 'toothbrush', 'toothpaste',
                       'platter', 'rug', 'squeeze tube', 'shower cap', 'soap',
                       'towel rod', 'towel', 'bathtub', 'candle', 'tissue box',
                       'toilet paper', 'container', 'clothes',
                       'electric toothbrush', 'floor mat', 'lamp', 'drum',
                       'flower pot', 'banana', 'candlestick', 'shoe', 'stool',
                       'urn', 'earplugs', 'mailshelf', 'placemat',
                       'excercise ball', 'alarm clock', 'bed', 'night stand',
                       'deoderant', 'headphones', 'headboard',
                       'basketball hoop', 'foot rest', 'laundry basket',
                       'sock', 'football', 'mens suit', 'cable box', 'dresser',
                       'dvd player', 'shaver', 'television',
                       'contact lens solution bottle', 'drawer',
                       'remote control', 'cologne', 'stuffed animal',
                       'lint roller', 'tray', 'lock', 'purse', 'toy bottle',
                       'crate', 'vasoline', 'gift wrapping roll',
                       'wall decoration', 'hookah', 'radio', 'bicycle',
                       'pen box', 'mask', 'shorts', 'hat', 'hockey glove',
                       'hockey stick', 'vuvuzela', 'dvd', 'chessboard',
                       'suitcase', 'calculator', 'flashcard', 'staple remover',
                       'umbrella', 'bench', 'yoga mat', 'backpack', 'cd',
                       'sign', 'hangers', 'notebook', 'hanger',
                       'security camera', 'folders', 'clothing hanger',
                       'stairs', 'glass rack', 'saucer', 'tag', 'dolly',
                       'machine', 'trolly', 'shopping baskets', 'gate',
                       'bookrack', 'blackboard', 'coffee bag', 'coffee packet',
                       'hot water heater', 'muffins', 'napkin dispenser',
                       'plaque', 'plastic tub', 'plate', 'coffee machine',
                       'napkin holder', 'radiator', 'coffee grinder', 'oven',
                       'plant pot', 'scarf', 'spice rack', 'stove',
                       'tea kettle', 'napkin', 'bag of chips', 'bread',
                       'cutting board', 'dish brush', 'serving spoon',
                       'sponge', 'toaster', 'cooking pan', 'kitchen items',
                       'ladel', 'spatula', 'spice stand', 'trivet',
                       'knife rack', 'knife', 'baking dish', 'dish scrubber',
                       'drying rack', 'vessel', 'kichen towel', 'tin foil',
                       'kitchen utensil', 'utensil', 'blender', 'garbage bag',
                       'sink protector', 'box of ziplock bags', 'spice bottle',
                       'pitcher', 'pizza box', 'toaster oven', 'step stool',
                       'vegetable peeler', 'washing machine', 'can opener',
                       'can of food', 'paper towel holder', 'spoon stand',
                       'spoon', 'wooden kitchen utensils', 'bag of flour',
                       'fruit', 'sheet of metal', 'waffle maker', 'cake',
                       'cell phone', 'tv stand', 'tablecloth', 'wine glass',
                       'sculpture', 'wall stand', 'iphone', 'coke bottle',
                       'piano', 'wine rack', 'guitar', 'light switch',
                       'shirts in hanger', 'router', 'glass pot', 'cart',
                       'vacuum cleaner', 'bin', 'coins', 'hand sculpture',
                       'ipod', 'jersey', 'blanket', 'ironing board',
                       'pen stand', 'mens tie', 'glass baking dish',
                       'utensils', 'frying pan', 'shopping cart',
                       'plastic bowl', 'wooden container', 'onion', 'potato',
                       'jacket', 'dvds', 'surge protector', 'tumbler', 'broom',
                       'can', 'crock pot', 'person', 'salt shaker',
                       'wine bottle', 'apple', 'eye glasses', 'menorah',
                       'bicycle helmet', 'fire alarm', 'water fountain',
                       'humidifier', 'necklace', 'chandelier', 'barrel',
                       'chest', 'decanter', 'wooden utensils', 'globe',
                       'sheets', 'fork', 'napkin ring', 'gift wrapping',
                       'bed sheets', 'spot light', 'lighting track',
                       'cannister', 'coffee table', 'mortar and pestle',
                       'stack of plates', 'ottoman', 'server',
                       'salt container', 'utensil container', 'phone jack',
                       'switchbox', 'casserole dish', 'oven handle', 'whisk',
                       'dish cover', 'electric mixer', 'decorative platter',
                       'drawer handle', 'fireplace', 'stroller', 'bookend',
                       'table runner', 'typewriter', 'ashtray', 'key',
                       'suit jacket', 'range hood', 'cleaning wipes',
                       'six pack of beer', 'decorative plate', 'watch',
                       'balloon', 'ipad', 'coaster', 'whiteboard eraser',
                       'toy', 'toys basket', 'toy truck', 'classroom board',
                       'chart stand', 'picture of fish', 'plastic box',
                       'pencil', 'carton', 'walkie talkie', 'binder',
                       'coat hanger', 'filing shelves', 'plastic crate',
                       'plastic rack', 'plastic tray', 'flag', 'poster board',
                       'lunch bag', 'board', 'leg of a girl', 'file holder',
                       'chart', 'glass pane', 'cardboard tube', 'bassinet',
                       'toy car', 'toy shelf', 'toy bin', 'toys shelf',
                       'educational display', 'placard', 'soft toy group',
                       'soft toy', 'toy cube', 'toy cylinder', 'toy rectangle',
                       'toy triangle', 'bucket', 'chalkboard', 'game table',
                       'storage shelvesbooks', 'toy cuboid', 'toy tree',
                       'wooden toy', 'toy box', 'toy phone', 'toy sink',
                       'toyhouse', 'notecards', 'toy trucks',
                       'wall hand sanitizer dispenser', 'cap stand',
                       'music stereo', 'toys rack', 'display board',
                       'lid of jar', 'stacked bins  boxes',
                       'stacked plastic racks', 'storage rack',
                       'roll of paper towels', 'cables', 'power surge',
                       'cardboard sheet', 'banister', 'show piece',
                       'pepper shaker', 'kitchen island',
                       'excercise equipment', 'treadmill', 'ornamental plant',
                       'piano bench', 'sheet music', 'grandfather clock',
                       'iron grill', 'pen holder', 'toy doll', 'globe stand',
                       'telescope', 'magazine holder', 'file container',
                       'paper holder', 'flower box', 'pyramid', 'desk mat',
                       'cordless phone', 'desk drawer', 'envelope',
                       'window frame', 'id card', 'file stand', 'paper weight',
                       'toy plane', 'money', 'papers', 'comforter', 'crib',
                       'doll house', 'toy chair', 'toy sofa', 'plastic chair',
                       'toy house', 'child carrier', 'cloth bag', 'cradle',
                       'baby chair', 'chart roll', 'toys box', 'railing',
                       'clothing dryer', 'clothing washer',
                       'laundry detergent jug', 'clothing detergent',
                       'bottle of soap', 'box of paper', 'trolley',
                       'hand sanitizer dispenser', 'soap holder',
                       'water dispenser', 'photo', 'water cooler',
                       'foosball table', 'crayon', 'hoola hoop', 'horse toy',
                       'plastic toy container', 'pool table', 'game system',
                       'pool sticks', 'console system', 'video game',
                       'pool ball', 'trampoline', 'tricycle', 'wii',
                       'furniture', 'alarm', 'toy table', 'ornamental item',
                       'copper vessel', 'stick', 'car', 'mezuza',
                       'toy cash register', 'lid', 'paper bundle',
                       'business cards', 'clipboard', 'flatbed scanner',
                       'paper tray', 'mouse pad', 'display case',
                       'tree sculpture', 'basketball', 'fiberglass case',
                       'framed certificate', 'cordless telephone', 'shofar',
                       'trophy', 'cleaner', 'cloth drying stand',
                       'electric box', 'furnace', 'piece of wood',
                       'wooden pillar', 'drying stand', 'cane',
                       'clothing drying rack', 'iron box', 'excercise machine',
                       'sheet', 'rope', 'sticks', 'wooden planks',
                       'toilet plunger', 'bar of soap', 'toilet bowl brush',
                       'light bulb', 'drain', 'faucet handle', 'nailclipper',
                       'shaving cream', 'rolled carpet', 'clothing iron',
                       'window cover', 'charger and wire', 'quilt', 'mattress',
                       'hair dryer', 'stones', 'pepper grinder', 'cat cage',
                       'dish rack', 'curtain rod', 'calendar', 'head phones',
                       'cd disc', 'head phone', 'usb drive', 'water heater',
                       'pan', 'tuna cans', 'baby gate', 'spoon sets',
                       'cans of cat food', 'cat', 'flower basket',
                       'fruit platter', 'grapefruit', 'kiwi', 'hand blender',
                       'knobs', 'vessels', 'cell phone charger', 'wire basket',
                       'tub of tupperware', 'candelabra', 'litter box',
                       'shovel', 'cat bed', 'door way', 'belt',
                       'surge protect', 'glass', 'console controller',
                       'shoe rack', 'door frame', 'computer disk', 'briefcase',
                       'mail tray', 'file pad', 'letter stand',
                       'plastic cup of coffee', 'glass box', 'ping pong ball',
                       'ping pong racket', 'ping pong table', 'tennis racket',
                       'ping pong racquet', 'xbox', 'electric toothbrush base',
                       'toilet brush', 'toiletries', 'razor',
                       'bottle of contact lens solution', 'contact lens case',
                       'cream', 'glass container', 'container of skin cream',
                       'soap dish', 'scale', 'soap stand', 'cactus',
                       'door  window  reflection', 'ceramic frog',
                       'incense candle', 'storage space', 'door lock',
                       'toilet paper holder', 'tissue', 'personal care liquid',
                       'shower head', 'shower knob', 'knob', 'cream tube',
                       'perfume box', 'perfume', 'back scrubber',
                       'door facing trimreflection', 'doorreflection',
                       'light switchreflection', 'medicine tube', 'wallet',
                       'soap tray', 'door curtain', 'shower pipe',
                       'face wash cream', 'flashlight', 'shower base',
                       'window shelf', 'shower hose', 'toothpaste holder',
                       'soap box', 'incense holder', 'conch shell',
                       'roll of toilet paper', 'shower tube',
                       'bottle of listerine', 'bottle of hand wash liquid',
                       'tea pot', 'lazy susan', 'avocado', 'fruit stand',
                       'fruitplate', 'oil container', 'package of water',
                       'bottle of liquid', 'door way arch', 'jug', 'bulb',
                       'bagel', 'bag of bagels', 'banana peel', 'bag of oreo',
                       'flask', 'collander', 'brick', 'torch', 'dog bowl',
                       'wooden plank', 'eggs', 'grill', 'dog', 'chimney',
                       'dog cage', 'orange plastic cap', 'glass set',
                       'vessel set', 'mellon', 'aluminium foil', 'orange',
                       'peach', 'tea coaster', 'butterfly sculpture',
                       'corkscrew', 'heating tray', 'food processor', 'corn',
                       'squash', 'watermellon', 'vegetables', 'celery',
                       'glass dish', 'hot dogs', 'plastic dish', 'vegetable',
                       'sticker', 'chapstick', 'sifter', 'fruit basket',
                       'glove', 'measuring cup', 'water filter',
                       'wine accessory', 'dishes', 'file box',
                       'ornamental pot', 'dog toy', 'salt and pepper',
                       'electrical kettle', 'kitchen container plastic',
                       'pineapple', 'suger jar', 'steamer', 'charger',
                       'mug holder', 'orange juicer', 'juicer',
                       'bag of hot dog buns', 'hamburger bun', 'mug hanger',
                       'bottle of ketchup', 'toy kitchen',
                       'food wrapped on a tray', 'kitchen utensils',
                       'oven mitt', 'bottle of comet', 'wooden utensil',
                       'decorative dish', 'handle', 'label', 'flask set',
                       'cooking pot cover', 'tupperware', 'garlic',
                       'tissue roll', 'lemon', 'wine', 'decorative bottle',
                       'wire tray', 'tea cannister', 'clothing hamper',
                       'guitar case', 'wardrobe', 'boomerang', 'button',
                       'karate belts', 'medal', 'window seat', 'window box',
                       'necklace holder', 'beeper', 'webcam', 'fish tank',
                       'luggage', 'life jacket', 'shoelace', 'pen cup',
                       'eyeball plastic ball', 'toy pyramid', 'model boat',
                       'certificate', 'puppy toy', 'wire board', 'quill',
                       'canister', 'toy boat', 'antenna', 'bean bag',
                       'lint comb', 'travel bag', 'wall divider', 'toy chest',
                       'headband', 'luggage rack', 'bunk bed', 'lego',
                       'yarmulka', 'package of bedroom sheets',
                       'bedding package', 'comb', 'dollar bill', 'pig',
                       'storage bin', 'storage chest', 'slide', 'playpen',
                       'electronic drumset', 'ipod dock', 'microphone',
                       'music keyboard', 'music stand', 'microphone stand',
                       'album', 'kinect', 'inkwell', 'baseball',
                       'decorative bowl', 'book holder', 'toy horse', 'desser',
                       'toy apple', 'toy dog', 'scenary', 'drawer knob',
                       'shoe hanger', 'tent', 'figurine', 'soccer ball',
                       'hand weight', 'magic 8ball', 'bottle of perfume',
                       'sleeping bag', 'decoration item', 'envelopes',
                       'trinket', 'hand fan',
                       'sculpture of the chrysler building',
                       'sculpture of the eiffel tower',
                       'sculpture of the empire state building', 'jeans',
                       'garage door', 'case', 'rags', 'decorative item',
                       'toy stroller', 'shelf frame', 'cat house',
                       'can of beer', 'dog bed', 'lamp shade', 'bracelet',
                       'reflection of window shutters', 'decorative egg',
                       'indoor fountain', 'photo album', 'decorative candle',
                       'walkietalkie', 'serving dish', 'floor trim',
                       'mini display platform', 'american flag', 'vhs tapes',
                       'throw', 'newspapers', 'mantle',
                       'package of bottled water', 'serving platter',
                       'display platter', 'centerpiece', 'tea box',
                       'gold piece', 'wreathe', 'lectern', 'hammer',
                       'matchbox', 'pepper', 'yellow pepper', 'duck',
                       'eggplant', 'glass ware', 'sewing machine',
                       'rolled up rug', 'doily', 'coffee pot', 'torah']

    CLASS_COLORS_13 = [[0, 0, 0],
                       [0, 0, 255],
                       [232, 88, 47],
                       [0, 217, 0],
                       [148, 0, 240],
                       [222, 241, 23],
                       [255, 205, 205],
                       [0, 223, 228],
                       [106, 135, 204],
                       [116, 28, 41],
                       [240, 35, 235],
                       [0, 166, 156],
                       [249, 139, 0],
                       [225, 228, 194]]

    CLASS_COLORS_40 = _get_colormap(1+40).tolist()
    CLASS_COLORS_894 = _get_colormap(1+894).tolist()


class NYUv2(NYUv2Base, DatasetBase):
    def __init__(self,
                 data_dir='./datasets/nyudv2/',
                 n_classes=40,
                 split='train',
                 depth_mode='raw',
                 with_input_orig=False):
        super(NYUv2, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['kv1']

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # load filenames
            fp = os.path.join(self._data_dir,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        # load class names
        self._class_names = getattr(self, f'CLASS_NAMES_{self._n_classes}')

        # load class colors
        self._class_colors = np.array(
            getattr(self, f'CLASS_COLORS_{self._n_classes}'),
            dtype='uint8'
        )

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 2769.0187903686697, std: 1350.4174149841133
        self._depth_mean = 2841.94941272766
        self._depth_std = 1417.2594281672277

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, directory, filename):
        fp = os.path.join(self._data_dir,
                          self.split,
                          directory,
                          f'{filename}.png')
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im

    def load_image(self, idx):
        return self._load(self.RGB_DIR, self._filenames[idx])

    def load_depth(self, idx):
        if self._depth_mode == 'raw':
            return self._load(self.DEPTH_RAW_DIR, self._filenames[idx])
        else:
            return self._load(self.DEPTH_DIR, self._filenames[idx])

    def load_label(self, idx):
        return (self._load(self.LABELS_DIR_FMT.format(self._n_classes),
                          self._filenames[idx]) - 1)
        # return self._load(self.LABELS_DIR_FMT.format(self._n_classes),
        #                   self._filenames[idx])

    def __len__(self):
        return len(self._filenames)
