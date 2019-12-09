import os
import torch
import numpy as np

from PIL import Image
from torch.utils import data

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class MapillaryLoader(data.Dataset):

    class_names = [
        "Bird",
        "Ground Animal",
        "Curb",
        "Fence",
        "Guard Rail",
        "Barrier",
        "Wall",
        "Bike Lane",
        "Crosswalk - Plain",
        "Curb Cut",
        "Parking",
        "Pedestrian Area",
        "Rail Track",
        "Road",
        "Service Lane",
        "Sidewalk",
        "Bridge",
        "Building",
        "Tunnel",
        "Person",
        "Bicyclist",
        "Motorcyclist",
        "Other Rider",
        "Lane Marking - Crosswalk",
        "Lane Marking - General",
        "Mountain",
        "Sand",
        "Sky",
        "Snow",
        "Terrain",
        "Vegetation",
        "Water",
        "Banner",
        "Bench",
        "Bike Rack",
        "Billboard",
        "Catch Basin",
        "CCTV Camera",
        "Fire Hydrant",
        "Junction Box",
        "Mailbox",
        "Manhole",
        "Phone Booth",
        "Pothole",
        "Street Light",
        "Pole",
        "Traffic Sign Frame",
        "Utility Pole",
        "Traffic Light",
        "Traffic Sign (Back)",
        "Traffic Sign (Front)",
        "Trash Can",
        "Bicycle",
        "Boat",
        "Bus",
        "Car",
        "Caravan",
        "Motorcycle",
        "On Rails",
        "Other Vehicle",
        "Trailer",
        "Truck",
        "Wheeled Slow",
        "Car Mount",
        "Ego Vehicle",
        "Unlabeled",
    ]

    colors = [
        [128, 128, 128],
        [244, 244, 244],
        [70, 70, 70],
        [102, 102, 102],
        [153, 153, 153],
    ]

    label_colours = dict(zip(range(5), colors))

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=(1024, 2048),
        test_mode=False,
    ):

        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 5
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, "images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        self.valid_classes = [
            2, 24,
            13, 14, 43,
            8, 11, 15, 23,
            19, 20, 21, 22,
            52, 54, 55, 56, 57, 58, 59, 60, 61
        ]

        self.void_classes = [i for i in range(1, 66) if i not in self.valid_classes]

        self.ignore_index = 250

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")
        )
        name = img_path.split(os.sep)[-1][:-4] + ".png"

        img = Image.open(img_path)
        img = np.array(img)

        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, name

    def transform(self, img, lbl):
        img = np.array(Image.fromarray(img).resize(
                (self.img_size[1], self.img_size[0])))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = np.array(img).astype(np.float64) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW

        lbl = lbl.astype(float)
        lbl = np.array(Image.fromarray(lbl).resize(
                (self.img_size[1], self.img_size[0]), resample=Image.NEAREST))
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def mapToMainClass(self, mapillaryId):
        if mapillaryId in [2, 24]:
            return 0
        elif mapillaryId in [13, 14, 43]:
            return 1
        elif mapillaryId in [8, 11, 15, 23]:
            return 2
        elif mapillaryId in [19, 20, 21, 22]:
            return 3
        elif mapillaryId in [52, 54, 55, 56, 57, 58, 59, 60, 61]:
            return 4
        else:
            return self.ignore_index

    def decode_segmap_id(self, temp):
        ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
        for l in range(0, self.n_classes):
            ids[temp == l] = l
        return ids

    def encode_segmap(self, mask):
        for _voidc  in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.mapToMainClass(_validc)
        return mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    local_path = "/home/tomasz/Downloads/mapillary-vistas-dataset_public_v1.1/"
    dst = MapillaryLoader(local_path, is_transform=True)
    bs = 4

    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels, _ = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
