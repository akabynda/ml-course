from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
import cv2


class ImageMaskDataset(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_UNCHANGED)[..., ::-1]
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_UNCHANGED)[..., ::-1]

        image = cv2.resize(image[100:-100], (256, 256))
        mask = cv2.resize(mask[100:-100], (256, 256)) > 100

        image = Tensor(image.copy()).permute(2, 0, 1) / 255.
        mask = Tensor(mask.astype(int))[None, ...]

        return image, mask.float()
