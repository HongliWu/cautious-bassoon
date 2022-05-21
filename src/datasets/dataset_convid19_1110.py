import os
import SimpleITK as sitk
import cv2
from  matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import csv
from torch.utils.data import Dataset
from utils.utils import TransformCfg, timeit_context
from imgaug import augmenters as iaa



class Dataset_Convid19_CT(Dataset):
    def __init__(self, fold: int, is_training: bool, debug: bool, img_size: int, ct_dir, output_dir, augmentation_level=10, crop_source=512):
        """
        Args:
            fold              : integer, number of the fold
            is_training       : if True, runs the training mode, else runs evaluation mode
            debug             : if True, runs the debugging on few images
            img_size          : the desired image size to resize to        
            augmentation_level: level of augmentations from the set        
        """
        # super(dataset_convid19_CT, self).__init__()  # inherit it from torch Dataset
        self.ct_dir = ct_dir
        self.output_dir = output_dir
        self.npy_dir = os.path.join(self.output_dir, 'numpy')
        self.jpg_dir = os.path.join(self.output_dir, 'jpg')
        self.label_path = os.path.join(self.output_dir, 'label.csv')
        
        self.fold = fold
        self.is_training = is_training
        self.img_size = img_size
        self.debug = debug
        self.crop_source = crop_source
        self.augmentation_level = augmentation_level
        self.categories = ["Normal", "Covid19"]
        samples = pd.read_csv(self.label_path)
        # samples = samples.merge(pd.read_csv(os.path.join(DATA_DIR, "folds.csv")), on="patientId", how="left")

        if self.debug:
            samples = samples.head(32)
            print("Debug mode, samples: ", samples)
        if is_training:
            self.samples = samples[samples.fold != fold]
        else:
            self.samples = samples[samples.fold == fold]

        self.patient_ids = list(sorted(self.samples.patientId.unique()))
        self.patient_categories = {}
        self.annotations = defaultdict(list)
        # add annotation points for rotation
        for _, row in self.samples.iterrows():
            patient_id = row["patientId"]
            # self.patient_categories[patient_id] = self.categories.index(row["class"])
            self.patient_categories[patient_id] = "COVID19"
            if row["Target"] > 0:
                x, y, w, h = row.x, row.y, row.width, row.height
                points = np.array(
                    [
                        [x, y + h / 3],
                        [x, y + h * 2 / 3],
                        [x + w, y + h / 3],
                        [x + w, y + h * 2 / 3],
                        [x + w / 3, y],
                        [x + w * 2 / 3, y],
                        [x + w / 3, y + h],
                        [x + w * 2 / 3, y + h],
                    ]
                )
                self.annotations[patient_id].append(points)

    def CT2Jpg(self):
        for i in range(1, 51):
            nii_name = f"COVID_{i:0>3}_0000.nii.gz"
            nii_path = os.path.join(self.ct_dir, nii_name)
            print(f"read: {nii_path}")
            nii_file = sitk.ReadImage(nii_path)# 读取要转换格式的图像
            nii_data = sitk.GetArrayFromImage(nii_file)
            lower_bound = np.percentile(nii_data, 0.5)
            upper_bound = np.percentile(nii_data, 99.5)
            nii_data = np.clip(nii_data, lower_bound, upper_bound)
            nii_data = (nii_data - lower_bound) / (upper_bound - lower_bound) * 255
            nii_data = nii_data.round().astype('uint8')
            for j in range(nii_data.shape[0]):
                npy_name = f"COVID_{i:0>3}_{j+1:0>4}.npy"
                jpg_name = f"COVID_{i:0>3}_{j+1:0>4}.jpg"
                npy_path = os.path.join(self.npy_dir, npy_name)
                jpg_path = os.path.join(self.jpg_dir, jpg_name)
                np.save(npy_path, nii_data[j, :, :])
                cv2.imwrite(jpg_path, nii_data[j, :, :])
                print(f"saved: {npy_path}")
    
    def create_empty_label(self):
        slice_list = os.listdir(self.npy_dir)
        header_row = ['patientId','x','y','width','height','Target','class','fold']
        with open(self.label_path, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header_row)
            for row in slice_list:
                if row[6:9] in ['002', '030', '035', '037', '038', '042', '047', '048', '049', '050']:
                    fold_idx = 1
                else:
                    fold_idx = 0
                if row[-4:] == '.npy':
                    f_csv.writerow([row[:-4], '', '', '', '', '', '', fold_idx])
    
    def get_image(self, patient_id):
        """Load a dicom image to an array"""
        try:
            npy_path = os.path.join(self.npy_dir, f"{patient_id}.npy")
            img = np.load(npy_path)
            # dcm_data = pydicom.read_file(os.path.join(TRAIN_DIR, f"{patient_id}.dcm"))
            # img = dcm_data.pixel_array
            return img
        except:
            print(f"Read failure: {patient_id}")
            pass

    def num_classes(self):
        return 2

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        img = self.get_image(patient_id)

        if self.crop_source != 512:
            img_source_w = self.crop_source
            img_source_h = self.crop_source
        else:
            img_source_h, img_source_w = img.shape[:2]
        img_h, img_w = img.shape[:2]

        # set augmentation levels
        augmentation_sigma = {
            1: dict(scale=0, angle=0, shear=0, gamma=0, hflip=False),
            10: dict(scale=0.1, angle=5.0, shear=2.5, gamma=0.2, hflip=False),
            11: dict(scale=0.1, angle=0.0, shear=2.5, gamma=0.2, hflip=False),
            15: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.2, hflip=np.random.choice([True, False])),
            20: dict(scale=0.15, angle=6.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
            21: dict(scale=0.15, angle=0.0, shear=4.0, gamma=0.25, hflip=np.random.choice([True, False])),
        }[self.augmentation_level]
        # training mode augments
        if self.is_training:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2 + np.random.uniform(-32, 32),
                src_center_y=img_h / 2 + np.random.uniform(-32, 32),
                scale_x=self.img_size / img_source_w * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                scale_y=self.img_size / img_source_h * (2 ** np.random.normal(0, augmentation_sigma["scale"])),
                angle=np.random.normal(0, augmentation_sigma["angle"]),
                shear=np.random.normal(0, augmentation_sigma["shear"]),
                hflip=augmentation_sigma["hflip"],
                vflip=False,
            )
        # validation mode augments
        else:
            cfg = TransformCfg(
                crop_size=self.img_size,
                src_center_x=img_w / 2,
                src_center_y=img_h / 2,
                scale_x=self.img_size / img_source_w,
                scale_y=self.img_size / img_source_h,
                angle=0,
                shear=0,
                hflip=False,
                vflip=False,
            )
        # add more augs in training modes
        crop = cfg.transform_image(img)
        if self.is_training:
            crop = np.power(crop, 2.0 ** np.random.normal(0, augmentation_sigma["gamma"]))
            if self.augmentation_level == 20 or self.augmentation_level == 21:
                aug = iaa.Sequential(
                    [
                        iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),
                        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
                    ]
                )
                crop = (
                    aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:, :, 0].astype(np.float32)
                    / 255.0
                )
            if self.augmentation_level == 15:
                aug = iaa.Sequential(
                    [iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 1.0))), iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)))]
                )
                crop = (
                    aug.augment_image(np.clip(np.stack([crop, crop, crop], axis=2) * 255, 0, 255).astype(np.uint8))[:, :, 0].astype(np.float32)
                    / 255.0
                )
        # add annotation points
        annotations = []
        for annotation in self.annotations[patient_id]:
            points = cfg.transform().inverse(annotation)
            res = np.zeros((1, 5))
            p0 = np.min(points, axis=0)
            p1 = np.max(points, axis=0)
            res[0, 0:2] = p0
            res[0, 2:4] = p1
            res[0, 4] = 0
            annotations.append(res)
        if len(annotations):
            annotations = np.row_stack(annotations)
        else:
            annotations = np.zeros((0, 5))
        # print('patient_id', patient_id)
        sample = {"img": crop, "annot": annotations, "scale": 1.0, "category": self.patient_categories[patient_id]}
        return sample


def test_dataset_sample(sample_num, ct_dir, output_dir):
    """Test dataset on a single sample
    Args:
        sample_num: sample number from the dataset
    """
    dataset = Dataset_Convid19_CT(fold=0, is_training=False, debug=False, img_size=512, ct_dir=ct_dir, output_dir=output_dir)
    # print and plot sample
    print("dataset sample: \n", dataset[sample_num])
    plt.figure()
    plt.imshow(dataset[sample_num]["img"])
    annot = dataset[sample_num]["annot"]
    print("annotations: \n", annot)
    plt.show()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    output_dir = r"G:\03_research_file\210322NR-Dice\Task101_COVID19-1110\2d_slice"
    input_dir = r"G:\03_research_file\210322NR-Dice\Task101_COVID19-1110\imagesTr"
    # dataset_ct = dataset_convid19_CT(fold=0, is_training=False, debug=False, img_size=512, ct_dir=input_dir, output_dir=output_dir)
    # dataset_ct.create_empty_label()
    test_dataset_sample(sample_num=12, ct_dir=input_dir, output_dir=output_dir)
    

