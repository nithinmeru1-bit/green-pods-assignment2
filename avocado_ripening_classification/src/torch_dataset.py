from PIL import Image
from torch.utils.data import Dataset


class OrangeImageDataset(Dataset):
    def __init__(self, samples, class_names, transform=None):
        self.samples = list(samples)
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_index[label]
