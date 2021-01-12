class DataSet(object):
    def __init__(self, dataset_path, img_width=128, img_height=128, num_images=None, train=True,
                 shuffle=False, low=-1, high=1):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width

        self.images = None
        self.labels = None
        self.num_images = num_images
        self.indices = None

    def __getitem__(self, index):
        inds = self.indices[index] % self.num_images
        return (self.images[inds], self.labels[inds])

    def __len__(self):
        return self.num_images

