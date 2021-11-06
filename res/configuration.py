class Config:
    def __init__(self):
        # Learning rate
        self.lr = 1e-5

        # Epochs
        self.epoch = 5
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Basic
        self.device = 'cuda'
        self.num_classes = 1
        self.seed = 42
        self.batch_size = 64
        self.num_workers = 2
        self.split_rate = 0.8
        self.checkpoint = '../models/checkpoint.pth'
        self.clip_max_norm = 0.1
        self.input_shape = (224, 224)
        self.encoder = None
        # Dataset
        self.dir = '../dataset/'
