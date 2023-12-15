import torch

import utils
import data_loader
from utils import set_seed

from trainer import Trainer
from config import get_config
import numpy as np


def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    if config.random_seed == False:
        config.random_seed = np.random.randint(np.iinfo(np.int32).max // 2)

    set_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.dataset,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    else:
        dloader = data_loader.get_test_loader(
            config.data_dir, config.batch_size, config.dataset, **kwargs,
        )

    glimpses_list = config.glimpse_list

    for glimpse in glimpses_list:

        config.num_glimpses = glimpse
        trainer = Trainer(config, dloader)

        if config.is_train:
            utils.save_config(config)
            trainer.train()
            trainer.test()
        else:
            trainer.valid_acc_list = [0]
            trainer.test()
            print('============================================')

    # trainer = Trainer(config, dloader)
    # print(f'Memory type: {trainer.model.rnn.memory_type}')

    # # either train
    # if config.is_train:
    #     utils.save_config(config)
    #     trainer.train()
    # # or load a pretrained model and test
    # else:
    #     trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
