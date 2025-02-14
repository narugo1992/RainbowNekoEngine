import argparse

from accelerate import Accelerator

from .trainer_ac import Trainer, load_config_with_cli, set_seed


class TrainerSingleCard(Trainer):
    def init_context(self, cfgs_raw):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = 0
        print('num process', self.accelerator.num_processes)
        self.world_size = 1

        set_seed(self.cfgs.seed + self.local_rank)

    @property
    def model_raw(self):
        return self.model_wrapper

def neko_train():
    import subprocess
    import sys
    subprocess.run(["accelerate", "launch", "-m", "rainbowneko.train.trainer.trainer_ac_single"] + sys.argv[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RainbowNeko Trainer for one GPU')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(parser, conf)
    trainer.train()
