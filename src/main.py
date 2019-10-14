import torch
import os
import utility
import data
import model
import loss
from option import args, setup_args
from trainer import Trainer

torch.manual_seed(args.seed)


def main(myargs=None):
    setup_args(args)
    checkpoint = utility.checkpoint(args, outdir=myargs.args.outdir)
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()


def run(args1, myargs):
    myargs.config = getattr(myargs.config, args1.command)
    myargs.args = args1
    for k, v in myargs.config.items():
        setattr(args, k, v)
    args.dir_data = os.path.expanduser(args.dir_data)
    main(myargs)
    pass


if __name__ == '__main__':
    main()
