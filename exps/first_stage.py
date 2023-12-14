import os
import json

import torch

from tools.trainer import first_stage_train
from tools.dataloader import get_loaders
from models.autoencoder.autoencoder_vit import ViTAutoencoder 
from losses.perceptual import LPIPSWithDiscriminator

from utils import file_name, Logger

def first_stage(rank, args):
    device = torch.device(args.device)

    """ ROOT DIRECTORY """
    fn = file_name(args)
    logger = Logger(fn)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')
    rootdir = logger.logdir
    log_ = logger.log
    log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, test_loader, total_vid = get_loaders(rank, args.data, args.res, args.timesteps, args.skip, args.batch_size, args.n_gpus, args.seed, cond=False, data_location=args.data_path)
    log_(f"Generating model")

    model = ViTAutoencoder(args.embed_dim, args.ddconfig)

    if args.checkpoint:
        model_ckpt = torch.load(args.checkpoint)
        #xy_pos_embedding
        #coords
        #encoder.to_patch_embedding.weight
        #to_pixel.1.weight
        #to_pixel.1.bias
        del model_ckpt["coords"]
        del model_ckpt["xy_pos_embedding"]
        del model_ckpt["encoder.to_patch_embedding.weight"]
        del model_ckpt["to_pixel.1.weight"]
        del model_ckpt["to_pixel.1.bias"]
        print("Partial loading")
        model.load_state_dict(model_ckpt, strict=False)
        del model_ckpt
    model = model.to(device)

    criterion = LPIPSWithDiscriminator(disc_start   = args.lossconfig.params.disc_start,
                                       timesteps    = args.ddconfig.timesteps).to(device)


    opt = torch.optim.AdamW(model.parameters(), 
                             lr=args.lr, 
                             betas=(0.5, 0.9)
                             )

    d_opt = torch.optim.AdamW(list(criterion.discriminator_2d.parameters()) + list(criterion.discriminator_3d.parameters()), 
                             lr=args.lr, 
                             betas=(0.5, 0.9))

    if args.resume:
        model_ckpt = torch.load(os.path.join(args.first_stage_folder, 'model_last.pth'))
        model.load_state_dict(model_ckpt)
        opt_ckpt = torch.load(os.path.join(args.first_stage_folder, 'opt.pth'))
        opt.load_state_dict(opt_ckpt)

        del model_ckpt
        del opt_ckpt

    torch.save(model.state_dict(), rootdir + f'net_init.pth')

    fp = args.amp
    first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, args.first_model, fp, logger)

    torch.save(model.state_dict(), rootdir + f'net_meta.pth')
