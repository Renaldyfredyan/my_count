import argparse


def get_argparser():

    parser = argparse.ArgumentParser("Efficient", add_help=False)

    parser.add_argument('--model_name', default='efficient_swin_base', type=str)
    parser.add_argument(
        '--data_path',
        default='/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/',
        type=str
    )
    parser.add_argument(
        '--model_path',
        default='/home/renaldy_fredyan/PhDResearch/ELS/checkpoints/',
        type=str
    )
    parser.add_argument('--backbone', default='swin_base', type=str)
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_iefl_iterative_steps', default=3, type=int)
    parser.add_argument('--emb_dim', default=512, type=int)  # Ditingkatkan dari 256 ke 512 untuk Swin Base
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')

    return parser