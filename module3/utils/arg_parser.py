import argparse


# def get_argparser():

#     parser = argparse.ArgumentParser("Efficient Low-Shot", add_help=False)
    
#     # Ganti --backbone menjadi --backbone_name
#     parser.add_argument('--backbone_name', type=str, default='swinT1k',
#                     choices=['swinT1k', 'swinT1K', 'swinB1K', 'swinB22K'],
#                     help='Backbone architecture')
    
#     # Tambahkan parameter baru untuk encoder_type
#     parser.add_argument('--encoder_type', type=str, default='hybrid',
#                     choices=['standard', 'deformable', 'hybrid'],
#                     help='Type of encoder to use')

#     # Tambahkan parameter untuk pretrained dan requires_grad
#     parser.add_argument('--pretrained', action='store_true',
#                     help='Use pretrained backbone')
#     parser.add_argument('--requires_grad', action='store_true',
#                     help='Allow backbone gradients')
#     parser.add_argument('--dilation', action='store_true',
#                     help='Use dilated convolutions')

#     parser.add_argument('--model_name', default='efficient', type=str)
#     parser.add_argument('--data_path', 
#                         default='/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/'
#                         , type=str)
#     parser.add_argument('--model_path', default='checkpoints', type=str)

#     parser.add_argument('--reduction', default=8, type=int)
#     parser.add_argument('--image_size', default=512, type=int)
#     parser.add_argument('--num_enc_layers', default=3, type=int)
#     parser.add_argument('--num_ope_iterative_steps', default=3, type=int)
#     parser.add_argument('--emb_dim', default=256, type=int)
#     parser.add_argument('--num_heads', default=8, type=int)
#     parser.add_argument('--kernel_dim', default=3, type=int)
#     parser.add_argument('--num_objects', default=3, type=int)
#     parser.add_argument('--epochs', default=50, type=int)
    
#     parser.add_argument('--lr', default=1e-4, type=float)
#     parser.add_argument('--backbone_lr', default=0, type=float)
#     parser.add_argument('--lr_drop', default=200, type=int)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#     parser.add_argument('--batch_size', default=4, type=int)
#     parser.add_argument('--dropout', default=0.1, type=float)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--max_grad_norm', default=0.1, type=float)
#     parser.add_argument('--aux_weight', default=0.3, type=float)
#     parser.add_argument('--tiling_p', default=0.5, type=float)

#     parser.add_argument('--swav_backbone', action='store_true')
#     parser.add_argument('--resume_training', action='store_true')
#     parser.add_argument('--zero_shot', action='store_true')
#     parser.add_argument('--pre_norm', action='store_true')

#     parser.add_argument('--exemplar_weight', default=0.1, type=float)
#     parser.add_argument('--num_iterations', default=3, type=int)

#     return parser

def get_argparser():
    parser = argparse.ArgumentParser("Efficient Low-Shot", add_help=False)
    
    # Basic params
    parser.add_argument('--model_name', default='efficient', type=str)
    parser.add_argument('--data_path', 
                        default='/home/renaldy_fredyan/PhDResearch/LOCA/Dataset/', type=str)
    parser.add_argument('--model_path', default='checkpoints', type=str)
    
    # Backbone & encoder params
    parser.add_argument('--backbone', type=str, default='swinT1K',
                    choices=['swinT1K', 'swinT22K', 'swinB1K', 'swinB22K'])
    parser.add_argument('--encoder_type', type=str, default='hybrid',
                    choices=['standard', 'deformable', 'hybrid'])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--requires_grad', type=bool, default=True)
    parser.add_argument('--dilation', type=bool, default=True)
    
    # Architecture params
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_ope_iterative_steps', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--num_iterations', default=3, type=int)
    
    # Training params
    parser.add_argument('--epochs', default=200, type=int)
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
    parser.add_argument('--exemplar_weight', default=0.1, type=float)

    # Flags that jarang diubah, tetap pakai store_true
    parser.add_argument('--swav_backbone', action='store_true')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')

    return parser