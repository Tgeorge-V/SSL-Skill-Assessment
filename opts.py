import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/data/tanqiaozhi/SSTAP-gai1/output/123/5out/')
    parser.add_argument(
        '--category',
        type=str,
        default='kt')      
    parser.add_argument(  
        '--use_semi',
        type=bool,
        default=True)
    parser.add_argument(  
        '--use_warmup',
        type=bool,
        default=True)
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.00005)#0.00005/0.05
    parser.add_argument(
        '--training_lr_finetune',
        type=float,
        default=0.00001)#0.00001/0.001
    parser.add_argument(
        '--unlabel_percent',
        type=float,
        default=0.5)   # 0.5
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-3)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=50)    #50
    parser.add_argument(
        '--warm_up_epochs',
        type=int,
        default=30)    #30
    parser.add_argument(
        '--label_batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--unlabel_batch_size',
        type=int,
        default=8)    
    parser.add_argument(
        '--step_size',
        type=int,
        default=10)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.5)
    parser.add_argument(
        '--step_size_finetune',
        type=int,
        default=10)
    parser.add_argument(
        '--step_gamma_finetune',
        type=float,
        default=0.8)
    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="/data/tanqiaozhi/SSTAP-gai1/data/jigsaws_annotations/splits/kt/loso/5out/video_info_new_0.5.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="/data/tanqiaozhi/SSTAP-gai1/data/jigsaws_annotations/splits/kt/loso/5out/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/data/tanqiaozhi/SSTAP-gai1/features/kt_feature/")
    parser.add_argument(
        '--aug_feature_path',
        type=str,
        default="/data/tanqiaozhi/SPOT/features/aug_feature/")
    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="/home/ren2/data/Long/SSL_Skill/SSTAP-main/output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="/home/ren2/data/Long/SSL_Skill/SSTAP-main/output/evaluation_result.jpg")

    args = parser.parse_args()

    return args

