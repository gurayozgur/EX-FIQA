import argparse
import os
import sys
sys.path.insert(0, 'EX-FIQA/')
sys.path.append("../backbones")
from QualityModel import QualityModel




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='../data/',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--output-dir', type=str, default='../outputs/',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--datasets', type=str, default='IJBC',
                        help='list of evaluation datasets (,)  e.g.  XQLFW,lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="../pretrained/",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="65144",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="token",
                        help='backbone network (token | crfiqa | full)')
    parser.add_argument('--score_file_name', type=str, default=".txt",
                        help='score file name, the file will be store in the same data dir')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')
    parser.add_argument('--early-exit-block', type=int, default=12,
                        help='Early exit block index (0-12, 12 for full model)')


    return parser.parse_args(argv)

def read_image_list(image_list_file, image_dir=''):
    image_lists = []
    with open(image_list_file) as f:
        absolute_list=f.readlines()
        for l in absolute_list:
            image_lists.append(os.path.join(image_dir, l.rstrip()))
    return image_lists, absolute_list

def main(param):
    datasets=param.datasets.split(',')
    face_model=QualityModel(param.model_path,param.model_id, param.gpu_id, param.backbone, early_exit_block=param.early_exit_block)
    for dataset in datasets:
        if param.early_exit_block >= 0:
            file_suffix = f"EE{param.early_exit_block}"
        else:
            file_suffix = "noEE"

        root=os.path.join(param.data_dir)
        image_list, absolute_list=read_image_list(os.path.join(param.data_dir,'quality_data',dataset,'image_path_list.txt'), root)
        _, quality=face_model.get_batch_feature(image_list,batch_size=16, color=param.color_channel)

        if not (os.path.isdir(os.path.join(param.output_dir,dataset))):
            os.makedirs(os.path.join(param.output_dir,dataset))

        output_file = os.path.join(param.output_dir,dataset,param.score_file_name.replace(".txt", f"{param.backbone}_{file_suffix}_{dataset}.txt"))

        with open(output_file, "w") as quality_score:
            for i in range(len(quality)):
                quality_score.write(absolute_list[i].rstrip()+ " "+f"{quality[i][0]:.10f}"+ "\n")

        print(f"Saved quality scores to {output_file} (early_exit_block: {param.early_exit_block})")

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
