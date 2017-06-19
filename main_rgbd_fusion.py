import argparse
import os
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--score1', default='', type=str, )
parser.add_argument('--score2', default='', type=str, )
parser.add_argument('--output_dir', default='logs/test/pred', type=str, )

parser.add_argument('--fuse_p', default=0.5, type=float, )

parser.add_argument('--eval_split', default='val', type=str)

args = parser.parse_args()

def read_score(score_path):
    with open(score_path, 'r') as f:
        raw_scores = f.read().splitlines()
    scores = []
    for score in raw_scores:
        score = score.split('\t')
        score_list = score[1:]
        score_list = np.array([float(s) for s in score_list])

        scores.append([score[0]+'.avi', score_list])
    return scores
def main():
    score1 = read_score(args.score1)
    score2 = read_score(args.score2)
    if args.eval_split == 'val':
        with open('./dataset/info/val_rgb_info.json', 'r') as f:
            infos = json.load(f)

    pred_list = []
    tp = 0
    for i, (s1, s2) in enumerate(zip(score1, score2)):
        s1[0] = s1[0].replace('K', 'M')
        s2[0] = s2[0].replace('M', 'K')

        fuse_s = s1[1] * args.fuse_p + s2[1] * (1-args.fuse_p)
        pred_t = int(np.argmax(fuse_s))
        line = '{} {} {}'.format(s1[0], s2[0], pred_t+1)
        pred_list.append(line)
        if args.eval_split == 'val':
            if pred_t == infos[i]['label']:
                tp += 1

    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.eval_split == 'val':
        output_path = os.path.join(output_dir, 'valid_prediction.txt')
    else:
        output_path = os.path.join(output_dir, 'test_prediction.txt')

    with open(output_path, 'w') as f:
        f.write('\n'.join(pred_list))

    if args.eval_split == 'val':
        print('acc: {}'.format(tp*1.0/len(infos)))

if __name__ == '__main__':
    main()