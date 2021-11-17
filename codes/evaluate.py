import os
import shutil
import argparse
from PIL import Image

from tqdm import tqdm
import numpy as np

from DFTR.utils import *



METRIC_DIC = {
              'MAE': cal_mae,
              'SM': cal_sm,
              'EM': cal_em,
              'FM': cal_fm,
              'FMAX': cal_fmax,
              'FMEAN': cal_fmean,
              'DICE': cal_dice,
              'IOU': cal_iou,
              'MSE': cal_mse,
              'RMSE': cal_rmse,
              'logMSE': cal_logmse,
              'logRMSE': cal_logrmse,
              'log10MAE': cal_log10mae,
              'MRE': cal_mre,
             }


def normalize(arr, maxvalue=255,binary=True):
    if arr.max()!=arr.min():
        arr = (arr-arr.min())/(arr.max()-arr.min())
    if binary:
        arr[arr < 0.5] = 0
        arr[arr >= 0.5] = 1
    arr*=maxvalue
    return arr.astype(np.uint8)


def saveImage(path, *arrs, binary=True):
    arr_list = []
    for arr in arrs:
        arr_list.append(normalize(arr, binary=binary))
    arr_cat = np.concatenate(arr_list, axis=1)
    Image.fromarray(arr_cat).convert('L').save(path)


def evaluate(dataset_name, input_path, output_path, gt_pre, save_img=False, flag='mask', metrics=['DICE']):
    print('\n' + '-' * 20 + dataset_name +  '-' * 20)
    print('gt_pre: ', gt_pre)
    print('input : ', input_path)
    print('output: ', output_path)

    savepath_dic = {}
    for s in ['RGB', 'gt', 'pred','gt_pred']:
        if s=='RGB':
            savepath_dic[s] = os.path.join(output_path, s)
        else:
            savepath_dic[s] = os.path.join(output_path, s+'_'+flag)
        mkdir(savepath_dic[s])

    gt_flag = 'GT' if flag == 'mask' else 'depth'
    gt_path = os.path.join(gt_pre, dataset_name, gt_flag)

    names = [i[:-5-len(flag)]+i[-4:] for i in os.listdir(input_path) if flag in i]
    has_flag = True
    if len(names)==0:
        names = [i for i in os.listdir(input_path)]
        has_flag = False


    pbar = tqdm(names, ncols=100)
    data_num = len(names)
    metric_recorders = {}
    for k in sorted(metrics):
        if k in {'FM', 'FMAX', 'FMEAN'}:
            metric_recorders[k] = METRIC_DIC[k](num=data_num)
        else:
            metric_recorders[k] = METRIC_DIC[k]()

    binary = flag=='mask'
    file_metric_dic = {}
    rgb_path = os.path.join(gt_pre, dataset_name, 'RGB')
    for i, name in enumerate(pbar):
        pbar.set_description('{:03d}/{:03d}: {}'.format(i + 1, data_num, name))
        gt = np.array(Image.open(os.path.join(gt_path, name)))
        flag_name = name[:-4]+f'_{flag}'+name[-4:] if has_flag else name
        pred = np.array(Image.open(os.path.join(input_path, flag_name)))
        if len(gt.shape)==3 or len(pred.shape)==3:
            print('Error shape pred, gt', name, pred.shape, gt.shape)
            if gt.shape[0]==3:
                gt = gt[0]
            elif len(gt.shape)==3 and gt.shape[2]==3:
                gt = gt[:,:,0]
            if pred.shape[0]==3:
                pred = pred[0]
            elif len(pred.shape)==3 and pred.shape[2]==3:
                pred = pred[:,:,0]

        if flag == 'mask':
            if pred.max()==pred.min():
                print('All zeros', name)
            else:
                pred = (pred-pred.min())/(pred.max()-pred.min())
            if gt.max()>1:
                gt = gt>128

        gt = gt.astype(np.float)
        pred = pred.astype(np.float)

        if save_img:
            saveImage(os.path.join(savepath_dic['gt'], flag_name ), gt, binary=binary)
            saveImage(os.path.join(savepath_dic['pred'], flag_name), pred, binary=binary)
            saveImage(os.path.join(savepath_dic['gt_pred'], flag_name), gt, pred, binary=binary)
            # suf = '.png' if  dataset_name.startswith('HKU') else '.jpg'
            suf = '.jpg'
            cur_name = name[:-4]+ suf
            shutil.copy(os.path.join(rgb_path, cur_name), os.path.join(savepath_dic['RGB'], cur_name))

        if flag=='mask':
            file_metric_dic[name[:-4]] = np2py(calDice(pred, gt))
        else:
            file_metric_dic[name[:-4]] = np2py(np.abs(pred-gt).mean())

        for metric, recorder in metric_recorders.items():
            recorder.update(pred.astype(np.float), gt.astype(np.float))
    toYaml(os.path.join(output_path, f'{flag}_metrics.yaml'), file_metric_dic)
    results = {k:v.show() for k, v in metric_recorders.items()}
    for key in ['FM', 'EM']:
        if key in results:
            pre = key[0]
            results[f'{pre}MAX'], results[f'{pre}MEAN'] = results[key]
            del results[key]
    for k in sorted(results):
        print(k, results[k])
    return results, data_num


def evaluate_main(input, prefix, eval_depth=False, save_img=False, output=None):
    if not output:
        output = os.path.dirname(input) + '/eval_'+os.path.basename(input)
    mkdir(output)
    flags = ['mask']
    if eval_depth:
        flags.append('depth')

    flag_metrics = {
                    'mask':['FM', 'MAE', 'SM', 'EM'],
                    'depth': ['logMSE', 'logRMSE', 'log10MAE', 'MRE'],
                   }
    metric_data = {}
    for flag in flags:
        data_dic = metric_data[flag] = {}
        num_dic = {}
        metrics = flag_metrics[flag]
        data_names = sorted(os.listdir(input))
        for dataset in data_names:
            out_pre = os.path.join(output, dataset)
            mkdir(out_pre)
            in_path = os.path.join(input, dataset)
            results, data_num = evaluate(dataset, in_path, out_pre, prefix,  save_img, flag, metrics)
            data_dic[dataset] = np2py(results)
            num_dic[dataset] = data_num
            toYaml(os.path.join(output,'summary.yaml'), metric_data)
        mean_metric = {}
        for key in ['FM', 'EM']:
            if key in metrics:
                metrics.remove(key)
                metrics.append(f'{key[0]}MAX')
                metrics.append(f'{key[0]}MEAN')
        for m in metrics:
            mean_metric[m] = sum(n*data_dic[d][m] for d,n in num_dic.items())/sum(num_dic.values())
        metric_data[flag+'_mean'] = mean_metric
        print(flag, 'mean')
        print(mean_metric)
        toYaml(os.path.join(output,'summary.yaml'), metric_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optinal
    parser.add_argument("-s", "--save_img", action='store_true')
    parser.add_argument("-d", "--depth", action='store_true')
    parser.add_argument("-p", "--prefix", default='/apdcephfs/share_1290796/heqinzhu/SOD_data')
    parser.add_argument("-o", "--output", type=str)
    # required
    parser.add_argument("-i", "--input", type=str, required=True)

    args =  parser.parse_args()
    evaluate_main(args.input, args.prefix, args.depth, args.save_img, args.output)
