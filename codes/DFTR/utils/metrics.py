# modified from https://github.com/zyjwuyan/SOD_Evaluation_Metric/

import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist


class cal_fm(object):
    # Fmeasure(maxFm,meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.thds = thds
        self.avg_f = self.img_num = 0.0
        self.score = np.zeros(255)
        self.prec_avg = np.zeros(255)
        self.recall_avg=np.zeros(255)

    def update(self, pred, gt):
        prec, recall= self.cal(pred, gt)
        self.prec_avg+=prec
        self.recall_avg+=recall
        beta2=0.3
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-20)
        f_score[f_score != f_score] = 0 # for Nan
        self.avg_f += f_score
        self.img_num += 1
        self.score = self.avg_f / self.img_num

    def cal(self, y_pred, y, num=255):
        prec, recall = np.zeros(num), np.zeros(num)
        thlist = np.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).astype(np.float)
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def show(self):
        return self.score.max(), self.score.mean()

class cal_fmax(cal_fm):
    def show(self):
        return super().show()[0]


class cal_fmean(cal_fm):
    def show(self):
        return super().show()[1]

class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self._S_object(pred, gt) + (1-self.alpha) * self._S_region(pred, gt)
            if score<0:
                score=0
        return score

    def _S_object(self, pred, gt):
        fg = np.where(gt==0, np.zeros_like(pred), pred)
        bg = np.where(gt==1, np.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg

        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)

        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.shape[-2:]
        gt = gt.reshape(rows, cols)
        if gt.sum() == 0:
            X = np.eye(1) * round(cols / 2)
            Y = np.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            i = np.arange(0,cols)
            j = np.arange(0,rows)
            X = np.round((np.sum(gt,axis=0)*i).sum() / total)
            Y = np.round((np.sum(gt,axis=1)*j).sum() / total)

        return X.astype(np.long), Y.astype(np.long)
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.shape[-2:]
        area = h*w
        gt = gt.reshape(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.astype(np.float)
        Y = Y.astype(np.float)
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.shape[-2:]
        pred = pred.reshape(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.astype(np.float)
        h, w = pred.shape[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0

        return Q
class cal_em(object):
    #Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y, num=255):
        score = np.zeros(num)
        thlist = np.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).astype(np.float)
            if np.mean(y) == 0.0:  # the ground-truth is totally black
                enhanced = 1 - y_pred_th
            elif np.mean(y) == 1.0:  # the ground-truth is totally white
                enhanced = y_pred_th
            else:  # normal cases
                fm = y_pred_th - y_pred_th.mean()
                gt = y - y.mean()
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

            score[i] = np.sum(enhanced) / (y.size - 1 + 1e-20)

        return score
    def show(self):
        score = np.mean(self.prediction, axis=0)
        return score.max(), score.mean()

class cal_emax(cal_em):
    def show(self):
        return super().show()[0]


class cal_emean(cal_em):
    def show(self):
        return super().show()[1]

def calDice(probs, gt, eps=1e-8):
    ''' targets: {0, 1}
    '''
    m1 = probs.flatten()
    m2 = gt.flatten()
    intersection = (m1 * m2)
    score = (2. * intersection.sum() + eps) / (m1.sum() + m2.sum() + eps)
    return score.item()


def calIOU(probs, gt, eps=1e-8):
    ''' targets: {0, 1}
    '''
    probs[probs < 0.5] = 0
    probs[probs >= 0.5] = 1
    m1 = probs.flatten()
    m2 = gt.flatten()
    intersection = (m1 * m2)
    score = (intersection.sum() + eps) / (m1.sum() + m2.sum() - intersection.sum() + eps)
    return score.item()


class cal_dice:
    def __init__(self, eps=1e-8):
        self.data = []
        self.eps = eps
    def update(self, pred, gt):
        res = calDice(pred, gt, self.eps)
        self.data.append(res)
    def show(self):
        return np.mean(self.data)
    def data(self):
        return self.data


class cal_iou:
    def __init__(self, eps=1e-8):
        self.data = []
        self.eps = eps
    def update(self, pred, gt):
        res = calIOU(pred, gt, self.eps)
        self.data.append(res)
    def show(self):
        return np.mean(self.data)


class cal_mre:
    ''' mean relative absolute error '''
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        pred = pred*255+1
        gt = gt*255+1
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt)/gt)
    def show(self):
        return np.mean(self.data)


class cal_rmse:
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        # pred = pred*255+1
        # gt = gt*255+1
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.sqrt(np.mean((pred - gt)**2))
    def show(self):
        return np.mean(self.data)


class cal_mse:
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        # pred = pred*255
        # gt = gt*255
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.mean((pred - gt)**2)
    def show(self):
        return np.mean(self.data)


class cal_logmse:
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        pred = 1 + pred*255
        gt = 1 + gt*255
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.mean((np.log(pred) - np.log(gt))**2)
    def show(self):
        return np.mean(self.data)


class cal_logrmse:
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        pred = 1 + pred*255
        gt = 1 + gt*255
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.sqrt(np.mean((np.log(pred) - np.log(gt))**2))
    def show(self):
        return np.mean(self.data)


class cal_log10mae:
    def __init__(self):
        self.data = []
    def update(self, pred, gt):
        pred = 1 + pred*255
        gt = 1 + gt*255
        res = self.cal(pred, gt)
        self.data.append(res)
    def cal(self, pred, gt):
        return np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    def show(self):
        return np.mean(self.data)
