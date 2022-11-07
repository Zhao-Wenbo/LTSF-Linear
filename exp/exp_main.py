from http import server
from operator import index
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, visual, test_params_flop, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd


import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from data_provider.data_loader import Dataset_Electricity
from dataset.forcast import get_electricity_df
from torch.utils.data import DataLoader
import seaborn as sns

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # self.df = get_electricity_df() if args.data == 'other' else None
        self.df = pd.read_csv('./dataset/all.csv', index_col=0)
        self.users = self.df.user.unique() if args.data == 'other' else None
        self.labels = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 
        0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 
        2, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 
        1, 1, 2, 0, 2, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 1, 2, 
        2, 1, 2, 0, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
        0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1
        ])

        # self.users = self.users[self.labels == 2]
        self.df = self.df[self.df.user.isin(self.users)]

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        for (k, v) in model.named_parameters():
            print(k, v.shape)

        if self.args.resume is not None:
            model.load_state_dict(torch.load(self.args.resume))
            print("load resume model succesfully!")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.df)
        return data_set, data_loader

    def _select_optimizer(self):
        series_parameters = list()
        date_parameters = list()
        for name, param in self.model.named_parameters():
            if 'date' in name:
                date_parameters.append(param)
            else:
                series_parameters.append(param)

        model_optim = optim.AdamW([
            {'params': series_parameters},
            {'params': date_parameters, 'lr': 1 * self.args.learning_rate}
        ], lr=self.args.learning_rate)
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _MAPE(self, pred_batch, y_batch):
            mape = torch.mean(torch.abs(pred_batch - y_batch) / (y_batch + 1e-3))
            # print(inverse(y_batch))
            # print((inverse(y_batch) >= 0).all())
            # print(mape)
            return mape

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = self._MAPE
        return criterion

    def _adjust_learning_rate(self, optimizer, iteration, total_warmup):
        # warmup
        if total_warmup > 0:
            if iteration == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / total_warmup
            elif 1 <= iteration < total_warmup:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / iteration * (iteration + 1)

    def _select_lrScheduler(self, optimizer, train_loader):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                T_max=(self.args.train_epochs - self.args.warmup) * len(train_loader),
                eta_min=1e-08)
        return scheduler

    def vali(self, vali_data, vali_loader, criterion):
        '''
        scaler = vali_data.scaler
        scaler_mean = torch.tensor(scaler.mean_, requires_grad=False)
        scaler_std = torch.tensor(scaler.scale_, requires_grad=False)
        inverse_transform = lambda values: values * scaler_std + scaler_mean
        '''

        total_loss = []
        total_mape = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, scale_param) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                inverse = lambda batch_y: batch_y * scale_param[1].reshape(-1,1,1) + scale_param[0].reshape(-1,1,1)
                loss = nn.MSELoss()(pred, true)
                mape = self._MAPE(inverse(pred), inverse(true))
                # loss = criterion(pred, true)
                # loss = criterion(pred, true, inverse_transform)

                total_loss.append(loss)
                total_mape.append(mape)
        total_loss = np.average(total_loss)
        total_mape = np.average(total_mape)
        self.model.train()
        return total_loss, total_mape

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        '''scaler = train_data.scaler
        scaler_mean = torch.tensor(scaler.mean_, requires_grad=False)
        scaler_std = torch.tensor(scaler.scale_, requires_grad=False)
        inverse_transform = lambda values: values * scaler_std + scaler_mean'''

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        lr_scheduler = self._select_lrScheduler(model_optim, train_loader)
        warmup = 0
        total_warmup = self.args.warmup * len(train_loader) / self.args.batch_size

        df_mape = pd.DataFrame(columns=['epoch', 'train_mape', 'valid_mape'])

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, scale_param) in enumerate(train_loader):
                '''if epoch <= self.args.warmup:
                    self._adjust_learning_rate(model_optim, warmup, total_warmup)
                    warmup += 1
                else:
                    lr_scheduler.step()'''

                # print(batch_x.shape, batch_x_mark.shape)
                # print('*****', (inverse_transform(batch_y) >= -0.01).all())
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # print(batch_y.shape, scale_param)
                    inverse = lambda batch_y: batch_y * scale_param[1].reshape(-1,1,1) + scale_param[0].reshape(-1,1,1)
                    # loss = criterion(outputs, batch_y)
                    loss = self._MAPE(inverse(outputs), inverse(batch_y))
                    # loss = criterion(outputs, batch_y)
                    # loss = criterion(outputs, batch_y, inverse_transform)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mape = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mape = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali Loss: {3:.5f}/{4:.5f} Test Loss: {5:.5f}/{6:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mape, test_loss, test_mape))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            df_mape.loc[len(df_mape)] = [i, train_loss, vali_mape]
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        '''df_mape.to_csv('./dataset/mape.csv')
        fig, ax = plt.subplots(1,1,figsize=(6,10))
        sns.lineplot(x='epoch', y=['train_mape', 'valid_mape'], data=df_mape)
        plt.show()'''
        return self.model

    def test(self, setting):
        self.model.eval()
        criterion = self._select_criterion()

        step = self.args.step
        pred_df = self.df[pd.to_datetime(self.df.date).dt.year==2014].copy(deep=True)
        all_pred = []
        mape_all = []
        with torch.no_grad():
            for user in self.users:
                pred = np.array([])
                print(f'Test user {user}')
                
                test_data = Dataset_Electricity(self.df, flag='test', 
                    size=[self.args.seq_len, self.args.label_len, 0], # self.args.pred_len], 
                    features='S', user=user, target='sum_per_day', scale=True, timeenc=1, freq='D')
                test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
                print(len(test_loader))
                
                # print(self.vali(test_data, test_loader, self._MAPE))
                # mape_user = []
                for i, (x, y, x_date, y_date, scale_param) in enumerate(test_loader):
                    # scale_param_user = scale_param
                    if i % step != 0:
                        continue
                    replace = min(i, self.args.pred_len)

                    x_val = x.detach().numpy().reshape(-1)
                    if replace > 0:
                        x_val[-replace: ] = pred[-replace: ]
                    x = torch.tensor(x_val.reshape(1, -1, 1), requires_grad=False, dtype=x.dtype)
                    
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    x_date = x_date.float().to(self.device)
                    y_date = y_date.float().to(self.device)

                    y_pred = self.model(x, x_date)

                    # print(y_pred.shape, y.shape)

                    inverse = lambda batch_y: batch_y * scale_param[1].reshape(-1,1,1) + scale_param[0].reshape(-1,1,1)
                    # mape = self._MAPE(inverse(y_pred), inverse(y))
                    # mape = criterion(y_pred*scale_param[1]+scale_param[0], y[:, -self.args.pred_len]*scale_param[1]+scale_param[0]).item()
                    # print(mape)

                    # mape_user.append(mape)
                    
                    pred = np.concatenate([pred, (y_pred).reshape(-1).detach().numpy()[:step]])
                    # print(pred.shape)
                    # print(x.shape, y.shape, x_date.shape, y_date.shape)
                    # print((y[0,-self.args.pred_len:]*scale_param[1]+scale_param[0]))
                    # print(self.df[(self.df.user==user) & (pd.to_datetime(self.df.date).dt.year==2014) ])
                pred = torch.tensor(pred)
                pred = inverse(pred[:self.args.test_size])
                truth = self.df[(self.df.user==user) & (pd.to_datetime(self.df.date).dt.year==2014)].sum_per_day.values

                # print(pred.shape)
                # pred_df[pred_df.user == user].sum_per_day.values = pred.values
                # print(all_pred, pred.detach().numpy())
                all_pred = all_pred + list(pred.detach().numpy().reshape(-1))

                mape_user = self._MAPE(torch.tensor(pred), torch.tensor(truth))
                print(len(all_pred))
                
                mape_all.append(mape_user.item())
                print(mape_all[-1])
                # break
            
            pred_df['pred'] = all_pred
            pred_df.to_csv(f'./dataset/pred_pl{self.args.pred_len}_step{step}_mape{np.mean(mape_all)}.csv')
            print(np.mean(mape_all))




    def _test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, scale_param) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
