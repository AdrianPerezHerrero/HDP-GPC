# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:43:23 2021

@author: adrianperez
"""

import os

import torch
import wfdb
import numpy as np
from sklearn.preprocessing import scale
from wfdb import processing
import pandas as pd

#Class to get data from UCR or MIT-BIH 
included_labels = ['N', 'L', 'R', 'a', 'A', 'J', 'S', 'e', 'j', 'V', 'E', 'F', '/', 'f', 'Q', '!', 'n']

def get_data(database = "mitdb", record = "100", deriv = 0, test = False, d2_data = False, scale_data = True,
             scale_type ="all", samples = [0,220], ann='atr', filter_labels=True, return_annotations=False, return_snr=False):
    #Define workbench
    homedir = os.getenv('HOME')
    if not homedir is None:
        #We are on linux
        if homedir == '/root':
            homedir = '/home' 
        direct = homedir + "/Documents/just-experiments/data/"
    else:
        #We are on windows
        direct = "D:/Programs/Workspaces/spyder-workspace/just-experiments/data/"
    
    use_wfdb = True
    
    if database == "ucr":
        database = "ucr/UCRArchive_2018/"
        use_wfdb = False
    elif database == "long-term":
        database = "long-term/mit-bih-long-term-ecg-database-1.0.0/"
    elif database == "fantasia":
        database = "fantasia-database-1.0.0/"
    elif database == "apnea":
        database = "apnea-ecg-database-1.0.0/"
    elif database == "mitdb":
        database = "mitdb/"
    elif database == "filtered":
        database = "datos_registros_mit/mit_MEDIAN_OFFLINE_filtered_2023/"
        #database = "datos_registros_mit/mit_WAVELET_filtered_2023/"
        record = record + "_filtered"
    elif database == "stt":
        database = "stt-1.0.0/"
    full_path = direct + database + record
    
    if not use_wfdb:
        file_path_train = full_path + "/" + record + "_TRAIN.tsv"
        file_path_test = full_path + "/" + record + "_TEST.tsv"
    
        data_train = np.genfromtxt(fname = file_path_train, delimiter="\t", skip_header=0)
        labels_train = data_train[:,0].astype(int)
        data_train = data_train[:,1:].astype(np.float64)
        data_train_2d = []
        for d in data_train:
            if scale_data:
                d = scale(d)
                if d2_data:
                    d = np.atleast_2d(d).T
            data_train_2d.append(d)
        data_train = np.array(data_train_2d)
        labels_train = np.array(labels_train)
        
        if test:
            data_test = np.genfromtxt(fname = file_path_test, delimiter="\t", skip_header=0)
            labels_test = data_test[:,0].astype(int)
            data_test = data_test[:,1:].astype(np.float64)
            data_test_2d = []
            for d in data_test:
                if scale_data:
                    d = scale(d)
                if d2_data:
                    d = np.atleast_2d(d).T
                data_test_2d.append(d)
            data_test = np.array(data_test_2d)
            labels_test = np.array(labels_test)
            
            return data_train, labels_train, data_test, labels_test
        else:
            return data_train, labels_train
    else:
        record_breath = None
        file_path = full_path
        record = wfdb.rdrecord(file_path, return_res=32, physical=False)
        if database == "apnea-ecg-database-1.0.0/":
            #labels = wfdb.rdann(file_path, 'qrs', return_label_elements=['symbol']).symbol
            labels = []
            labels.append(wfdb.rdann(file_path, 'apn', return_label_elements=['symbol']).symbol)
            labels.append(wfdb.rdann(file_path, 'qrs', return_label_elements=['symbol']).symbol)
            record_breath = wfdb.rdrecord(file_path, return_res=32, physical=False)
        elif database == "fantasia-database-1.0.0/":
            labels = wfdb.rdann(file_path, 'ecg', return_label_elements=['symbol']).symbol
        else:
            labels_original = wfdb.rdann(file_path, 'atr', return_label_elements=['symbol']).symbol
            if filter_labels:
                if ann == 'xqrs':
                    labels = []
                    for i in labels_original:
                        if i in included_labels:
                            labels.append(i)
                else:
                    labels = []
                    for i in labels_original:
                        if i in included_labels:
                            labels.append(i)
            else:
                labels = []
                for i in labels_original:
                    labels.append(i)

        if database == "apnea-ecg-database-1.0.0/":
            annotation = wfdb.rdann(file_path, 'qrs').sample
        elif database == "fantasia-database-1.0.0/":
            annotation = wfdb.rdann(file_path, 'ecg').sample
        else:
            if ann == 'xqrs':
                sig, fields = wfdb.rdsamp(file_path, channels=[0])
                xqrs = processing.XQRS(sig=sig[:, 0], fs=fields['fs'])
                xqrs.detect()
                annotation = xqrs.qrs_inds
            elif ann == 'atr':
                annotation = wfdb.rdann(file_path, 'atr').sample
                annotation_ = []
                for i, l in enumerate(labels_original):
                    if filter_labels:
                        if l in included_labels:
                            annotation_.append(annotation[i])
                    else:
                        annotation_.append(annotation[i])
                annotation = annotation_
                for i in range(len(annotation)):
                    if annotation[0] - 87 + samples[0] < 0:
                        annotation = annotation[1:]
                        labels = labels[1:]
                    else:
                        break
            if len(labels) != len(annotation):
                print("ERROR ANNOTATION LABELS:")
                print("--- DataXQRS: " + str(len(annotation)) + " - Labels: " + str(len(labels)))
                annotation_atr = wfdb.rdann(file_path, 'atr').sample
                annotation_ = []
                for i, l in enumerate(labels_original):
                    if filter_labels:
                        if l in included_labels:
                            annotation_.append(annotation_atr[i])
                    else:
                        annotation_.append(annotation_atr[i])
                annotation_atr = annotation_
                for i in range(len(annotation)):
                    if annotation[0] - 87 + samples[0] < 0:
                        annotation = annotation[1:]
                        labels = labels[1:]
                    else:
                        if annotation_atr[0] - 87 + samples[0] < 0:
                            annotation_atr = annotation_atr[1:]
                        else:
                            break
                comparer = processing.compare_annotations(np.array(annotation_atr), annotation, 60, sig[:, 0])
                annotation = np.delete(annotation, comparer.unmatched_test_inds)
                annotation = np.append(annotation, comparer.unmatched_ref_sample)
                annotation = np.sort(annotation)
                print("Removed unmatched reference index.")
            

        
        data = []
        if scale_data and scale_type == "all":
            signal = scale(record.d_signal)
            if record_breath is not None:
                signal_br = scale(record_breath.d_signal)
        else:
            signal = record.d_signal
            if record_breath is not None:
                signal_br = record_breath.d_signal
        if scale_data and scale_type == "mean_all":
            signal = signal - np.mean(signal)
        for i in range(len(annotation)):
            # aux = signal[annotation.sample[i+1]-87+samples[0]:annotation.sample[i+1]+samples[1]-87, deriv]
            if annotation[i] + samples[1] - 87 < signal.shape[0]:
                if deriv==None:
                    aux = signal[annotation[i] - 87 + samples[0]:annotation[i] + samples[1] - 87, :]
                else:
                    aux = signal[annotation[i] - 87 + samples[0]:annotation[i] + samples[1] - 87, deriv]
                aux = np.array(aux, dtype=np.float64)
                if aux.shape[0] > 0 and i == 0:
                    first_mean = np.mean(aux)
                    first_sd = np.std(aux)
                if scale_data and scale_type == "single" and aux.shape[0] > 0:
                    aux = scale(aux)
                elif scale_type == "first":
                    aux = (aux - first_mean)/first_sd
                elif scale_type == "mean":
                    aux = aux - np.mean(aux, axis=0)
                if d2_data:
                    aux = np.atleast_2d(aux).T
                data.append(aux)
        data = np.array(data, dtype=np.float64)
        labels = np.array(labels)
        if not return_snr:
            if return_annotations:
                if record_breath is not None:
                    data_br = np.array(signal_br)
                    return data, labels, data_br, annotation
                else:
                    return data, labels, annotation
            else:
                if record_breath is not None:
                    data_br = np.array(signal_br)
                    return data, labels, data_br
                else:
                    return data, labels,
        else:
            #snr = np.array([rolling_snr(signal[:,i], window_size=250) for i in range(signal.shape[1])])
            snr = signaltonoise(signal, axis=0)
            if return_annotations:
                if record_breath is not None:
                    data_br = np.array(signal_br)
                    return data, labels, data_br, annotation, snr
                else:
                    return data, labels, annotation, snr
            else:
                if record_breath is not None:
                    data_br = np.array(signal_br)
                    return data, labels, data_br, snr
                else:
                    return data, labels, snr

def rolling_snr(signal, window_size: int):
    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=window_size).mean()[window_size:]
    rolling_std = signal_series.rolling(window=window_size).std()[window_size:]
    rolling_snr = 10 * np.log10(
        (rolling_mean ** 2).replace(0, np.finfo(float).eps) / (rolling_std ** 2).replace(0, np.finfo(float).eps))  # type: ignore
    return rolling_snr

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)** 2
    #m = 100.0
    sd = a.std(axis=axis, ddof=ddof) ** 2
    return np.where(sd == 0, 0, m / sd)


def take_standard_labels(data, labels, permutation=False, filter=None):
    # MitBih standard + | (not recognised as hearbeat)
    # included_labels = ['N', 'V', '/', 'f', 'Q', '|', '!', 'L', 'R', 'A']
    # included_labels = ['N', 'L', 'R', 'a', 'A', 'J', 'S', 'e', 'j', 'V', 'E', 'F', '/', 'f', 'Q', '!']
    if filter is None:
        included_lab = included_labels
    else:
        included_lab = filter
    subdata = np.zeros(data.shape)
    # TAKE ONLY LABELS 1
    take_normal = True
    permutation = False
    if take_normal:
        if len(data.shape) > 2:
            for ld in range(data.shape[2]):
                for d in range(data.shape[0]):
                    if labels[d] in included_lab:
                        subdata[d, :,ld] = np.array([0 if np.isnan(i) else i for i in data[d,:, ld]])
        else:
            for d in range(data.shape[0]):
                if labels[d] in included_lab:
                    subdata[d] = np.array([0 if np.isnan(i) else i for i in data[d]])
        data = subdata
        labels = [lab for lab in labels if lab in included_lab]
    if len(data.shape) > 2:
        data_2d = data
        #data_2d = np.transpose(data, (0, 2, 1))
    else:
        data_2d = []
        # mean_data = np.mean(data,axis=0)
        for d in data:
            data_2d.append(np.atleast_2d(d).T)
    if permutation:
        p = np.random.permutation(len(labels))
        # p = np.load('permutation_1.npy')
        labels_original = labels
        data_2d = [data_2d[i] for i in p]
        labels = [labels[i] for i in p]

    if permutation:
        return data, data_2d, labels, p
    else:
        return data, data_2d, labels

def compute_estimators_LDS(samples, n_f=None):
    if n_f is None:
        n_f = samples.shape[0] - 2
    samples_ = torch.from_numpy(samples[:n_f][:,:,0].T)
    samples__ = torch.from_numpy(samples[1:n_f + 1][:,:,0].T)

    std = torch.sqrt(torch.mean(torch.diag(torch.linalg.multi_dot(
        [(samples_ - torch.mean(samples_, dim=1)[:, np.newaxis]),
         (samples_ - torch.mean(samples_, dim=1)[:, np.newaxis]).T])) / n_f)).item()
    std_dif = torch.sqrt(torch.mean(torch.diag(torch.linalg.multi_dot(
        [(samples__ - samples_), (samples__ - samples_).T])) / n_f)).item()
    #std_dif = np.max([std, std_dif]) * 1.0
    std_dif = np.max([std * 1.1, std_dif]) * 1.0
    print("Sigma estimated:", str(std))
    print("Gamma estimated:", str(std_dif))
    bound_std = (std * 0.005, std * 1.0)
    bound_std_dif = (std_dif * 0.005, std_dif * 1.0)
    return std, std_dif, bound_std, bound_std_dif
