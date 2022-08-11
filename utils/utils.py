# @Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
#
# @Descrição: Implementa funções auxiliares para o modelo

import numpy as np
import os
from torch.utils.data import Dataset#, DataLoader
import torch

def saveDataset(index, matrix, name, path):
    '''
    Salva as matrizes do numpy no arquivo indicado

    Argumentos
        index: Índices das matrizes para salvamento
        matrix: Matriz com os dados para salvamento
        name: Nome do arquivo de saída
        path: Caminho para pasta do arquivo de saída
    '''
    with open(f'{path}/{name}.npy', 'wb') as f:
        for k in index:
            np.save(f, matrix[k])

def genTrainingData():
    '''
    Cria os datasets de treino e de validação com a parametrização do Eixo/ângulo do Euler
    '''
    NUM_SAMPLES = 16*10**3
    SIZE_VALIDATION = 0.2
    SAMPLES_INDEX = np.array([k for k in range(0, NUM_SAMPLES)])
    np.random.shuffle(SAMPLES_INDEX)
    VALIDATION_INDEX = SAMPLES_INDEX[0:int(NUM_SAMPLES*SIZE_VALIDATION)]
    TRAINING_INDEX = SAMPLES_INDEX[int(NUM_SAMPLES*SIZE_VALIDATION):]

    # Define a matriz de vetores r
    r = np.random.uniform(-100, 100, (NUM_SAMPLES, 3, 3))

    for k in r:
        k[0] = k[0]/np.linalg.norm(k[0])
        k[1] = k[1]/np.linalg.norm(k[1])
        k[2] = k[2]/np.linalg.norm(k[2])

    '''Estrutura da matriz de vetores de r
        Cada linha é um vetor de medidas
        r = [[a1 a2 a3],
             [b1 b2 b3],
             [c1 c2 c3]]'''

    # print('Vetor r:\n', r[0])

    # Define a matriz A_true
    angle = np.random.uniform(-np.pi, np.pi, NUM_SAMPLES)
    axis = np.random.uniform(-1, 1, (NUM_SAMPLES, 1, 3))

    # print('Angulo de Euler:\n', angle[0])
    # print('Eixo de Euler:\n', axis[0][0])

    for k in axis:
        k[0] = k[0]/np.linalg.norm(k[0])

    A_true = np.zeros((NUM_SAMPLES, 3, 3))

    for k in range(NUM_SAMPLES):
        e1, e2, e3 = axis[k][0]
        a11 = float(np.cos(angle[k]) + (e1**2)*(1 - np.cos(angle[k])))
        a12 = float(e1*e2*(1 - np.cos(angle[k])) + e3*np.sin(angle[k]))
        a13 = float(e1*e3*(1 - np.cos(angle[k])) - e2*np.sin(angle[k]))
        a21 = float(e1*e2*(1 - np.cos(angle[k])) - e3*np.sin(angle[k]))
        a22 = float(np.cos(angle[k]) + (e2**2)*(1 - np.cos(angle[k])))
        a23 = float(e2*e3*(1 - np.cos(angle[k])) + e1*np.sin(angle[k]))
        a31 = float(e1*e3*(1 - np.cos(angle[k])) + e2*np.sin(angle[k]))
        a32 = float(e2*e3*(1 - np.cos(angle[k])) - e1*np.sin(angle[k]))
        a33 = float(np.cos(angle[k]) + (e3**2)*(1 - np.cos(angle[k])))

        A_true[k] = [[a11, a12, a13],
                     [a21, a22, a23],
                     [a31, a32, a33]]

    '''Estrutura da matriz de de atitude
        A = [[a11 a12 a13],
             [b21 b22 b23],
             [c31 c32 c33]]'''

    # print('Matriz de atitude:\n', A_true[0])
    # print('Determinate da matriz de atitude:', np.linalg.det(A_true[0]))

    # Define o vetor b
    mu, sigma = 0, np.random.uniform(10**(-6), 10**(-2), (NUM_SAMPLES, 1, 3))
    n = np.random.normal(mu, sigma, (NUM_SAMPLES, 1, 3))

    b = np.zeros((NUM_SAMPLES, 3, 3))

    for k in range(NUM_SAMPLES):
        b[k][0] = (np.matmul(A_true[k], np.transpose(r[k][0]).reshape((3, 1))) + n[k][0][0]).ravel()
        b[k][1] = (np.matmul(A_true[k], np.transpose(r[k][1]).reshape((3, 1))) + n[k][0][1]).ravel()
        b[k][2] = (np.matmul(A_true[k], np.transpose(r[k][2]).reshape((3, 1))) + n[k][0][2]).ravel()

    '''Estrutura da matriz de vetores de b
        Cada linha é um vetor de medidas
        b = [[a1 a2 a3],
             [b1 b2 b3],
             [c1 c2 c3]]'''

    # print('Vetor b:\n', b[0])
    # print('Ruido na medida de b:\n', n[0][0])

    # Define a matriz B
    B = np.zeros((NUM_SAMPLES, 3, 3))

    for k in range(NUM_SAMPLES):
        a = 1/3
        B[k] = a*np.matmul(np.transpose(b[k][0]).reshape((3, 1)), r[k][0].reshape((1, 3)))
        B[k] += a*np.matmul(np.transpose(b[k][1]).reshape((3, 1)), r[k][1].reshape((1, 3)))
        B[k] += a*np.matmul(np.transpose(b[k][2]).reshape((3, 1)), r[k][2].reshape((1, 3)))
    
    '''Estrutura da matriz de perfil da atitude
        B = [[a11 a12 a3],
             [a21 a22 a23],
             [a31 a32 a33]]'''
    
    # print('Matriz de perfil de atitude:\n', B[0])

    # Salva o dataset de treino
    try:
        os.mkdir(f'data/training')
    except:
        pass
    saveDataset(TRAINING_INDEX, r, 'vector_r', 'data/training')
    saveDataset(TRAINING_INDEX, A_true, 'matrix_A', 'data/training')
    saveDataset(TRAINING_INDEX, b, 'vector_b', 'data/training')
    saveDataset(TRAINING_INDEX, n, 'vector_n', 'data/training')
    saveDataset(TRAINING_INDEX, B, 'matrix_B', 'data/training')

    # Salva o dataset de validação
    try:
        os.mkdir(f'data/validation')
    except:
        pass
    saveDataset(VALIDATION_INDEX, r, 'vector_r', 'data/validation')
    saveDataset(VALIDATION_INDEX, A_true, 'matrix_A', 'data/validation')
    saveDataset(VALIDATION_INDEX, b, 'vector_b', 'data/validation')
    saveDataset(VALIDATION_INDEX, n, 'vector_n', 'data/validation')
    saveDataset(VALIDATION_INDEX, B, 'matrix_B', 'data/validation')


class AttitudeProfileDataset(Dataset):
    
    def __init__(self, root_dir, mode='training', dataset_size=0.8, num_samples=16*10**3):
        '''
        Inicializa a classe do dataset

        Argumentos
            root_dir: Diretório root dos arquivos do dataset
            mode: Modo de inicialização 'training' ou 'validation'
            dataset_size: Tamanho do dataset a ser inicializado
            num_samples: Número total de amostras (validação + treinamento)
        '''
        self.matrices_dir = root_dir + mode
        num_samples = int(num_samples*dataset_size)
        self.matrix_A = []
        self.matrix_B = []

        with open(self.matrices_dir + '/matrix_A.npy', 'rb') as f:
            for k in range(num_samples):
                self.matrix_A.append(np.load(f))
            
        self.matrix_A = np.array(self.matrix_A)

        with open(self.matrices_dir + '/matrix_B.npy', 'rb') as f:
            for k in range(num_samples):
                self.matrix_B.append(np.load(f))

        self.matrix_B = np.array(self.matrix_B)

    def __len__(self):
        return len(self.matrix_A)

    def __getitem__(self, idx):
        sample = {'matrix B': torch.from_numpy(self.matrix_B[idx]), 'matrix A': torch.from_numpy(self.matrix_A[idx])}
        return sample

# dataset = AttitudeProfileDataset('data/', dataset_size=0.8)

# print(dataset[0])

# dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=True)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['matrix A'].size(),
#           sample_batched['matrix B'].size())
#     break

# genTrainingData()