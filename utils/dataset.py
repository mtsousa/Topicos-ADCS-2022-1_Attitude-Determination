# @Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
#
# @Descrição: Implementa as classes dos datasets

import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy

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

        with open(self.matrices_dir + '/matrix_A.npy', 'rb') as f1, open(self.matrices_dir + '/matrix_B.npy', 'rb') as f2:
            for k in range(num_samples):
                self.matrix_A.append(np.load(f1))
                self.matrix_B.append(np.load(f2))
            
        self.matrix_A = np.array(self.matrix_A)
        self.matrix_B = np.array(self.matrix_B)

    def __len__(self):
        return len(self.matrix_A)

    def __getitem__(self, idx):
        sample = {'matrix B': from_numpy(self.matrix_B[idx]), 'matrix A': from_numpy(self.matrix_A[idx])}
        return sample

class CrassidisDataset(Dataset):
    
    def __init__(self, case, N=1000):
        '''
        Inicializa a classe do dataset

        Argumentos
            case: Caso de teste
            N: Número de amostras
        '''
        A_true = np.array([[0.352, 0.864, 0.360],
                           [-0.864, 0.152, 0.480],
                           [0.360, -0.480, 0.800]])
        
        test_case = [self.case1, self.case2, self.case3, self.case4, self.case5, self.case6,
                     self.case7, self.case8, self.case9, self.case10, self.case11, self.case12]
        
        r, n = test_case[case-1](N)
        
        r_len = len(r[0])
        
        b = np.zeros((N, r_len, 3))
        
        for k in range(N):
            b[k][0] = (np.matmul(A_true, np.transpose(r[k][0]).reshape((3, 1))) + n[k][0][0]).ravel()
            b[k][1] = (np.matmul(A_true, np.transpose(r[k][1]).reshape((3, 1))) + n[k][0][1]).ravel()
            try:
                b[k][2] = (np.matmul(A_true, np.transpose(r[k][2]).reshape((3, 1))) + n[k][0][2]).ravel()
            except:
                pass
        
        for k in b:
            k[0] = k[0]/np.linalg.norm(k[0])
            k[1] = k[1]/np.linalg.norm(k[1])
            try:
                k[2] = k[2]/np.linalg.norm(k[2])
            except:
                pass
        
        B = np.zeros((N, 3, 3))

        for k in range(N):
            a = 1/r_len
            B[k] = a*np.matmul(np.transpose(b[k][0]).reshape((3, 1)), r[k][0].reshape((1, 3)))
            B[k] += a*np.matmul(np.transpose(b[k][1]).reshape((3, 1)), r[k][1].reshape((1, 3)))
            try:
                B[k] += a*np.matmul(np.transpose(b[k][2]).reshape((3, 1)), r[k][2].reshape((1, 3)))
            except:
                pass

        self.matrix_A = A_true
        self.matrix_B = B
        self.vector_r = r
        self.vector_b = b

    def __len__(self):
        return len(self.matrix_B)

    def __getitem__(self, idx):
        sample = {'matrix B': from_numpy(self.matrix_B[idx]), 'matrix A': from_numpy(self.matrix_A), 'vector r': from_numpy(self.vector_r[idx]), 'vector b': from_numpy(self.vector_b[idx])}
        return sample
    
    def case1(self, N):
        r = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (N, 1)).reshape(N, 3, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 10**(-6), 10**(-6)]]), (N, 1)).reshape(N, 1, 3)
        n = np.random.normal(mu, sigma, (N, 1, 3))
        return r, n

    def case2(self, N):
        r = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 10**(-6)]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case3(self, N):
        r = np.tile(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (N, 1)).reshape(N, 3, 3)
        mu, sigma = 0, np.tile(np.array([[0.01, 0.01, 0.01]]), (N, 1)).reshape(N, 1, 3)
        n = np.random.normal(mu, sigma, (N, 1, 3))
        return r, n

    def case4(self, N):
        r = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[0.01, 0.01]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case5(self, N):
        r = np.tile(np.array([[0.6, 0.8, 0], [0.8, -0.6, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 0.01]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case6(self, N):
        r = np.tile(np.array([[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01]]), (N, 1)).reshape(N, 3, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 10**(-6), 10**(-6)]]), (N, 1)).reshape(N, 1, 3)
        n = np.random.normal(mu, sigma, (N, 1, 3))
        return r, n

    def case7(self, N):
        r = np.tile(np.array([[1, 0, 0], [1, 0.01, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 10**(-6)]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case8(self, N):
        r = np.tile(np.array([[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01]]), (N, 1)).reshape(N, 3, 3)
        mu, sigma = 0, np.tile(np.array([[0.01, 0.01, 0.01]]), (N, 1)).reshape(N, 1, 3)
        n = np.random.normal(mu, sigma, (N, 1, 3))
        return r, n

    def case9(self, N):
        r = np.tile(np.array([[1, 0, 0], [1, 0.01, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[0.01, 0.01]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case10(self, N):
        r = np.tile(np.array([[1, 0, 0], [0.96, 0.28, 0], [0.96, 0, 0.28]]), (N, 1)).reshape(N, 3, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 0.01, 0.01]]), (N, 1)).reshape(N, 1, 3)
        n = np.random.normal(mu, sigma, (N, 1, 3))
        return r, n

    def case11(self, N):
        r = np.tile(np.array([[1, 0, 0], [0.96, 0.28, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[10**(-6), 0.01]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n

    def case12(self, N):
        r = np.tile(np.array([[1, 0, 0], [0.96, 0.28, 0]]), (N, 1)).reshape(N, 2, 3)
        mu, sigma = 0, np.tile(np.array([[0.01, 10**(-6)]]), (N, 1)).reshape(N, 1, 2)
        n = np.random.normal(mu, sigma, (N, 1, 2))
        return r, n
