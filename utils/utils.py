# @Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
#
# @Descrição: Implementa funções auxiliares para o modelo

import numpy as np
from os import mkdir, rename
import torch
import matplotlib.pyplot as plt
from glob import glob
from pandas import read_excel
from json import load, dump
from utils.dataset import CrassidisDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

def save_dataset(index, matrix, name, path):
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

def gen_training_data():
    '''
    Cria os datasets de treino e de validação com a parametrização do Eixo/ângulo do Euler
    '''
    TEST_SIZE = 3*10**3
    NUM_SAMPLES = 16*10**3 + TEST_SIZE # treino + validação + teste
    SIZE_VALIDATION = 0.2
    SAMPLES_INDEX = np.array([k for k in range(0, NUM_SAMPLES)])
    np.random.shuffle(SAMPLES_INDEX)
    TEST_INDEX = SAMPLES_INDEX[0:int(TEST_SIZE)]
    np.delete(SAMPLES_INDEX, [j for j in range(0, int(TEST_SIZE))])
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

    for k in b:
        k[0] = k[0]/np.linalg.norm(k[0])
        k[1] = k[1]/np.linalg.norm(k[1])
        k[2] = k[2]/np.linalg.norm(k[2])

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
        mkdir(f'data/training')
    except:
        pass
    save_dataset(TRAINING_INDEX, r, 'vector_r', 'data/training')
    save_dataset(TRAINING_INDEX, A_true, 'matrix_A', 'data/training')
    save_dataset(TRAINING_INDEX, b, 'vector_b', 'data/training')
    save_dataset(TRAINING_INDEX, n, 'vector_n', 'data/training')
    save_dataset(TRAINING_INDEX, B, 'matrix_B', 'data/training')

    # Salva o dataset de validação
    try:
        mkdir(f'data/validation')
    except:
        pass
    save_dataset(VALIDATION_INDEX, r, 'vector_r', 'data/validation')
    save_dataset(VALIDATION_INDEX, A_true, 'matrix_A', 'data/validation')
    save_dataset(VALIDATION_INDEX, b, 'vector_b', 'data/validation')
    save_dataset(VALIDATION_INDEX, n, 'vector_n', 'data/validation')
    save_dataset(VALIDATION_INDEX, B, 'matrix_B', 'data/validation')

    # Salva o dataset de teste
    try:
        mkdir(f'data/test')
    except:
        pass
    save_dataset(TEST_INDEX, r, 'vector_r', 'data/test')
    save_dataset(TEST_INDEX, A_true, 'matrix_A', 'data/test')
    save_dataset(TEST_INDEX, b, 'vector_b', 'data/test')
    save_dataset(TEST_INDEX, n, 'vector_n', 'data/test')
    save_dataset(TEST_INDEX, B, 'matrix_B', 'data/test')

def load_np(file_name, num_samples=16*10**3):
    '''
    Carrega o arquivo .npy como um vetor

    Argumentos
        file_name: String com o formato do nome dos arquivos para leitura
        num_samples: Número total de amostras
    
    Retorna
        list: Vetor com os valores
    '''
    files = glob(file_name)
    list = []
    for file in files:
        list.append((np.load(file)/num_samples)*180/np.pi)

    list = np.array(list).flatten()

    return list.reshape(1, len(list))

def plot_geodesic_loss(dir='data/geodesic_loss/', save=False):
    '''
    Cria o gráfico do erro geodésico durante o treinamento

    Argumentos
        dir: Diretório root dos arquivos dos dados do erro
        save: Flag de controle para salvar o gráfico
    '''
    training = load_np(dir + 'training*.npy', num_samples=(16*10**3)*0.8)
    validation = load_np(dir + 'validation*.npy', num_samples=(16*10**3)*0.2)
    x = np.linspace(1, len(training[0]), len(training[0]))

    fig, ax = plt.subplots()
    ax.plot(x, training[0], 'r-', label='treinamento')
    ax.plot(x, validation[0], 'g-', label='validação')
    ax.grid(True)
    ax.set_ylabel('Erro geodésico médio (graus)')
    ax.set_xlabel('Época')
    ax.legend()

    if save:
        plt.savefig('data/imagem/erro_treinamento.pdf', format='pdf')

    plt.show()

def get_vectors_from_json(input):
    '''
    Lê o arquivo .json e extrai os vetores r e b

    Argumentos
        input: Nome do arquivo .json de entrada
    '''
    with open(input) as f:
        data = load(f)
    f.close()
    
    r = torch.zeros((1, data['n'], 3))
    b = torch.zeros((1, data['n'], 3))

    r_keys = list(data['r'].keys())
    b_keys = list(data['b'].keys())
    for i in range(data['n']):
        aux = data['r'][r_keys[i]].split(', ')
        aux = [float(k) for k in aux]
        r[0][i] = torch.tensor(aux)

        aux = data['b'][b_keys[i]].split(', ')
        aux = [float(k) for k in aux]
        b[0][i] = torch.tensor(aux)

    return r, b

def save_attitude_matrix(A, output, parametrization):
    '''
    Salva a matriz de atitude no formato .json

    Argumentos
        A: Matriz de atitude
        output: Nome do arquivo .json de saída
        parametrization: Tipo de parametrização adotada
    '''
    A_list = A[0].tolist()
    A_dict = {'Parametrization': parametrization, 'Line 0': '', 'Line 1': '', 'Line 2': ''}
    
    A_dict['Line 0'] = str(A_list[0][0]) + ', ' + str(A_list[0][1]) + ', ' + str(A_list[0][2]) 
    A_dict['Line 1'] = str(A_list[1][0]) + ', ' + str(A_list[1][1]) + ', ' + str(A_list[1][2])
    A_dict['Line 2'] = str(A_list[2][0]) + ', ' + str(A_list[2][1]) + ', ' + str(A_list[2][2])

    with open(output, 'w') as f:
        dump(A_dict, f, indent=4)
    f.close()

def compute_attitude_profile_matrix(r, b, a=[], n=2):
    '''
    Calcula a matriz perfil de atitude do problema de Wahba

    Argumentos
        r: Vetores na referência 
        b: Vetores no corpo
        a: Pesos de Wahba
        n: Número de observações
    '''
    for k in range(n):
        r[0][k] = r[0][k]/torch.linalg.norm(r[0][k])

    for k in range(n):
        b[0][k] = b[0][k]/torch.linalg.norm(b[0][k])

    if a == []:
        a = torch.tensor([1/n for k in range(n)])
    
    B = torch.zeros((1, 3, 3))
    for k in range(n):
        B[0] += a[k]*torch.matmul(b[0][k].view(3, 1), r[0][k].view(1, 3))

    return B

def TRIAD(r, b):
    '''
    Calcula a matriz de atitude pelo método TRIAD

    Argumentos
        r: Vetores na referência
        b: Vetores no corpo
    '''
    r_x = torch.cross(r[0].view(-1, 3, 1), r[1].view(-1, 3, 1))
    r_x = r_x/torch.linalg.norm(r_x, dim=1)

    b_x = torch.cross(b[0].view(-1, 3, 1), b[1].view(-1, 3, 1))
    b_x = b_x/torch.linalg.norm(b_x, dim=1)
    
    A_pred = torch.matmul(b[0].view(-1, 3, 1), r[0].view(-1, 1, 3)) + torch.matmul(torch.cross(b[0].view(-1, 3, 1), b_x.view(-1, 3, 1)), torch.cross(r[0].view(-1, 3, 1), r_x.view(-1, 3, 1)).view(-1, 1, 3)) + torch.matmul(b_x.view(-1, 3, 1), r_x.view(-1, 1, 3))

    return A_pred

def rename_data(dir='data/geodesic_loss/'):
    '''
    Renomeia os arquivos de erro

    Argumentos
        dir: Pasta dos arquivos de erro
    '''
    training = glob(dir + 'training*.npy')
    validation = glob(dir + 'validation*.npy')
    
    for i, j in zip(training, validation):
        nome, inicio, fim = i.split('-')
        fim, extensao = fim.split('.')
        rename(i, nome + '-' + inicio.zfill(4) + '-' + fim.zfill(4) + '.' + extensao)

        nome, inicio, fim = j.split('-')
        rename(j, nome + '-' + inicio.zfill(4) + '-' + fim.zfill(4) + '.' + extensao)

def crassidis_test(model, device='cpu', num_workers=0):
    '''
    Realiza o teste proposto no Crassidis com o modelo

    Argumentos
        model: Modelo treinado
        device: Tipo de máquina para carregar os vetores
        num_workers: Número de subprocessos para carregar o dataset 
    '''
    for k in range(1, 13):
        print(f'Caso de teste {k}')
        print('-'*20)

        # Carrega o dataset
        test_dataset = CrassidisDataset(case=k)
        test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=True, num_workers=num_workers)
        
        test_loss = {'geodesic': [], 'wahba': []}
        for i_batch, samples in enumerate(test_dataloader):
            data, target = samples['matrix B'], samples['matrix A']

            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            
            data, target = Variable(data).float(), Variable(target).float()
            
            output = model(data)
            geo_loss = geodesic_loss(target, output, device, batch_size=25)
            test_loss['geodesic'].append(geo_loss)
            wb_loss = wahba_loss(data, output, device, batch_size=25)
            test_loss['wahba'].append(wb_loss)

            test_loss['geodesic'] = [k.cpu().detach().numpy() for k in test_loss['geodesic']]
            test_loss['wahba'] = [k.cpu().detach().numpy() for k in test_loss['wahba']]
    
    return test_loss

def TRIAD_test(device='cpu', num_samples=3000):
    '''
    Avalia o método TRIAD no dataset de teste

    Argumentos
        device: Tipo de máquina para carregar os vetores
        num_samples: Número de casos de teste
    '''
    A, B, r, b = [], [], [], []
    with open('data/test/matrix_A.npy', 'rb') as f1, open('data/test/matrix_B.npy', 'rb') as f2, open('data/test/vector_r.npy', 'rb') as f3, open('data/test/vector_b.npy', 'rb') as f4:
        for k in range(num_samples):
            A.append(np.load(f1))
            B.append(np.load(f2))
            r.append(np.load(f3))
            b.append(np.load(f4))

    A = torch.from_numpy(np.array(A)).view(-1, 3, 3)
    B = torch.from_numpy(np.array(B)).view(-1, 3, 3)
    r = torch.from_numpy(np.array(r))
    b = torch.from_numpy(np.array(b))

    test_loss = {'geodesic': [], 'wahba': []}
    for i, j, k, m in zip(r, b, A, B):
        output = TRIAD(i, j)
        geo_loss = geodesic_loss(k, output, device, batch_size=1)
        test_loss['geodesic'].append(geo_loss)
        wb_loss = wahba_loss(B, output, device, batch_size=1)
        test_loss['wahba'].append(wb_loss)

    test_loss['geodesic'] = [k.cpu().detach().numpy() for k in test_loss['geodesic']]
    test_loss['wahba'] = [k.cpu().detach().numpy() for k in test_loss['wahba']]

    return test_loss

def trace(A, device, batch_size=64, dim=3):
    '''
    Calcula o traço de um tensor 3D (batch_size, N, N)

    Argumentos
        A: Tensor 3D
        batch_size: Tamanho da primeira dimensão do tensor
        dim: Tamanho das outras duas dimensões do tensor
    
    Retorna
        output: Traço das matrizes (batch_size, N)
    '''
    mask = torch.zeros((batch_size, dim, dim)).to(device)
    mask[:, torch.arange(0, dim), torch.arange(0, dim) ] = 1.0
    output = A*mask
    output = torch.sum(output, axis=(1, 2))
    return output.to(device)

def geodesic_loss(A_true, A_pred, device, eps=1.e-7, batch_size=64):
    '''
    Calcula o erro geodésico

    Argumentos
        A_true: Matriz de rotação verdadeira
        A_pred: Matriz de rotação da saída da rede
        eps: Limites para normalização da entrada
    
    Retorna
        torch.sum(geodesic_error): Soma do erro geodésico
    '''
    A = torch.matmul(A_true, A_pred.transpose(1, 2).contiguous()).to(device)

    tr_A = trace(A, device, batch_size=batch_size)

    alfa = (tr_A-1)/2
    for i, k in enumerate(alfa):
        if k > 1-eps:
            alfa[i] = 1-eps
        elif k < -1+eps:
            alfa[i] = -1 + eps

    geodesic_error = torch.acos(alfa)

    return torch.sum(geodesic_error.to(device))

def wahba_loss(B, A_pred, device, lambda_0=1, batch_size=64):
    '''
    Calcula o erro de Wahba

    Argumentos
        B: Matriz perfil de atitude
        A_pred: Matriz de rotação da saída da rede
        device: Tipo de máquina para carregar os vetores
        lambda_0: Somatório dos pesos de Wahba
        batch_size: Tamanho do batch do tensor
    '''
    AB = torch.matmul(A_pred, B.transpose(1, 2).contiguous()).to(device)
    tr_AB = trace(AB, device, batch_size=batch_size)
    L = torch.tensor(lambda_0) - tr_AB
    
    return L

def axis_angle_to_quat(A, device):
    '''
    Convete a atitude de Eixo-ângulo de Euler para Quatérnion

    Argumentos
        A: Matriz de atitude
        device: Tipo de máquina para carregar os vetores
    '''
    q = dcm_to_q(A)
    A_quat = quaternion_attitude(q)
    
    return A_quat.float().to(device)

def quaternion_attitude(q):
    '''
    Calcula a matriz de atitude na parametrização de Quatérnion

    Argumentos
        q: Quatérnion
    '''
    q1, q2, q3, q4 = q[1], q[2], q[3], q[0]

    A = torch.zeros((1, 3, 3))

    a11 = q1**2 - q2**2 - q3**2 + q4**2
    a12 = 2*(q1*q2 + q3*q4)
    a13 = 2*(q1*q3 - q2*q4)
    a21 = 2*(q2*q1 - q3*q4)
    a22 = -q1**2 + q2**2 - q3**2 + q4**2
    a23 = 2*(q2*q3 + q1*q4)
    a31 = 2*(q3*q1 + q2*q4)
    a32 = 2*(q3*q2 - q1*q4)
    a33 = -q1**2 - q2**2 + q3**2 + q4**2

    A[0] = torch.tensor([[a11, a12, a13],
                         [a21, a22, a23],
                         [a31, a32, a33]])

    return A

def dcm_to_q(dcm):
    '''
    Extrai o quatérnion a partir da matriz de rotação

    Referência:
        - Shoemake, Quaternions,
        http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
        - ArduPilot,
        https://github.com/ArduPilot/pymavlink/blob/master/quaternion.py

    Argumentos
        dcm: Matriz de rotação
    '''
    assert(dcm.shape == (3, 3))
    q = np.zeros(4)

    tr = np.trace(dcm)
    if tr > 0:
        s = np.sqrt(tr + 1.0)
        q[0] = s * 0.5 # Escalar q4
        s = 0.5 / s
        q[1] = (dcm[2][1] - dcm[1][2]) * s
        q[2] = (dcm[0][2] - dcm[2][0]) * s
        q[3] = (dcm[1][0] - dcm[0][1]) * s
    else:
        dcm_i = np.argmax(np.diag(dcm))
        dcm_j = (dcm_i + 1) % 3
        dcm_k = (dcm_i + 2) % 3

        s = np.sqrt((dcm[dcm_i][dcm_i] - dcm[dcm_j][dcm_j] -
                        dcm[dcm_k][dcm_k]) + 1.0)
        q[dcm_i + 1] = s * 0.5
        s = 0.5 / s
        q[dcm_j + 1] = (dcm[dcm_i][dcm_j] + dcm[dcm_j][dcm_i]) * s
        q[dcm_k + 1] = (dcm[dcm_k][dcm_i] + dcm[dcm_i][dcm_k]) * s
        q[0] = (dcm[dcm_k][dcm_j] - dcm[dcm_j][dcm_k]) * s

    # quaternion no formato q = [q4,, q1, q2, q3]
    return q

def excel2json(input, data_sheet, time_sheet):
    '''
    Extrai as colunas do Excel e converte para .json

    Argumentos
        input: Arquivo .xlsx de entrada
        data_sheet: Vetor com os nomes das planilhas de dados
        time_sheet: Vetor com os nomes das planilhas do tempo dos dados 
    '''
    data = read_excel(input, sheet_name=data_sheet)
    time = read_excel(input, sheet_name=time_sheet, dtype=str)
    
    day = list(time)[:3]
    hour = list(time)[3:]
    time['day_name'] = time[day].apply(lambda x: "-".join(x), axis=1)
    time['hour_name'] = time[hour].apply(lambda x: "-".join(x), axis=1)
    time['name'] = time['day_name'] + '_' + time['hour_name']
    
    output_name = list(time.to_dict()['name'].values())

    grouped_vect = {}

    for key in data.keys():
        df = data[key].transpose().reset_index()
        df =  df.rename(columns=df.iloc[0]).loc[1:]

        dict = df.to_dict()
        
        aux = {}
        for i in range(1, len(dict['x'].values())+1):
            aux[i] = str(dict['x'][i]) + ', ' + str(dict['y'][i]) + ', ' + str(dict['z'][i])  

        grouped_vect[key] = aux

    dict_output = {}
    keys = grouped_vect.keys()
    for i in range(1, 11):
        aux = {'n': int(len(keys)/2), 'r': {} , 'b': {}}
        for j in keys:
            if 'eci' in j or 'ecf' in j:
                aux['r'][j] = grouped_vect[j][i]
            else:
                try:
                    aux['b'][j] = grouped_vect[j][i]
                except:
                    aux['b'][j] = grouped_vect[j][1]
        dict_output[output_name[i-1]] = aux

    for key in dict_output.keys():
        aux = key.split('.')
        output = 'data/alfacrux/' + aux[0] + '.json'
        with open(output, "w") as outfile:
            dump(dict_output[key], outfile, indent=4)

# gen_training_data()