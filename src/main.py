# @Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
#
# @Descrição: Implementa o treino e a validação para o modelo

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.model import BSwishNet
from utils.utils import AttitudeProfileDataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import glob
import numpy as np

def trace_of_A(A, device, batch_size=64, dim=3):
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

def geodesic_loss(A_true, A_pred, device, eps=1.e-7):
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

    tr_A = trace_of_A(A, device)
    
    alfa = (tr_A-1)/2
    for i, k in enumerate(alfa):
        if k > 1-eps:
            alfa[i] = 1-eps
        elif k < -1+eps:
            alfa[i] = -1 + eps

    geodesic_error = torch.acos(alfa)
    return torch.sum(geodesic_error.to(device))

def update_optimizer(epoch, optimizer):
    '''
    Atualiza os parâmetros do otimizador a cada 500 épocas

    Argumentos
        epoch: Época do treinamento
        optimizer: Objeto do otimizador
    '''
    if epoch%500 == 0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']/10
            g['weight_decay'] = 0.0001
    else:
        for g in optimizer.param_groups:
            g['weight_decay'] = 0.0

def fit_model(model, dataloader, optimizer, num_epochs, device):
    '''
    Treina o modelo

    Argumentos
        model: Objeto do modelo
        dataloader: Dicionário com o Dataloader de treinamento e de validação
        optimizer: Objeto do otimizador
        num_epochs: Tupla com a época de início e de fim 
    '''
    loss_list = {'training': [], 'validation': []}

    for epoch in range(num_epochs[0], num_epochs[1]+1):
        print(f'Epoch {epoch}/{num_epochs[1]}', flush=True)
        print('-'*20, flush=True)

        # Verifica se precisa atualizar o otimizador
        # update_optimizer(epoch, optimizer)
        
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0

            for i_batch, samples in enumerate(dataloader[phase]):
                data, target = samples['matrix B'], samples['matrix A']

                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)
                
                data, target = Variable(data).float(), Variable(target).float()

                if phase == 'training':
                    optimizer.zero_grad()
                
                output = model(data)
                loss = geodesic_loss(target, output, device)

                epoch_loss += loss

                if phase == 'training':
                    loss.backward()
                    optimizer.step()

            loss_list[phase].append(epoch_loss)
            print(f'{phase.capitalize()} - erro {epoch_loss/len(dataloader[phase])}', flush=True)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, f'model/bswish_model_{str(epoch).zfill(4)}.pth')
        print('', flush=True)

    loss_list['training'] = [k.cpu().detach().numpy() for k in loss_list['training']]
    loss_list['validation'] = [k.cpu().detach().numpy() for k in loss_list['validation']]

    np.save(f'data/geodesic_loss/training-{str(num_epochs[0])}-{str(num_epochs[1])}.npy', np.array(loss_list['training']))
    np.save(f'data/geodesic_loss/validation-{str(num_epochs[0])}-{str(num_epochs[1])}.npy', np.array(loss_list['validation']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Train B-Swish on Attitude profile dataset')
    
    parser.add_argument('command',
                        metavar='<command>',
                        help="'training' or 'evaluate'")
    parser.add_argument('--model', required=True,
                        metavar='/path/to/weights.pth',
                        help="Path to weights (.pth file), 'last' or 'first'")
    parser.add_argument('--epochs',
                        metavar='first_epoch-last_epoch',
                        help="Range of epochs to training")

    args = parser.parse_args()
    print('Command:', args.command)
    print('Model:', args.model)
    print('Epochs:', args.epochs)
    print()

    if args.command == 'training':
        model = BSwishNet(batch_size=64)
    
        if torch.cuda.is_available():
            device = 'cuda'
            num_workers = 2
        else:
            device = 'cpu'
            num_workers = 0

        model.to(device)

        # Define o otimizador
        optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

        try:
            # Carrega os pesos
            if args.model != 'first':
                if args.model == 'last':
                    params = glob.glob('model/*.pth')[-1]
                    model_params = torch.load(params)
                else:
                    model_params = torch.load(args.model)

                model.load_state_dict(model_params['model_state_dict'])
                optimizer.load_state_dict(model_params['optimizer_state_dict'])
   
            else:
                pass
        
            # Carrega o dataset de treino
            training_dataset = AttitudeProfileDataset('data/', dataset_size=0.8)
            training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

            # Carrega o dataset de validação
            validation_dataset = AttitudeProfileDataset('data/', mode='validation', dataset_size=0.2)
            validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

            dataloader = {'training': training_dataloader, 'validation': validation_dataloader}
            
            aux = args.epochs.split('-')
            num_epochs = (int(aux[0]), int(aux[1]))

            fit_model(model, dataloader, optimizer, num_epochs, device)
        except Exception as e:
            print(e)
        
    elif args.command == 'evaluate':
        model = BSwishNet(batch_size=1)

        try:
            # Carrega os pesos
            if args.model == 'last':
                params = glob.glob('model/')[-1]
                model_params = torch.load(params)
            else:
                model_params = torch.load(args.model)
            
            model.load_state_dict(model_params['model_state_dict'])

            model.eval()
        
            # TODO: Carrega a entrada e avalia com a rede neural
        
        except Exception as e:
            print(e)
    
    else:
        print(f"[ERROR] The command '{args.command}' was not identify")
