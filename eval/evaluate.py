"""
@Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Avalia o modelo treinado com os dados do AlfaCrux
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir))
sys.path.append(PROJECT_ROOT)

from utils.model import BSwishNet
from utils.utils import get_vectors_from_json, save_attitude_matrix, compute_attitude_profile_matrix, axis_angle_to_quat 
from torch.autograd import Variable
import torch
import argparse
from glob import glob
from time import perf_counter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Evaluate B-Swish model')
    
    parser.add_argument('-i', '--input', required=True,
                        metavar='/path/to/input.json',
                        help="Path to input file (.json file).")
    parser.add_argument('-m', '--model', required=True,
                        metavar='/path/to/weights.pth',
                        help="Path to weights (.pth file), 'last' or 'first'.")
    parser.add_argument('-o', '--output',
                        metavar='/path/to/output.json',
                        help="Path to input file (.json file).")
    parser.add_argument('-w', '--wahba_weights',
                        metavar='a1, a2, a3', default='1.0, 1.0, 1.0',
                        help="Coma separeted weights for Wahba's problem. (Default: 1.0, 1.0, 1.0)")
    parser.add_argument('-p', '--parameterization', default='axis-angle',
                        choices=['axis-angle', 'quaternion'],
                        help="Parameterization of output attitude matrix. (Default: axis-angle)")

    args = parser.parse_args()
    for key, value in args._get_kwargs():
        if value is not None:
            print(f'{key.capitalize()}: {value}')
    print()

    start = perf_counter()
    model = BSwishNet(batch_size=1)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    try:
        # Carrega os pesos
        if args.model == 'last':
            params = glob('model/*.pth')[-1]
            model_params = torch.load(params, map_location=torch.device(device))
        else:
            model_params = torch.load(args.model, map_location=torch.device(device))
        
        model.load_state_dict(model_params['model_state_dict'])

        model.eval()

        r, b = get_vectors_from_json(args.input)
        
        if args.wahba_weights:
            a = args.wahba_weights.split(', ')
            a = [float(k) for k in a]
        else:
            a = []
        
        B = compute_attitude_profile_matrix(r, b, a=a, n=r.shape[1])

        if torch.cuda.is_available():
            B = B.to(device)
                
        B = Variable(B).float()
        print('-------')
        A_pred = model(B)
        print(f'Tempo de decorrido: {round(perf_counter()-start, 6)}s\n')

        parametrization = 'axis-angle'
        if args.parametrization == 'quaternion':
            parametrization = 'quaternion'
            A_pred = axis_angle_to_quat(A_pred, device)

        output = args.output
        if not output:
            output = args.input[:-5] + '_output_' + parametrization + '.json'

        save_attitude_matrix(A_pred, output, parametrization)
    except Exception as e:
        print(e)
