# @Autor: Matheus Teixeira de Sousa (mtsousa14@gmail.com)
#
# @Descrição: Implementa a classe para o modelo

from torch import nn
import torch

class MappingFunction():
    '''
    Funções adaptadas de https://github.com/papagina/RotationContinuity
    '''
    def normalize_vector(self, v):
        '''
        Normaliza um vetor

        Argumentos
            v: Vetor (batch_size, 3)
        '''
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        
        if torch.cuda.is_available():
            v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        else:
            v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
        
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v/v_mag
        
        return v

    def cross_product(self, u, v):
        '''
        Calcula o produto vetorial

        Argumentos
            u, v: Vetores (batch_size, 3)
        '''
        batch = u.shape[0]
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        return out

    def compute_rotation_matrix_from_ortho6d(self, poses):
        '''
        Computa a matriz de rotação a partir de um vetor 6D

        Argumentos
            poses: Vetor 6D (batch_size, 6)
        '''
        x_raw = poses[:,0:3]
        y_raw = poses[:,3:6]

        x = self.normalize_vector(x_raw) 
        z = self.cross_product(x,y_raw) 
        z = self.normalize_vector(z)
        y = self.cross_product(z,x)

        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)

        matrix = torch.cat((x,y,z), 2)
        return matrix

class BSwishNet(nn.Module):

    def __init__(self, batch_size=64):
        '''
        Inicializa a classe da rede neural

        Argumentos
            batch_size: Tamanho do lote do arquivos de treino
        '''
        super(BSwishNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9)
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.swish4 = nn.SiLU()
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)
        self.drop4 = nn.Dropout(p=0.1)
        self.dense1 = nn.Linear(256, 512)
        self.dense2 = nn.Linear(512, 6)
        self.mapping = MappingFunction()

        self.batch_size = batch_size

    def forward(self, x):
        x = x.view(self.batch_size, 1, 9).contiguous()
        x = self.swish1(self.conv1(x))
        x = self.swish2(self.conv2(self.drop1(x)))
        x = self.swish3(self.conv3(self.drop2(x)))
        x = x.view(self.batch_size, -1).contiguous()
        x = self.swish4(self.dense1(self.drop3(x)))
        x = self.dense2(self.drop4(x))
        return self.mapping.compute_rotation_matrix_from_ortho6d(x)

# model = BSwishNet(batch_size=2)
# batch_size, H, W = 2, 3, 3
# x = torch.randn(batch_size, H, W)
# output = model(x)
# print(output.size())
# print(output.det())
