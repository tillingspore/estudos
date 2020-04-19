import random
from math import exp
import json
import time

"""
s = ([x1,x2,x3,x4] * [[w1],[w2],[w3],[w4]]) + bias

sig = 1 / (e ** (-s))


"""

class perceptron:

    def __init__(self):
        #
        self.weight_path = 'nn.json'
        self.sets = {}
        #
        self.inputs = 0
        self.weight = []       
        self.bias = 0
        #
        self.train_obj = 0
        self.inputs_lenght = 0
        self.trainset = []
        self.trainstatus = True

    ####
    #prepara para treinar (preparetotraining)

    def ptt(self):
    
        print("prepare setings files")

        self.getweight()
        
        print("starting generations")        

        for i in range(self.inputs_lenght,self.train_obj):
        
            print("treinamento com {} entradas".format(i))
            time.sleep(2)

            self.gerData(i)
        
            print("pesos iniciais gerados")

            self.gerIn(i)
        
            self.trainer()

            print("pesos finais prontos")

            self.sets["weights"].append({str(i):self.weight})

            self.sets["bias"].append({str(i):self.bias})

            self.sets["training"]["lenght_in"] = i + 1
        
            self.getweight(mode='w')
            print('='*50)
            time.sleep(10)
    ####
    # gerar pesos

    def gerData(self,h):
        for i in range(h):
            self.weight.append(self.rand())
        self.bias = self.rand()

    
    
    ####
    #gerador de trainsets
    def gerIn(self,h):

        print(2**h,"possibilidades")

        tentativas = 0
    
        while len(self.trainset) != 2**h:
    
            g = []
            t = 0
    
            for i in range(h):
    
                g.append(random.randint(0,1))
    
            if any(g):
    
                t = 1
    
            else :
    
                t = 0  
          
            if {"inputs":g,"output":t} not in self.trainset :

                self.trainset.append({"inputs":g,"output":t})

                tentativas = 0        

            else :
                tentativas += 1

        print("trainsets finalizados")


    ####
    #pega/salva os pesos no json
    
    def getweight(self,mode='r'):

        with open(self.weight_path,mode) as pesos:

            if mode == 'r':

                self.sets = json.load(pesos)

                self.inputs_lenght = self.sets['training']['lenght_in']

                self.train_obj = 2*self.inputs_lenght
                
            if mode == 'w':

                json.dump(self.sets,pesos)

    ####       
    #prepara para salvar os pesos    
     
    def prepareToSave(self,node):

        self.sets['weights'][str(node)] = self.weight

        self.sets['bias'][str(node)] = self.bias

        self.sets['training']['lenght_in'] = self.inputs_lenght

    ####
    #gerador de pesos para o neuronio
    
    def rand(self):

        return random.uniform(0,1)

    ####
    #funcao sigmoid

    def sigmoid(self,x):

        return 1 / (1 + exp(-x)) #sigmoid

    ####
    #executa o neuronio
    
    def run(self,inp): 
        
        p = 0.0
  
        for i in range(len(inp)):

            p += inp[i] * self.weight[i]

        p += self.bias

        return self.sigmoid(p)

    ####
    #modo de aprendizado do neuronio

    def trainer(self):

        for i in range(2000000):##loop de tamanho do treino
           
            for j in range(len(self.trainset)): #loop dos trainsets
            
                y = self.run(self.trainset[j]["inputs"]) #executa
                
                for k in range(len(self.trainset[j]["inputs"])):
        
                    self.weight[k] += (self.trainset[j]["output"] - y) * self.trainset[j]["inputs"][k]
        
                self.bias += (self.trainset[j]["output"] - y)
    
                print("geracao {} |esperado: {} -> saida: {} ".format(i,self.trainset[j]["output"],y))

   ####
    #funcao de ativacao

    def active(x):

        if x >= 0.5:

            return True

        else:

            return False



p = perceptron()

p.ptt()
    

#note
#
# f(x) = 1 / (1+e**(-x))
#
# f'(x) = x(1-x)
#
#f(x) = (e**2x-1)/(e**2x+1)
#
#f'(x) = 1-x**2
#
#

#matriz = lin * col


