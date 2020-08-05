import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


Tamanho = [[6], [8], [10], [14], [18]]
preço = [[7], [9], [13], [17.5], [18]]

# Passamos a formula matemática
model = LinearRegression()
# Treinamos o nosso modelo
model.fit(Tamanho, preço)

Tamanho_Hom = [[19], [42], [50]]
preço_hom = [[22.60], [43.5], [51.60]]
predito = model.predict(Tamanho_Hom)

real      = pd.DataFrame(Tamanho_Hom, columns=['real'])
predito   = pd.DataFrame(model.predict(Tamanho_Hom), columns=['predito'])
resultado = pd.concat([real, predito], axis=1, sort=False)
resultado['diferença'] = resultado['real'] - resultado['predito']
print ("Linear Regression Resultados" )
print ("\nComparando os valores" )
print(resultado)
print('\nEficiência do treinamento:  ', model.score(Tamanho, preço))
# Calculando o erro médio
# Quanto mais próximo de zero melhor, significa que o modelo está errando pouco
print('O erro médio foi:           ', mean_squared_error(preço_hom, model.predict(Tamanho_Hom)))
print('A eficência do modelo foi:  ', model.score(Tamanho_Hom, preço_hom))