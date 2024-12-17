import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
# 1. Obter Dados Históricos
ticker = 'VIVT3.SA'
data_atual = datetime.now().strftime('%Y-%m-%d')  # Data atual
# Baixar dados históricos até a data atual
dados = yf.download(ticker, start='2023-01-01', end=data_atual)
# 2. Calcular Retornos Diários Logarítmicos
dados['Retorno'] = dados['Adj Close'].pct_change()
dados.dropna(inplace=True)
log_retornos = np.log(1 + dados['Retorno'])
# 3. Estatísticas dos Retornos
media_log_retornos = log_retornos.mean()
desvio_log_retornos = log_retornos.std()
# 4. Configurar Parâmetros da Simulação
dias = 252 * 1  # Número de dias úteis em 1 anos
simulacoes = 1000  # Número de simulações
preco_atual = dados['Adj Close'].iloc[-1].item()  # Extrair valor escalar
# 5. Executar Simulações de Monte Carlo usando GBM
np.random.seed(42)  # Para reprodutibilidade
simulacao = np.zeros((simulacoes, dias))
for i in range(simulacoes):
    # Gerar choques aleatórios
    choques = np.random.normal(0, 1, dias)
    # Calcular variação diária
    variacao_diaria = (media_log_retornos - 0.5 * desvio_log_retornos ** 2) + desvio_log_retornos * choques
    # Calcular preço simulado
    preco_simulado = preco_atual * np.exp(np.cumsum(variacao_diaria))
    simulacao[i] = preco_simulado
# 6. Visualizar Resultados

# Gráfico das simulações


plt.figure(figsize=(12, 6))
plt.plot(simulacao.T, alpha=0.05, color='purple')
plt.title('Simulações de Monte Carlo para VIVT3.SA - Próximos 1 Ano')
plt.xlabel('Dias')
plt.ylabel('Preço Simulado (R$)')
plt.grid(True)
plt.show()

# Histograma dos Preços Finais
# colocar os valores de 10 ate ate 50 de vermelho e de 50 a 70 de roxo e de 70 a 120 de verde

precos_finais = simulacao[:, -1]
plt.figure(figsize=(10, 6))
plt.hist(precos_finais, bins=50, color='purple', edgecolor='black')
plt.title('Distribuição dos Preços Finais após 1 Anos')
plt.xlabel('Preço Final (R$)')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()
# Histograma dos Preços Finais com cores específicas
plt.figure(figsize=(10, 6))

# Definir intervalos de cores
bins = np.linspace(min(precos_finais), max(precos_finais), 70)
colors = ['red' if 10 <= x < 60 else 'purple' if 60 <= x < 67 else 'green' for x in bins[:-1]]

# Plotar histograma com cores específicas
for i in range(len(bins) - 1):
    plt.hist(precos_finais, bins=[bins[i], bins[i + 1]], color=colors[i], edgecolor='black', alpha=0.7)

plt.title('Distribuição dos Preços Finais após 1 Ano')
#tira as linhas de grade
plt.grid(False)


plt.xlabel('Preço Final (R$)')
plt.ylabel('Frequência')

plt.show()

# 7. Estatísticas dos Resultados
percentis = np.percentile(precos_finais, [5, 50, 95])
# faça um print dos valores de media de crescimento, desvio padrão e percentis no ano
print(f'Média de Crescimento: {media_log_retornos:.6f}')
print(f'Desvio Padrão de Crescimento: {desvio_log_retornos:.6f}')

print(f'Preço Atual: R$ {preco_atual:.2f}')
print(f'Preço Projetado (Mediana) após 1 anos: R$ {percentis[1]:.2f}')
print(f'Intervalo de Confiança de 90%: R$ {percentis[0]:.2f} - R$ {percentis[2]:.2f}')

# 6. Visualizar Resultados Adicionais

# Gráfico da Distribuição dos Retornos Diários
plt.figure(figsize=(10, 6))
plt.hist(log_retornos, bins=50, color='blue', edgecolor='black')
plt.title('Distribuição dos Retornos Diários Logarítmicos')
plt.xlabel('Retorno Logarítmico')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()
# Calcula a porcentagem de reduções (red), estabilidade (purple) e aumentos (green)
num_simulacoes = len(precos_finais)
num_red = np.sum((precos_finais >= 10) & (precos_finais < 60))
num_purple = np.sum((precos_finais >= 60) & (precos_finais < 67))
num_green = np.sum(precos_finais >= 67)

percent_red = (num_red / num_simulacoes) * 100
percent_purple = (num_purple / num_simulacoes) * 100
percent_green = (num_green / num_simulacoes) * 100

print(f'Porcentagem de Reduções (Red): {percent_red:.2f}%')
print(f'Porcentagem de Estabilidade (Purple): {percent_purple:.2f}%')
print(f'Porcentagem de Aumentos (Green): {percent_green:.2f}%')
