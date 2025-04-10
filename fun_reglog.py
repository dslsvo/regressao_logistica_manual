import numpy as np
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definindo a classe
class RegressaoLogisticaManual:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        # Inicializa os atributos da classe
        # Taxa de aprendizado
        self.learning_rate = learning_rate
        logging.info(f"Taxa de aprendizado definida como: {learning_rate}")

        # Número de iterações
        self.n_iter = n_iter
        logging.info(f"Número de iterações definido como: {n_iter}")

        # Array de pesos (None)
        self.weights = None
        logging.info("Array de pesos inicializado como None")

        # Viés (None)
        self.bias = None
        logging.info("Viés inicializado como None")

    # Função sigmoide para mapear valores entre 0 e 1
    def _sigmoid(self, z):
        logging.info("Calculando a função sigmoide")
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Recebendo os valores de linhas e colunas de X
        logging.info("Iniciando o treinamento do modelo")
        n_samples, n_features = X.shape

        # Inicializa os pesos com valores aleatórios entre 0 e 1
        self.weights = np.random.rand(n_features)
        logging.info(f"Pesos inicializados com valores aleatórios: {self.weights}")

        # Inicializa o viés com um valor aleatório entre 0 e 1
        self.bias = np.random.rand()
        logging.info(f"Viés inicializado com valor aleatório: {self.bias}")

        # Iterando um número (n_iter) de vezes
        for i in range(self.n_iter):
            logging.info(f"Iniciando iteração {i+1}")

            # Calcula a saída y
            linear_model = np.dot(X, self.weights) + self.bias
            logging.info(f"Saída linear calculada: {linear_model[:5]}") # Mostra os 5 primeiros valores

            # Aplica a função sigmoide para obter as probabilidades
            y_predicted = self._sigmoid(linear_model)
            logging.info(f"Probabilidades calculadas: {y_predicted[:5]}") # Mostra os 5 primeiros valores

            # Calcula os gradientes dos pesos e do viés
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            logging.info(f"Gradiente dos pesos: {dw[:5]}, Gradiente do viés: {db}") # Mostra os 5 primeiros valores de dw

            # Atualiza os pesos usando o gradiente descendente
            self.weights -= self.learning_rate * dw
            logging.info(f"Pesos atualizados: {self.weights[:5]}") # Mostra os 5 primeiros valores

            # Atualiza o viés usando o gradiente descendente
            self.bias -= self.learning_rate * db
            logging.info(f"Viés atualizado: {self.bias}")

    def predict(self, X):
        logging.info("Iniciando previsões")

        # Calcula a saída y
        linear_model = np.dot(X, self.weights) + self.bias
        logging.info(f"Saída linear calculada: {linear_model[:5]}") # Mostra os 5 primeiros valores

        # Aplica a função sigmoide para obter as probabilidades
        y_predicted = self._sigmoid(linear_model)
        logging.info(f"Probabilidades calculadas: {y_predicted[:5]}") # Mostra os 5 primeiros valores

        # Gera as previsões binárias com base nas probabilidades
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        logging.info(f"Previsões binárias geradas: {y_predicted_cls[:5]}") # Mostra os 5 primeiros valores

        # Retorna o y_pred
        return np.array(y_predicted_cls)
