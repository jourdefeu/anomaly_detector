import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
import ipaddress

INPUT_DATA = 'traffic_data.csv'

# Предобработкa данных
def preprocess_data(data=INPUT_DATA):
    print('Загрузка данных...')
    # Разбор строк на отдельные компоненты
    df = pd.read_csv(data)

    # Удаление строк с отсутствующими значениями (NaN)
    df.dropna(inplace=True)

    # Преобразование временных меток в числовой формат
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].map(datetime.timestamp)

    # Преобразование IP-адресов в числовые значения
    def ip_to_int(ip):
        return int(ipaddress.ip_address(ip))

    df['src_ip'] = df['src_ip'].astype(str)
    df['dst_ip'] = df['dst_ip'].astype(str)

    df['src_ip'] = df['src_ip'].apply(ip_to_int)
    df['dst_ip'] = df['dst_ip'].apply(ip_to_int)

    # Преобразование категориальных признаков в числовые значения
    label_encoder = LabelEncoder()
    df['protocol'] = label_encoder.fit_transform(df['protocol'])

    # Нормализация данных
    scaler = StandardScaler()
    df[['time', 'src_ip', 'dst_ip', 'length']] = scaler.fit_transform(df[['time', 'src_ip', 'dst_ip', 'length']])

    # Преобразование DataFrame в массив numpy
    normalized_data = df.values
    return normalized_data

def train_test(data):
    print('Разделение данных на выборки...')
    # Определение размера выборки
    total_samples = len(data)
    train_size = int(0.7 * total_samples)  # 70% данных для обучения
    test_size = total_samples - train_size # 30% данных для тестирования

    print(f'Total samples: {total_samples}')
    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')

    # Разделение данных на обучающую и тестовую выборки
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    return train_data, test_data

# Архитектура автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, encod_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encod_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encod_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def saveModel(model): 
    path = "./AeModel.pth" 
    torch.save(model.state_dict(), path) 

# Обучение модели
def train(model, num_epochs, train_loader):
    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    min_loss = 0.6

    print("Идет обучение...")
    for epoch in range(1, num_epochs+1):
        running_train_loss = 0.0

        # Обучающий цикл
        for batch in train_loader:
            inputs = batch[0]  # получение входных данных; batch - список инпутов [inputs]
            optimizer.zero_grad()   # обнуляем градиент, чтобы не накапливался и не тормозил процесс обучения
            outputs = model(inputs)   # предсказания модели
            train_loss = criterion(outputs, inputs)   # подсчитать потери для прогнозируемого выхода
            train_loss.backward()   # обратная передача потерь
            optimizer.step()        # настройка параметров на основе рассчитанных градиентов
            running_train_loss +=train_loss.item()  # отследить величину потерь

        # Подсчитать величину потерь при обучении
        train_loss_value = running_train_loss/len(train_loader)

        # Обновляем минимальные потери
        if train_loss_value < min_loss:
            min_loss = train_loss_value
        
        # Определяем порог потерь как 100% выше минимальных потерь
        loss_threshold = min_loss * 2

        # Сохраняем модель, если средние потери ниже порога
        if train_loss_value < loss_threshold:
            saveModel(model)

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {train_loss_value:.4f}')

# Тестирование модели
def test(model, test_loader):
    # Загружаем модель, которую сохранили в конце цикла обучения 
    path = "AeModel.pth" 
    model.load_state_dict(torch.load(path, weights_only=True)) 

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0]
            outputs = model(inputs)
            errors = torch.mean((outputs - inputs) ** 2, dim=0)  # Усредняем ошибку по признакам
            errors = errors.cpu().numpy().flatten()
     
        print('Средняя квадратичная ошибка (MSE) при тестировании', errors)

# Конвертируем в ONNX 
def convert(model, input_dim): 

    # переводим модель в режим вывода 
    model.eval() 

    # создадим фиктивный входной тензор  
    dummy_input = torch.randn(1, input_dim, requires_grad=True)  

    # Экспортируем модель   
    torch.onnx.export(model,         # запускаемая модель 
         dummy_input,  
         "Autoencoder.onnx",       # куда модель конвертируется  
         export_params=True,  # хранилище обученных весов параметров в файле модели 
         opset_version=11,    # версия ONNX для экспорта модели 
         do_constant_folding=True,  # нужно ли выполнять складывание констант для оптимизации 
         input_names = ['input'],
         output_names = ['output'],
         dynamic_axes={'input' : {0 : 'batch_size'},    # оси переменной длины 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Модель переконвертирована в ONNX ✓') 

def main():
    normalized_data = preprocess_data()
    train_df, test_df = train_test(normalized_data)

    # Преобразование данных в тензоры
    train_data = torch.tensor(train_df, dtype=torch.float32)
    test_data = torch.tensor(test_df, dtype=torch.float32)

    # Создание DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Параметры модели
    input_dim = train_df.shape[1]  # Количество признаков в данных
    encoding_dim = input_dim // 2
    encod_dim = encoding_dim // 2

    # Создание модели
    model = Autoencoder(input_dim, encoding_dim, encod_dim)

    num_epochs = int(input('Введите количество эпох: '))
    train(model, num_epochs, train_loader)
    test(model, test_loader)
    convert(model, input_dim)

if __name__ == "__main__":
    main()
