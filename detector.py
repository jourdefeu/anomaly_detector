import numpy as np
import pandas as pd
import onnxruntime as ort
import scapy.all as scapy
from datetime import datetime
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
import ipaddress
import matplotlib.pyplot as plt

MODEL_PATH = 'Autoencoder.onnx'
OUTPUT_CSV = 'network_data.csv'
ANOMALY_REPORT = 'anomaly_report.txt'

# Функция для сбора сетевого трафика
def collect_network_data(packet):
    src_ip = None
    dst_ip = None
    protocol = None

    if packet.haslayer(scapy.IP):
        src_ip = packet[scapy.IP].src
        dst_ip = packet[scapy.IP].dst
        protocol = "IP"
    elif packet.haslayer(scapy.IPv6):
        src_ip = packet[scapy.IPv6].src
        dst_ip = packet[scapy.IPv6].dst
        protocol = "IPv6"

    if packet.haslayer(scapy.TCP):
        protocol += " TCP"
    elif packet.haslayer(scapy.UDP):
        protocol += " UDP"
    elif packet.haslayer(scapy.ICMP):
        protocol += " ICMP"

    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'protocol': protocol,
        'length': len(packet)
    }
    
# Сохранение данных в csv
def save_to_csv(data, filename=OUTPUT_CSV):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"Данные сохранены в файл: {filename}")

# Предобработкa данных
def preprocess_data(data=OUTPUT_CSV):
    print('Обработка данных...')
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

    print('Обработка данных завершена ✓')
    return normalized_data

def find_anomaly(data, ort_session, normalized_data, report_data=ANOMALY_REPORT):
    # Преобразование данных в формат, подходящий для модели
    input_data = normalized_data.astype(np.float32)  # Преобразуем в массив NumPy
    # Выполнение предсказания
    predicted_data = ort_session.run(None, {ort_session.get_inputs()[0].name: input_data})[0]
    
    mse = np.mean(np.square(input_data - predicted_data), axis=1)
    # Установка порога для определения аномалий
    threshold = np.percentile(mse, 95)  # Например, 95-й процентиль
    print('Средняя квадратичная ошибка (MSE) при тестировании', mse)

    # Определение аномалий
    anomalies = mse > threshold

    # Создание отчета об аномалиях
    anomaly_report = data.copy()  # Используем оригинальный DataFrame
    anomaly_report['MSE'] = mse  
    anomaly_report['Anomaly'] = anomalies

    # Сохранение отчета в файл
    anomaly_report.to_csv(report_data, index=False, sep='\t')
    print(f"Отчет об аномалиях сохранен в {report_data}")

    # Визуализация аномалий
    plt.figure(figsize=(12, 6))
    plt.plot(anomaly_report.index, mse, label='MSE', color='lightblue')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Пороговое значение')
    plt.scatter(anomaly_report.index[anomalies], mse[anomalies], color='orange', label='Аномалии', marker='o')
    plt.title('Обнаружение аномалий трафика сети')
    plt.xlabel('Index')
    plt.ylabel('Средняя квадратичная ошибка (MSE)')
    plt.legend()
    plt.show()

    return anomaly_report

def main():
    print('Ведется сбор трафика...')
    traffic_data = []

    def packet_callback(packet):
        packet_info = collect_network_data(packet)
        traffic_data.append(packet_info)

    try:
        # Запускаем бесконечный сбор трафика
        scapy.sniff(prn=packet_callback) 
    except KeyboardInterrupt:
        print("Сбор трафика остановлен пользователем.")
    except Exception as e:
        print(f"Ошибка во время перехвата: {e}")

    print(f"Сбор трафика завершен ✓")
    save_to_csv(traffic_data)

    data = pd.read_csv(OUTPUT_CSV)
    data.dropna(inplace=True)
    normalized_data = preprocess_data()

    ort_session = ort.InferenceSession(MODEL_PATH)
    # Преобразование данных в формат, подходящий для модели

    an_report = find_anomaly(data, ort_session, normalized_data)
    with open(ANOMALY_REPORT, 'r', encoding='utf-8') as file:
        content = file.read()
    print(content)

    return an_report



if __name__ == "__main__":
    main()