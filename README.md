# 🕸 Anomaly Detector

A program designed to detect anomalous network activity on a host using machine learning techniques.

## ✨ Technologies

- `Scapy`
- `Numpy`
- `Pandas`
- `PyTorch`
- `scikit-learn`

## 🚀 Features

### File `sniff.py`
- Collects network interface traffic as packets and saves them in `.csv` format.
- There is a possibility to set the training sample collection period (24 hours by default).

### File `train.py`
- Transforms the collected sample to a two-dimensional matrix, with each line being a packet.
- Trains a neural network using normalized sampling. Neural network type is `Autoencoder`.
- Adapts the anomaly threshold to the sample based on reconstruction loss.

### File `detector.py`
- Непрерывно собирает тестовую выборку на сетевом интерфейсе и сохраняет в формате `.csv`
- Преобразует и проверяет выборку на предмет аномальности с помощью обученной нейронной сети. 
- Формирует отчет об обнаруженных аномалиях (на консоли и в файле).
- **`НЕЛЬЗЯ`** Можно при запуске программы подгрузить и использовать обученную ранее свою нейросеть.
 
## 📍 The Process

У меня была задача совместить две главные сферы XXI века - cybersecurity и data science. Большинство защитных механизмов уже сделано, но угрозы информационной безопасности развиваются с каждым днем всё сильнее и появляется необходимость внедрения нейронных сетей и LLM в существующие продукты безопасности. Этим проектом я хотела показать, что для выявления аномалий трафика сети, не являющегося размеченным, можно использовать предварительно обученную на данных этого же трафика нейронную сеть.   

## 🚦 Running the Project

1. Установи python_version 3.7-3.9 (у меня python_version==3.8.20)
2. Склонируй репозиторий
3. Установи зависимости в виртуальном окружении: `pip install -r requirements.txt`
4. Запусти скрипт `python3 sniff.py`, введи время в часах, дождись результат. 
5. Запусти скрипт `python3 train.py`, дождись результат.
6. Запусти скрипт `python3 detector.py`. Трафик будет собираться до тех пор, пока не нажать клавищи `ctrl+C`, введи количество эпох, дождись результат.

## 🎞️ Preview

<img src="https://github.com/user-attachments/assets/7e801460-a1c6-4f41-8a90-6e86891565db" width="473.3" height="79"><br/>

<img src="https://github.com/user-attachments/assets/68a9b4b1-782a-41b4-82e0-b818ef9efe8f" width="432.7" height="187.56"><br/>

<img src="https://github.com/user-attachments/assets/f73c33bf-740a-4b26-8ada-ec7ad1c40319" width="795.3" height="443.3"><br/>

<img src="https://github.com/user-attachments/assets/b924f39f-0655-4dde-9876-7343dcb9b729" width="657.3" height="244.3"><br/>

<img src="https://github.com/user-attachments/assets/39e6dccd-a640-459e-a2c0-3f8cf9fbe7e8" width="754.7" height="377.3"><br/>


Открыть модель можно [здесь](https://netron.app/).
