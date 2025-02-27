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
- Collects the test sample on network interface nonstop and saves it in `.csv' format
- Converts and checks the sample for anomalies using a trained neural network. 
- Generates a report on detected anomalies (on the console and in a file).
 
## 📍 The Process

I've been on a mission to combine two major areas of the 21st century - cybersecurity and data science. Most of the security mechanisms are already made, but threats to information security are growing daily and there is a need to include neural networks and LLM in existing security products. With this project I wanted to show that it is possible to use a neural network pre-trained on traffic data to detect anomalies in unlabeled network traffic.   

## 🚦 Running the Project

1. Install python_version 3.7-3.9 (I have python_version==3.8.20)
2. Clone the repository
3. Install dependencies in a virtual environment: `pip install -r requirements.txt`
4. Run the `python3 sniff.py` script, enter the time in hours, wait for the result. 
5. Run the `python3 train.py` script, wait for the result.
6. Run the `python3 detector.py` script. Traffic will be collected until you press `ctrl+C`, enter the number of epochs, wait for the result.

## 🎞️ Preview

<img src="https://github.com/user-attachments/assets/7e801460-a1c6-4f41-8a90-6e86891565db" width="473.3" height="79"><br/>

<img src="https://github.com/user-attachments/assets/68a9b4b1-782a-41b4-82e0-b818ef9efe8f" width="432.7" height="187.56"><br/>

<img src="https://github.com/user-attachments/assets/f73c33bf-740a-4b26-8ada-ec7ad1c40319" width="795.3" height="443.3"><br/>

<img src="https://github.com/user-attachments/assets/b924f39f-0655-4dde-9876-7343dcb9b729" width="657.3" height="244.3"><br/>

<img src="https://github.com/user-attachments/assets/39e6dccd-a640-459e-a2c0-3f8cf9fbe7e8" width="754.7" height="377.3"><br/>


You can see the model [here](https://netron.app/).
