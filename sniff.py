import scapy.all as scapy
from datetime import datetime
import csv

OUTPUT_CSV = 'traffic_data.csv'

# Анализ пакетов
def analyze_packet(packet):
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

def main():
    user_duration = input("Введите продолжительность сбора трафика в часах (по умолчанию 24): ")
    if user_duration.strip():  # Проверяем, что ввод не пустой
        duration = int(user_duration) * 3600  # Преобразуем часы в секунды
    else:
        duration = 86400  # 24 часа по умолчанию
    
    print('Ведется сбор трафика...')
    traffic_data = []

    def packet_callback(packet):
        packet_info = analyze_packet(packet)
        traffic_data.append(packet_info)

    try:
        scapy.sniff(prn=packet_callback, timeout=duration) 
    except Exception as e:
        print(f"Ошибка во время перехвата: {e}")
        return

    save_to_csv(traffic_data)
    print(f"Сбор трафика завершен ✓")

if __name__ == "__main__":
    main()