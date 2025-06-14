import insightface
import cv2
import numpy as np
import os
import time
import traceback

# InsightFace yüz tanıma modülü kurulumu
app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Haar cascade ile göz tespiti için sınıflandırıcı
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Yüz veritabanı dosyası
db_dosyasi = "yuz_veritabani.npz"
if os.path.exists(db_dosyasi):
    data = np.load(db_dosyasi, allow_pickle=True)
    kayitli_yuzler = dict(data["arr1"].item())
else:
    kayitli_yuzler = {}

from numpy import dot
from numpy.linalg import norm

def kosinüs_benzerligi(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Yoklama için kullanılan yapılar
yoklama_kayitlari = {}
YOKLAMA_GECIKME = 20      # saniye cinsinden, aynı kişinin tekrarlı yoklaması arasındaki minimum süre
CANLILIK_YAW_ESIGI = 10   # piksel, kafa hareketi için eşik
CANLILIK_SURESI = 5       # canlılık testi süresi (saniye)

# Kamera açılıyor
kamera = cv2.VideoCapture(0)

yuz_tarama_aktif = False
yaw_pozisyonlari = []
yaw_baslangic_zamani = None

while True:
    ret, goruntu = kamera.read()
    if not ret:
        print("Kameradan görüntü alınamıyor!")
        break

    tus = cv2.waitKey(1) & 0xFF

    # 't' tuşu ile yüz tarama aktif/pasif toggle
    if tus == ord('t'):
        yuz_tarama_aktif = not yuz_tarama_aktif
        print(f"Yüz tarama durumu: {'Aktif' if yuz_tarama_aktif else 'Kapalı'}")
        yaw_pozisyonlari = []
        yaw_baslangic_zamani = None

    # 'k' tuşu ile program kapatma
    if tus == ord('k'):
        print("Program kapatılıyor...")
        break

    if not yuz_tarama_aktif:
        cv2.putText(goruntu, "Yuz tarama kapali. 't' tusuna basarak baslatabilirsiniz.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Yuz ve Goz Tanima", goruntu)
        continue

    yuzler = app.get(goruntu)
    gri_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    for yuz in yuzler:
        (x1, y1, x2, y2) = map(int, yuz.bbox)
        embedding = yuz.normed_embedding

        bulunan_isim = "Bilinmiyor"
        max_benzerlik = 0

        # Veritabanı karşılaştırması
        for isim, embed in kayitli_yuzler.items():
            benzerlik = kosinüs_benzerligi(embedding, embed)
            if benzerlik > 0.65 and benzerlik > max_benzerlik:
                max_benzerlik = benzerlik
                bulunan_isim = isim

        # Yüzü kutu içine al ve isim yazdır
        cv2.rectangle(goruntu, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(goruntu, f"{bulunan_isim} ({max_benzerlik:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Göz tespiti
        yuz_gri = gri_goruntu[y1:y2, x1:x2]
        yuz_renkli = goruntu[y1:y2, x1:x2]
        gozler = eye_cascade.detectMultiScale(yuz_gri)
        goz_acik = False
        for (gx, gy, gw, gh) in gozler:
            cv2.rectangle(yuz_renkli, (gx, gy), (gx+gw, gy+gh), (255, 255, 0), 2)
            goz_acik = True

        cv2.putText(goruntu, "Gozler Acik" if goz_acik else "Gozler Kapali", (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if goz_acik else (0, 0, 255), 2)

        # Canlılık testi - kafa yatış hareketi
        if yaw_baslangic_zamani is None:
            yaw_baslangic_zamani = time.time()
            yaw_pozisyonlari = []

        yaw_pozisyonlari.append(yuz.bbox[0])

        gecen_sure = time.time() - yaw_baslangic_zamani
        kalan_sure = max(0, CANLILIK_SURESI - gecen_sure)

        if gecen_sure < CANLILIK_SURESI:
            cv2.putText(goruntu, f"Canlilik testi suruyor: {kalan_sure:.1f} sn", (x1, y2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        # Canlılık testi tamamlandı, kafa hareketini kontrol et
        yaw_farki = max(yaw_pozisyonlari) - min(yaw_pozisyonlari)
        canli_mi = yaw_farki > CANLILIK_YAW_ESIGI

        cv2.putText(goruntu, "Canli: Evet" if canli_mi else "Canli: Hayir", (x1, y2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if canli_mi else (0, 0, 255), 2)

        # Yoklama alma koşulu: tanımlı kişi, canlılık var, göz açık
        simdi = time.time()
        if bulunan_isim != "Bilinmiyor" and canli_mi and goz_acik:
            if bulunan_isim not in yoklama_kayitlari or (simdi - yoklama_kayitlari[bulunan_isim]) > YOKLAMA_GECIKME:
                yoklama_kayitlari[bulunan_isim] = simdi

                try:
                    with open("yoklama_listesi.txt", "a", encoding="utf-8") as dosya:
                        satir = f"{time.strftime('%H:%M:%S')} - {bulunan_isim}\n"
                        dosya.write(satir)
                    print(f"{bulunan_isim} için yoklama alındı.")

                    # Ekranda bildirim göstermek için zaman ve metin kaydet
                    bildirim_baslangic = time.time()
                    bildirim_metni = f"Yoklama alindi: {bulunan_isim}"
                except Exception:
                    print("Yoklama kaydına yazarken hata oluştu:")
                    traceback.print_exc()

        # Bilinmeyen kişiyi kaydetme seçeneği ('e' tuşu ile)
        if bulunan_isim == "Bilinmiyor" and tus == ord('e'):
            yeni_isim = input("Yeni kişinin ismini girin: ")
            kayitli_yuzler[yeni_isim] = embedding
            np.savez(db_dosyasi, arr1=kayitli_yuzler)
            print(f"{yeni_isim} veritabanına eklendi.")

    # Yoklama bildirimi ekranda gösterme (3 saniye)
    if 'bildirim_baslangic' in globals():
        gecen_bildirim = time.time() - bildirim_baslangic
        if gecen_bildirim < 3:
            cv2.putText(goruntu, bildirim_metni, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            del bildirim_baslangic
            del bildirim_metni

    # Kullanıcıya bilgi verme
    cv2.putText(goruntu, "T: Yuz taramayi baslat/durdur", (10, goruntu.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(goruntu, "E: Bilinmeyen kisiyi kaydet (yuz tarama aktifken)", (10, goruntu.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(goruntu, "K: Programi kapat", (10, goruntu.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Yuz ve Goz Tanima", goruntu)

kamera.release()
cv2.destroyAllWindows()
