# yuztanimayoklamasistemi
insightface ile canlılık kontrolü ve göz açık olup olmadığını kontrolü yaparak yoklama sistemi

Genel İşleyiş
Bu Python programı, InsightFace kütüphanesi ile yüz tanıma yapıyor, OpenCV ile göz tespiti gerçekleştiriyor ve kişilerin canlılık durumunu kafa hareketleriyle kontrol ederek yoklama (varlık kontrolü) alıyor.

Adım Adım Açıklama:
Yüz Tanıma Modeli Hazırlığı
InsightFace FaceAnalysis modülü CPU üzerinde çalışacak şekilde hazırlanıyor (providers=['CPUExecutionProvider']), kamera görüntülerinde yüz algılamak için hazır hale getiriliyor.

Göz Tespiti için Haar Cascade Yükleniyor
OpenCV’nin hazır haarcascade_eye.xml sınıflandırıcısı yükleniyor. Bu, yüz üzerinde gözlerin yerini belirlemek için kullanılıyor.

Yüz Veritabanı
Daha önce kaydedilmiş yüz vektörlerini ve isimlerini içeren bir dosya (yuz_veritabani.npz) varsa yükleniyor. Yoksa boş bir sözlük oluşturuluyor.

Kosinüs Benzerliği Fonksiyonu
Yüz gömme vektörleri (embedding) arasındaki benzerliği ölçmek için kosinüs benzerliği kullanılıyor. Bu, tanınan yüzün veritabanındaki hangi yüze daha çok benzediğini bulmak için.

Yoklama ve Canlılık Testi için Değişkenler

Aynı kişinin yoklamasının tekrarlanması arasında en az 20 saniye olmalı.

Kafanın yatış hareketi 10 pikselden fazla olmalı (canlılık işareti).

Canlılık testi 5 saniye sürüyor (bu süre boyunca kafa hareketi izleniyor).

Kamera Görüntüsü Alımı ve Döngü
Sürekli olarak kamera görüntüsü okunuyor ve ekranda gösteriliyor. Kullanıcı aşağıdaki tuşlarla etkileşimde bulunabilir:

't': Yüz tanımayı başlat / durdur toggle.

'e': Tanınmayan yüzü kaydetme.

'k': Programı kapatma.

Yüzlerin Algılanması ve Tanınması

Her karede InsightFace ile yüzler tespit ediliyor.

Her yüzün embedding'i çıkarılıyor.

Veritabanındaki kayıtlarla karşılaştırılarak en yüksek benzerlik ve karşılık gelen isim bulunuyor.

Benzerlik %65’in üzerindeyse tanıma başarılı sayılıyor.

Gözlerin Tespiti

Algılanan yüzün gri tonlamalı bölgesinde gözler Haar cascade ile aranıyor.

Göz açık mı kapalı mı olduğu belirleniyor.

Canlılık Testi (Kafa Yatış Hareketi)

Son 5 saniye içinde yüzün yatay konumundaki değişim ölçülüyor.

Eğer hareket 10 pikselden büyükse kişi canlı kabul ediliyor.

Yoklama Alma

Tanınan kişi, gözleri açık ve canlılık testi geçen kişi ise, aynı kişinin yoklaması 20 saniyeden daha önce alınmamışsa yoklama dosyasına (txt) kayıt ekleniyor.

Ekrana yoklama alındığına dair bildirim gösteriliyor.

Yeni Kişi Kaydetme

Eğer yüz tanınmıyorsa ve kullanıcı e tuşuna basarsa, komut satırından ismi girilerek yüz embedding veritabanına ekleniyor.

Arayüz ve Kullanıcı Bilgilendirme

Ekranda hangi tuşların ne işe yaradığı gösteriliyor.

Yüzün kutusu, ismi, benzerlik skoru, göz durumu ve canlılık sonucu yazılıyor.

Özetle:
Kamera açılır.

't' ile yüz tarama başlatılır.

Algılanan yüzler InsightFace ile tanınır.

Gözler Haar cascade ile kontrol edilir.

Kafa yatış hareketi ile canlılık testi yapılır.

Canlı, gözleri açık ve tanınan kişiler için yoklama alınır ve dosyaya kaydedilir.

Yeni yüzler kullanıcı tarafından eklenebilir.

'k' ile program kapatılır.
