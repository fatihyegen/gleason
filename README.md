# Prostat Kanseri Tespiti – Gleason Skoru ile Derin Öğrenme

> **Not:** Bu proje hâlâ geliştirilmektedir. Kodlar ve dokümantasyon zamanla güncellenecektir.

## 📦 Veri Seti

Veri seti boyut ve telif hakkı nedeniyle bu repoya dahil edilmemiştir. Gerekli görüntü ve maske verilerini yüklemek için:

- `patch_data/` içine orijinal görüntüleri yerleştirin.
- Maskeler için:
      0=Benign (green), 1=Gleason_3 (blue), 2=Gleason_4 (yellow), 3=Gleason_5 (red), 4=unlabelled (white).


## 📝 Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Kullanım, dağıtım ve katkı serbesttir (ayrıntılar için `LICENSE` dosyasına bakabilirsiniz).

## 🙏 Teşekkür

Bu proje, **Dr. Öğr. Üyesi Enes AYAN** danışmanlığında geliştirilmiştir. Destekleri ve yönlendirmeleri için kendisine teşekkür ederim.

---

## 🎯 Proje Amacı

Prostat kanseri, erkeklerde en sık görülen kanser türlerinden biridir. Bu projede amaç, histopatolojik görüntüler üzerinde Gleason skorlarına göre:

- **Benign (iyi huylu)**
- **Gleason 3**
- **Gleason 4**
- **Gleason 5**

alanlarının derin öğrenme ile otomatik segmentasyonudur.

## 🧠 Kullanılan Yöntemler

- **U-Net mimarisi** ile çok sınıflı segmentasyon
- Renk kodlu maskeleme:
  - 🟩 **Yeşil**: Benign
  - 🔵 **Mavi**: Gleason 3
  - 🟡 **Sarı**: Gleason 4
  - 🔴 **Kırmızı**: Gleason 5
- Eğitim için 641 adet histopatolojik görüntü ve renkli maske kullanılmıştır.



