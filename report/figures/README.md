# Görseller

Bu klasör, raporun "GÖRSEL BURAYA" kutularıyla işaretlenmiş yerlerine
koyacağınız kendi görselleriniz (mimari ekran görüntüleri, `tracked.mp4`
klplerinden alınmış kareler, `frame_output/`'taki `latency.png` /
`stages.png` grafikleri, `results/`'taki `03_speed*.png` vb.) içindir.

Bu klasördeki PNG'ler depoya işlenmiyor, ama rapor onlar olmadan da
hatasız derlenir: `preamble.tex` içindeki sarmalayıcı, `\includegraphics`
çağrısının işaret ettiği dosya yoksa otomatik olarak "GÖRSEL BURAYA"
kutusuna düşer ve eksik dosyanın adını yazar. Dosya yerindeyse hiçbir
şey değişmez, gerçek görsel çizilir. Yani derleme hatası yerine, PDF'te
hangi görselin eksik olduğunu gösteren bir kutu görürsünüz.

`segfly-instance-conversion.png` gerçek bir SegFly thermal frame üzerinde semantic map'ten
instance target üretimini gösterir. Aynı basename'e sahip JSON ölçümleri ve LICENSE dosyası
ile birlikte version control altında tutulur. Görsel `tools/analyze_segfly_instances.py`,
Colab akışı ise `notebooks/26_segfly_instance_audit.ipynb` ile yeniden üretilebilir.

Bir görseli eklemek için, ilgili `deneyler.tex` içindeki

```latex
\gorselyertutucu{buraya-neyin-geleceğinin-açıklaması}
```

satırını, bu klasöre koyduğunuz dosyaya işaret eden gerçek bir figürle
değiştirin, örneğin:

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\linewidth]{figures/crop512_kare.png}
  \caption{\texttt{crop512} modunda hedefin kırpma penceresi içinde
    kaldığı bir kare.}
\end{figure}
```

`report/main.tex`'in `\graphicspath` ayarlamadığını unutmayın --
`includegraphics` çağrılarında yolu `figures/dosya.png` şeklinde,
`report/` köküne göre yazın (yukarıdaki örnekteki gibi).

## Dosya adlandırma kuralı

Kayıt/çözünürlük karşılaştırma görselleri şu kalıbı izler:

```
<kaynak>-<çözünürlük>[-crop]-<içerik>
```

- `<kaynak>`: `drone` ya da `ucak`
- `<çözünürlük>`: `1024` ya da `512`
- `-crop`: opsiyonel, ortalanmış kırpma modunu işaretler (yoksa tam kare)
- `<içerik>`: görselin ne gösterdiği (örn. `latency`)

Örnek: `ucak-1024-crop-latency.png`.
