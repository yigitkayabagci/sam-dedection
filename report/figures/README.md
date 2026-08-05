# Görseller

Bu klasör, raporun "GÖRSEL BURAYA" kutularıyla işaretlenmiş yerlerine
koyacağınız kendi görselleriniz (mimari ekran görüntüleri, `tracked.mp4`
klplerinden alınmış kareler, `frame_output/`'taki `latency.png` /
`stages.png` grafikleri, `results/`'taki `03_speed*.png` vb.) içindir.

Şu an raporda hiçbir yer bir dosyaya bağlı `\includegraphics` çağrısı
yapmıyor -- her görsel önce `\gorselyertutucu{...}` ile işaretli bir kutu
olarak duruyor, böylece görselleri henüz eklemeseniz bile rapor hatasız
derlenir. Bir görseli eklemek için, ilgili `deneyler.tex` içindeki

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

Örnek: `ucak-crop-1024-latency.png`.
