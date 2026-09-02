# Sunum: Havadan Termal Görüntüde Promptable Object Tracking

`EdgeTAM_Thermal_Tracking_Sunum.pptx`, ASELSAN 2024 Portal sunum kural setine
göre (Segoe UI Black / Semibold başlık, Calibri gövde, lacivert `#002D7A` +
turuncu `#F7931D`, `n/N` sayfa numarası, gizlilik derecesi) `pptxgenjs` ile
üretilir. Drive'daki 9,7 MB'lık şablon dosyası connector üzerinden inmediği
için master/layout yerine kural seti birebir uygulandı; slaytlar gerçek
şablona "Reuse Slides" ile taşınabilir.

## Yeniden üretmek

```bash
cd report/sunum
npm install pptxgenjs@3.12.0
node build_sunum.js          # EdgeTAM_Thermal_Tracking_Sunum.pptx
```

## Doldurulacak yerler

- Slayt 11 (`Neden 768?`): `full768` / `crop768` satırlarındaki
  `[Orin ölçümü]` hücreleri -- 768 engine'lerinin cihaz ölçümü.
- Slayt 27 (`Anti-UAV410`): "Video Alanı" kutusuna takip demosu mp4'ü.
- Slayt 26 (RGB bonus): istenirse Drive'daki
  `edgetam-stage-c/.../preview/*.mp4` klipleri.

## Kaynaklar

Sayılar `docs/EXPERIMENT_LOG.md`, `report/bolumler/*.tex`, Drive'daki
`edgetam-stage-b/aerial_thermal_stable/score_*.json` (stock vs fine-tune,
per-contrast) ve `SegFly_mentor` temizlik raporundan; görseller
`SegFly_mentor/03_gorseller` ve `edgetam-stage-c/*/inspect` sayfalarından
kırpılarak `img/` altına konuldu.
