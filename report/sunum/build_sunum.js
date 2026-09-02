// EdgeTAM thermal tracking sunumu -- ASELSAN 2024 Portal sunum kural setine gore
// (Segoe UI Black basliklar, Calibri govde, lacivert/turuncu palet, n/N sayfa, gizlilik derecesi).
const pptxgen = require("pptxgenjs");
const path = require("path");

const IMG = (n) => path.join(__dirname, "img", n);

const NAVY = "002D7A", ORANGE = "F7931D", GREY = "4D4D4D", ACCENT = "FF5620";
const INK = "1A1A1A", MUTED = "7F7F7F", TINT = "F2F4F8", TINT2 = "FFF3E6", WHITE = "FFFFFF";
const F_TITLE = "Segoe UI Black", F_SUB = "Segoe UI Semibold", F_BODY = "Calibri", F_LABEL = "Segoe UI";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Yiğit Kaya Bağcı";
pres.title = "EdgeTAM Havadan Termal Tracking";

const W = 13.333, L = 0.6, CW = 12.13;
const SECTIONS = [
  "Problem, SAM ve EdgeTAM",
  "Jetson AGX Orin: TensorRT ve Çözünürlük",
  "Veri: Mask İhtiyacı ve Dataset'ler",
  "Eğitim ve Sonuçlar",
  "Bonus: RGB ve Anti-UAV410",
  "Sonuç ve Sonraki Adımlar",
];
const TOTAL = 29;
let page = 0;

function footer(slide) {
  page += 1;
  slide.addText("ASELSAN ÖZEL", { x: L, y: 7.02, w: 3, h: 0.3, fontFace: F_LABEL, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  slide.addText(`${page}/${TOTAL}`, { x: W - 2.2, y: 7.02, w: 1.6, h: 0.3, fontFace: F_LABEL, fontSize: 10, color: MUTED, align: "right", margin: 0, isTextBox: true });
}

function header(slide, title, subtitle) {
  slide.addText(title, { x: L, y: 0.32, w: CW, h: 0.72, fontFace: F_TITLE, fontSize: 30, color: NAVY, margin: 0, isTextBox: true, valign: "middle" });
  if (subtitle) slide.addText(subtitle, { x: L, y: 1.0, w: CW, h: 0.42, fontFace: F_SUB, fontSize: 18, color: ORANGE, margin: 0, isTextBox: true, valign: "middle" });
}

function content(title, subtitle) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, title, subtitle);
  footer(s);
  return s;
}

function bullets(slide, items, o) {
  const runs = items.map((t, i) => ({
    text: t,
    options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 7 },
  }));
  slide.addText(runs, {
    x: o.x, y: o.y, w: o.w, h: o.h, fontFace: F_BODY, fontSize: o.size ?? 17, color: INK,
    valign: "top", margin: 0, isTextBox: true, lineSpacingMultiple: 1.0,
  });
}

function label(slide, text, x, y, w, opts = {}) {
  slide.addText(text, { x, y, w, h: opts.h ?? 0.36, fontFace: F_LABEL, bold: true, fontSize: opts.size ?? 16, color: opts.color ?? NAVY, margin: 0, isTextBox: true, valign: "middle" });
}

function para(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { x, y, w, h, fontFace: F_BODY, fontSize: opts.size ?? 15, color: opts.color ?? INK, margin: 0, isTextBox: true, valign: opts.valign ?? "top", align: opts.align ?? "left", italic: !!opts.italic });
}

function bigNumber(slide, value, caption, x, y, w, opts = {}) {
  slide.addText(value, { x, y, w, h: 0.95, fontFace: F_TITLE, fontSize: opts.size ?? 54, color: opts.color ?? NAVY, margin: 0, isTextBox: true, valign: "bottom", align: opts.align ?? "left" });
  slide.addText(caption, { x, y: y + 0.98, w, h: 0.6, fontFace: F_BODY, fontSize: 15, color: GREY, margin: 0, isTextBox: true, valign: "top", align: opts.align ?? "left" });
}

function tintBox(slide, x, y, w, h, fill = TINT) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.12 });
}

function node(slide, text, x, y, w, h, opts = {}) {
  const fill = opts.fill ?? NAVY;
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
  slide.addText(text, { x, y, w, h, fontFace: F_LABEL, bold: true, fontSize: opts.size ?? 13, color: opts.color ?? WHITE, align: "center", valign: "middle", margin: 4, isTextBox: true });
}

function arrow(slide, x1, y1, x2, y2, color = GREY) {
  slide.addShape(pres.ShapeType.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color, width: 1.75, endArrowType: "triangle" } });
}

function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.32, fontFace: F_LABEL, fontSize: 11, color: MUTED, margin: 0, isTextBox: true, italic: true });
}

const cellH = (t, o = {}) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontFace: F_LABEL, fontSize: o.size ?? 13, align: o.align ?? "left", valign: "middle" } });
const cell = (t, o = {}) => ({ text: t, options: { color: o.color ?? INK, bold: !!o.bold, fontFace: F_BODY, fontSize: o.size ?? 14, align: o.align ?? "left", valign: "middle", fill: o.fill ? { color: o.fill } : undefined } });

function table(slide, rows, o) {
  slide.addTable(rows, {
    x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH ?? 0.36,
    border: { type: "solid", color: "D9D9D9", pt: 0.5 }, margin: [3, 6, 3, 6],
    autoPage: false,
  });
}

// ---------------------------------------------------------------- 1. Kapak
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText("ASELSAN", { x: L, y: 0.45, w: 3, h: 0.5, fontFace: F_TITLE, fontSize: 20, color: NAVY, margin: 0, isTextBox: true });
  s.addText("Havadan Termal Görüntüde\nPromptable Object Tracking", { x: L, y: 1.9, w: 11.5, h: 1.9, fontFace: F_TITLE, fontSize: 44, color: NAVY, margin: 0, isTextBox: true, valign: "top" });
  s.addText("EdgeTAM: Jetson AGX Orin optimizasyonu ve thermal fine-tuning", { x: L, y: 3.95, w: 11.5, h: 0.55, fontFace: F_SUB, fontSize: 24, color: ORANGE, margin: 0, isTextBox: true });
  s.addText("Yiğit Kaya Bağcı", { x: L, y: 5.4, w: 6, h: 0.45, fontFace: F_BODY, bold: true, fontSize: 24, color: INK, margin: 0, isTextBox: true });
  s.addText("Eylül 2026", { x: L, y: 5.9, w: 6, h: 0.4, fontFace: F_BODY, fontSize: 18, color: GREY, margin: 0, isTextBox: true });
  page += 1;
  s.addText(`${page}/${TOTAL}`, { x: W - 2.2, y: 7.02, w: 1.6, h: 0.3, fontFace: F_LABEL, fontSize: 10, color: MUTED, align: "right", margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- Gündem
function agenda(active) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText(String(active), { x: L, y: 1.2, w: 2.6, h: 1.6, fontFace: F_TITLE, fontSize: 66, color: ORANGE, margin: 0, isTextBox: true, valign: "middle" });
  s.addText("gündem", { x: L, y: 2.8, w: 2.6, h: 0.5, fontFace: F_SUB, fontSize: 20, color: GREY, margin: 0, isTextBox: true });
  const runs = SECTIONS.map((t, i) => ({
    text: `${i + 1}   ${t.toLocaleUpperCase("tr-TR")}`,
    options: { color: i + 1 === active ? NAVY : "BFBFBF", bold: true, breakLine: i < SECTIONS.length - 1, paraSpaceAfter: 14 },
  }));
  s.addText(runs, { x: 3.9, y: 1.3, w: 8.8, h: 4.8, fontFace: F_LABEL, fontSize: 20, margin: 0, isTextBox: true, valign: "top" });
  footer(s);
}

agenda(1);

// ---------------------------------------------------------------- 3. Problem ve hedef
{
  const s = content("Problem ve Hedef", "Prompt ile seçilen hedefi termal videoda gerçek zamanlı takip etmek");
  bullets(s, [
    "Operatör box / point verir; model hedefi segment edip takip eder",
    "Class-agnostic: hedefin ne olduğu sorulmaz",
    "Platform: Jetson AGX Orin, thermal kamera (640×512, 1280×720)",
    "SAM 2 Orin'de ~1 FPS → EdgeTAM (SAM 2'nin edge sürümü)",
    "Stock EdgeTAM RGB video ile eğitildi; thermal alan dışı",
  ], { x: L, y: 1.7, w: 7.4, h: 4.6, size: 18, gap: 10 });
  tintBox(s, 8.5, 1.6, 4.23, 4.9);
  bigNumber(s, "40 ms", "frame başına bütçe (≥ 25 FPS)", 8.9, 1.9, 3.6);
  bigNumber(s, "13,9 M", "EdgeTAM parametre sayısı", 8.9, 4.0, 3.6);
}

// ---------------------------------------------------------------- 4. SAM dense prediction
{
  const s = content("SAM Nasıl Çalışır: Dense Prediction", "Her piksel için karar → eğitim de piksel düzeyinde mask ister");
  // diagram
  const y = 1.75;
  node(s, "Görüntü", 0.6, y + 0.35, 1.6, 0.7, { fill: GREY });
  arrow(s, 2.25, y + 0.7, 2.7, y + 0.7);
  node(s, "Image Encoder\n(RepViT + FPN)", 2.75, y + 0.2, 2.3, 1.0);
  arrow(s, 5.1, y + 0.7, 5.55, y + 0.7);
  node(s, "Feature map\n256 kanal, stride 16", 5.6, y + 0.2, 2.3, 1.0, { fill: "5B6B8C" });
  arrow(s, 7.95, y + 0.7, 8.4, y + 0.7);
  node(s, "Mask Decoder", 8.45, y + 0.2, 2.1, 1.0);
  arrow(s, 10.6, y + 0.7, 11.05, y + 0.7);
  node(s, "Mask", 11.1, y + 0.35, 1.6, 0.7, { fill: ORANGE });
  node(s, "Prompt: box / point", 5.6, y + 1.75, 2.3, 0.6, { fill: ORANGE });
  arrow(s, 7.95, y + 2.05, 9.5, y + 2.05);
  arrow(s, 9.5, y + 2.05, 9.5, y + 1.25);
  s.addText("Prompt Encoder", { x: 8.0, y: y + 2.1, w: 1.5, h: 0.3, fontFace: F_LABEL, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  bullets(s, [
    "Encoder bir kez çalışır; decoder her prompt için ucuz",
    "Loss: 20·focal + dice + IoU-L1, tamamı mask üzerinde",
    "Box etiketi mask'e çevrilmeden eğitim sinyali olamaz",
  ], { x: L, y: 4.5, w: 7.2, h: 1.9, size: 17, gap: 8 });
  tintBox(s, 8.2, 4.4, 4.53, 2.0, TINT2);
  label(s, "Sonuç", 8.5, 4.55, 4.0, { color: ACCENT });
  para(s, "Aerial thermal setlerin çoğu box-only. Dataset'i kullanmak için önce mask üretmek gerekti.", 8.5, 4.95, 4.0, 1.4, { size: 15 });
}

// ---------------------------------------------------------------- 5. SAM 2 -> EdgeTAM
{
  const s = content("SAM 2 → EdgeTAM: Memory ve Perceiver", "Videoda hafıza; edge için hafızayı sıkıştırma");
  bullets(s, [
    "SAM 2: memory bank + memory attention + memory encoder",
    "Her memory frame 4096 token → attention maliyeti H²W² ile büyür",
    "EdgeTAM: 2D Spatial Perceiver, frame başına 512 latent token",
    "Attention 4 → 2 layer, backbone RepViT-M1",
    "Knowledge distillation: SAM 2 teacher, EdgeTAM student",
  ], { x: L, y: 1.65, w: 6.6, h: 3.2, size: 17, gap: 8 });
  bigNumber(s, "8×", "daha az memory token (4096 → 512)", L, 4.95, 5.5);
  label(s, "Frame başına FLOP dağılımı (1024², gerçek checkpoint)", 7.45, 1.62, 5.3, { size: 14 });
  table(s, [
    [cellH("Modül"), cellH("GFLOP", { align: "right" }), cellH("Pay", { align: "right" }), cellH("Param (M)", { align: "right" })],
    [cell("image encoder"), cell("38,7", { align: "right" }), cell("%26,8", { align: "right" }), cell("4,92", { align: "right" })],
    [cell("memory attention", { bold: true, color: ACCENT }), cell("89,3", { align: "right", bold: true, color: ACCENT }), cell("%61,9", { align: "right", bold: true, color: ACCENT }), cell("2,96", { align: "right" })],
    [cell("memory encoder + Perceiver"), cell("12,6", { align: "right" }), cell("%8,8", { align: "right" }), cell("1,62", { align: "right" })],
    [cell("SAM head"), cell("3,6", { align: "right" }), cell("%2,5", { align: "right" }), cell("4,41", { align: "right" })],
  ], { x: 7.45, y: 2.05, w: 5.28, colW: [2.28, 1.0, 1.0, 1.0], rowH: 0.42 });
  tintBox(s, 7.45, 4.45, 5.28, 1.9, TINT2);
  label(s, "Darboğaz image encoder değil", 7.75, 4.6, 4.8, { color: ACCENT });
  para(s, "Parametrelerin %21'i olan memory attention işin %62'si. Bu yüzden tek modül değil dört modül birden hızlandırıldı.", 7.75, 5.0, 4.8, 1.3, { size: 15 });
}

agenda(2);

// ---------------------------------------------------------------- 7. TensorRT 4 engine
{
  const s = content("TensorRT: Dört Engine, fp16, CUDA Graph", "Engine sınırları Python'un araya girdiği yerlerde");
  const y = 1.75;
  node(s, "image\nencoder", 0.6, y, 2.2, 0.95);
  node(s, "Python:\nmemory bank dict", 3.05, y + 0.08, 2.0, 0.8, { fill: "BFBFBF", color: INK, size: 11 });
  node(s, "memory\nattention", 5.3, y, 2.2, 0.95);
  node(s, "SAM head", 7.75, y, 2.2, 0.95);
  node(s, "memory\nencoder", 10.2, y, 2.2, 0.95);
  arrow(s, 2.82, y + 0.48, 3.03, y + 0.48);
  arrow(s, 5.07, y + 0.48, 5.28, y + 0.48);
  arrow(s, 7.52, y + 0.48, 7.73, y + 0.48);
  arrow(s, 9.97, y + 0.48, 10.18, y + 0.48);
  node(s, "Python: mask + pointer bankaya yazılır", 5.3, y + 1.2, 7.1, 0.5, { fill: "BFBFBF", color: INK, size: 11 });
  arrow(s, 11.3, y + 0.97, 11.3, y + 1.18);
  bullets(s, [
    "ONNX trace: dict, .get(), if grafiğe girmez → engine orada biter",
    "Memory bank sabit 7 slot + additive attention mask",
    "fp16 seçildi: fp32'ye karşı IoU 0,9999 (bf16 0,9997)",
    "CUDA Graph replay: kernel launch jitter'ı kaldırır",
    "Tek engine'e fusion kazancı ölçüldü: 0,217 ms (%0,5) → yapılmadı",
  ], { x: L, y: 3.75, w: 12.1, h: 2.9, size: 17, gap: 8 });
}

// ---------------------------------------------------------------- 8. Orin sonuçları
{
  const s = content("Orin Sonuçları: 3,34× Hızlanma", "500 frame, 1024², tek nesne, TensorRT 10.3, fp16");
  tintBox(s, L, 1.6, 12.13, 2.0);
  bigNumber(s, "26,4 ms", "frame süresi, TensorRT + CUDA Graph", 0.95, 1.7, 3.8);
  bigNumber(s, "3,34×", "stock PyTorch 88,3 ms'ye göre", 4.95, 1.7, 3.6);
  bigNumber(s, "0,9993", "mask IoU vs fp32, 500 frame, drift yok", 8.75, 1.7, 3.8);
  table(s, [
    [cellH("Backend"), cellH("ms / frame", { align: "right" }), cellH("FPS", { align: "right" }), cellH("p50 / p90 / p99 (ms)", { align: "right" })],
    [cell("Stock PyTorch (bf16)"), cell("88,30", { align: "right" }), cell("11,3", { align: "right" }), cell("84,4 / 101,3 / 109,1", { align: "right" })],
    [cell("TensorRT fp16 + CUDA Graph", { bold: true }), cell("26,40", { align: "right", bold: true }), cell("37,9", { align: "right", bold: true }), cell("25,0 / 30,8 / 39,9", { align: "right", bold: true })],
  ], { x: L, y: 3.95, w: 12.13, colW: [4.4, 2.2, 1.8, 3.73], rowH: 0.44 });
  bullets(s, [
    "p99 da 40 ms bütçesinin altında; frame düşüren kuyruktur, ortalama değil",
    "CUDA Graph kapalı p99 47,2 ms → açık 37,7 ms; jitter bandı yarıya iner",
  ], { x: L, y: 5.5, w: 12.1, h: 1.2, size: 16, gap: 6 });
}

// ---------------------------------------------------------------- 9. Preprocessing
{
  const s = content("Preprocessing: Model Kadar Pahalıydı", "Ölçülünce 1024'te ~30 ms çıktı; yeniden yazılınca 4,7× ucuzladı");
  table(s, [
    [cellH("Model girdisi"), cellH("Önce", { align: "right" }), cellH("Sonra", { align: "right" }), cellH("Kazanç", { align: "right" })],
    [cell("1024² (1280×720 kaynak)"), cell("42,8 ms", { align: "right" }), cell("9,2 ms", { align: "right", bold: true }), cell("4,7×", { align: "right", bold: true, color: ACCENT })],
    [cell("512² (1280×720 kaynak)"), cell("18,9 ms", { align: "right" }), cell("7,4 ms", { align: "right", bold: true }), cell("2,6×", { align: "right", bold: true, color: ACCENT })],
    [cell("512² center crop, resize yok"), cell("5,2 ms", { align: "right" }), cell("5,4 ms", { align: "right" }), cell("—", { align: "right" })],
  ], { x: L, y: 1.65, w: 7.2, colW: [3.4, 1.2, 1.2, 1.4], rowH: 0.44 });
  bullets(s, [
    "PIL resize → cv2.resize: 30× daha hızlı",
    "uint8 / 255.0 float64'e yükseltiyordu → float32 in-place: 7×",
    "Dosya decode (libtiff ~34 ms) bütçe dışı: kamera frame'i bellekte verir",
    "Doğruluk maliyeti yok: eski/yeni mask IoU 0,9993",
  ], { x: L, y: 3.75, w: 7.2, h: 2.9, size: 17, gap: 8 });
  tintBox(s, 8.2, 1.65, 4.53, 4.95, TINT2);
  label(s, "Frame bütçesi", 8.5, 1.8, 4.0, { color: ACCENT });
  para(s, "pre: crop + resize + normalize\ninference: 4 engine + bookkeeping\npost: mask'i kaynak çözünürlüğe geri ölçekleme", 8.5, 2.25, 4.0, 1.8, { size: 15 });
  para(s, "Bütçe dışı: dosya okuma/decode, overlay çizimi, mp4 kodlama. Ölçülür, raporlanır, sayılmaz.", 8.5, 4.2, 4.0, 1.6, { size: 15, color: GREY });
}

// ---------------------------------------------------------------- 10. Çözünürlük nasıl değişiyor
{
  const s = content("Çözünürlük Nasıl Değişiyor: 512 / 768 / 1024", "Checkpoint 1024'te eğitildi; girdi boyutu tek config alanı");
  bullets(s, [
    "RepViT backbone + FPN neck fully convolutional",
    "Positional encoding her frame hesaplanır → image_size serbest",
    "Tek istisna: memory attention RoPE tablosu (q_sizes) → override ile image_size / 16",
    "Koşul: stride-2 zincirleri tam bölünmeli, token sayısı tam kare",
    "768 = 3·2⁸ geçer; 720 geçmez (720 / 16 = 45)",
  ], { x: L, y: 1.65, w: 6.2, h: 3.4, size: 16, gap: 8 });
  table(s, [
    [cellH(""), cellH("512", { align: "right" }), cellH("768", { align: "right" }), cellH("1024", { align: "right" })],
    [cell("feature map"), cell("32×32", { align: "right" }), cell("48×48", { align: "right", bold: true }), cell("64×64", { align: "right" })],
    [cell("spatial token / frame"), cell("1 024", { align: "right" }), cell("2 304", { align: "right", bold: true }), cell("4 096", { align: "right" })],
    [cell("SAM head high-res (stride 8, 4)"), cell("64, 128", { align: "right" }), cell("96, 192", { align: "right", bold: true }), cell("128, 256", { align: "right" })],
    [cell("memory tarafı (k_sizes)"), cell("16×16", { align: "right" }), cell("16×16", { align: "right", bold: true }), cell("16×16", { align: "right" })],
    [cell("1024'e göre uzamsal iş"), cell("%25", { align: "right" }), cell("%56", { align: "right", bold: true }), cell("%100", { align: "right" })],
  ], { x: 7.0, y: 1.65, w: 5.73, colW: [2.53, 1.0, 1.1, 1.1], rowH: 0.42 });
  tintBox(s, L, 5.2, 12.13, 1.35, TINT2);
  para(s, "Memory bank latent grid sabit kaldığı için sadece image encoder ve self-attention büyür; engine'ler yeniden export edilir, checkpoint aynı.", 0.9, 5.35, 11.6, 1.1, { size: 15 });
}

// ---------------------------------------------------------------- 11. Neden 768
{
  const s = content("Neden 768? Inference ve Detay Dengesi", "Orin, drone kaydı 1280×720, TensorRT fp16 (crop = ortalanmış native pencere)");
  table(s, [
    [cellH("Mod"), cellH("inference ort.", { align: "right" }), cellH("maks", { align: "right" }), cellH("pre + inf + post FPS", { align: "right" })],
    [cell("full1024"), cell("27,6 ms", { align: "right" }), cell("33,8 ms", { align: "right" }), cell("30,0", { align: "right" })],
    [cell("crop1024"), cell("26,4 ms", { align: "right" }), cell("32,0 ms", { align: "right" }), cell("33,3", { align: "right" })],
    [cell("full768", { bold: true }), cell("[Orin ölçümü]", { align: "right", color: ACCENT, bold: true }), cell("[…]", { align: "right", color: ACCENT }), cell("[…]", { align: "right", color: ACCENT })],
    [cell("crop768", { bold: true }), cell("[Orin ölçümü]", { align: "right", color: ACCENT, bold: true }), cell("[…]", { align: "right", color: ACCENT }), cell("[…]", { align: "right", color: ACCENT })],
    [cell("full512"), cell("10,4 ms", { align: "right" }), cell("15,7 ms", { align: "right" }), cell("65,8", { align: "right" })],
    [cell("crop512"), cell("8,1 ms", { align: "right" }), cell("12,2 ms", { align: "right" }), cell("96,2", { align: "right" })],
  ], { x: L, y: 1.65, w: 6.6, colW: [1.5, 1.8, 1.4, 1.9], rowH: 0.4 });
  bullets(s, [
    "512: 720p frame 2,5× küçülür; 6 px hedef 2–3 px olur",
    "1024: 720p'yi upsample eder, 4× token, 40 ms sınırında",
    "768: 720p / 1080p'den native crop, resample yok",
    "768 işi 1024'ün %56'sı; küçük hedef native kalır",
    "Karar: deployment 768; RGB Stage C eğitimi de 768'de",
  ], { x: 7.6, y: 1.65, w: 5.15, h: 3.6, size: 16, gap: 8 });
  tintBox(s, L, 5.3, 12.13, 1.25, TINT2);
  label(s, "Karar", 0.9, 5.4, 2, { color: ACCENT });
  para(s, "768, detay kaybetmeden 25 FPS bütçesinde kalan tek nokta; 512 hız için, 1024 sınırda.", 0.9, 5.8, 11.5, 0.7, { size: 15 });
}

agenda(3);

// ---------------------------------------------------------------- 13. Veri formatı
{
  const s = content("Veri Formatı: Mask Lazım, Elde Box Var", "Aerial thermal setlerin etiket tipi ile SAM'ın eğitim sinyali uyuşmuyor");
  tintBox(s, L, 1.65, 5.85, 4.95);
  label(s, "Ne var", 0.9, 1.8, 5, { color: NAVY, size: 18 });
  bullets(s, [
    "Box: Anti-UAV410, DroneVehicle, HIT-UAV, VTUAV, VisDrone",
    "Semantic map: Kust4K, SegFly (piksel → sınıf)",
    "Instance mask: VTUAV-VIS (100 video), AeroVIS — az ve çoğu RGB",
    "Dense GT olmadan J&F ölçülemez",
  ], { x: 0.9, y: 2.35, w: 5.3, h: 4.0, size: 16, gap: 10 });
  tintBox(s, 6.88, 1.65, 5.85, 4.95, TINT2);
  label(s, "Ne yapıldı", 7.18, 1.8, 5, { color: ACCENT, size: 18 });
  bullets(s, [
    "Teacher SAM: box prompt → mask (pseudo-mask pool)",
    "Semantic map → connected component → instance + box prompt",
    "Çıktı: (image, prompt, mask) üçlüsü, SAM 2'nin kendi tarifi",
    "Drawn mask'li setler eğitime değil ölçüme (role = eval)",
  ], { x: 7.18, y: 2.35, w: 5.3, h: 4.0, size: 16, gap: 10 });
}

// ---------------------------------------------------------------- 14. Dataset taraması
{
  const s = content("Dataset Taraması", "Eleme: indirilebilir mi, aerial mi, etiket tipi, modalite");
  const r = (a, b, c, d, e, o = {}) => [cell(a, { size: 12, bold: true }), cell(b, { size: 12 }), cell(c, { size: 12 }), cell(d, { size: 12, align: "right" }), cell(e, { size: 12, color: o.color ?? INK })];
  table(s, [
    [cellH("Dataset", { size: 12 }), cellH("Modalite / çözünürlük", { size: 12 }), cellH("Etiket", { size: 12 }), cellH("Hacim", { size: 12, align: "right" }), cellH("Kullanım", { size: 12 })],
    r("Kust4K", "RGB-T hizalı, 640×512", "semantic, 9 sınıf", "4 024 çift", "Stage B + drawn eval"),
    r("SegFly", "RGB-T, thermal 640×512", "semantic, 15 sınıf", "15 007 thermal", "Stage B (temizlik sonrası)"),
    r("VTUAV", "RGB-T video, 1920×1080", "box; 100 videoda mask", "~1,7 M çift", "thermal pool + Stage C"),
    r("DroneVehicle", "RGB-T, 640×512", "oriented box", "28 442 çift", "thermal pool (155 K instance)"),
    r("HIT-UAV", "thermal, 640×512", "box", "2 898 kare", "thermal pool"),
    r("Kaggle UAV thermal", "thermal", "box", "13 923 kare", "thermal pool (SAM 3)"),
    r("BIRDSAI", "thermal gece video", "box + track ID", "~62 K kare", "Stage C adayı"),
    r("Anti-UAV410", "thermal video, 640×512", "box + exist", "410 video", "bonus: yerden göğe"),
    r("VisDrone / AeroVIS", "RGB", "box / instance mask", "6 471 kare / 117 video", "RGB pool"),
  ], { x: L, y: 1.6, w: 12.13, colW: [2.0, 2.9, 2.4, 2.0, 2.83], rowH: 0.42 });
  caption(s, "Elenenler: UAV-VisLoc, CrossLoc (GT = GPS), CPD-UAV (küçük), SAM2DV (ücretli, DroneVehicle türevi), LasHeR (224 GB), NII-CU (Dropbox tek akış)", L, 6.05, 12.1);
}

// ---------------------------------------------------------------- 15. Teacher pipeline
{
  const s = content("Teacher ile Pseudo-Mask Havuzu", "Box etiketli setler SAM teacher'dan geçip (image, box, mask) oldu");
  bullets(s, [
    "Teacher: SAM 3 (transformers ≥ 5, gated); yedek SAM 2.1 Hiera-Large",
    "Zoom-crop: küçük hedefte teacher'a kolay problem",
    "4 gate: teacher IoU ≥ 0,7 · box IoU ≥ 0,6 · alan 0,15–1,3 · tek parça ≥ 0,8",
    "Geçemeyen kare havuza girmez; kabul oranı raporlanır",
    "DroneVehicle: %53 aynı polygon → RGB'de bir geçiş, thermal'e aynalama",
  ], { x: L, y: 1.65, w: 6.4, h: 3.8, size: 16, gap: 9 });
  label(s, "Toplanan thermal havuzlar (kabul edilen)", 7.3, 1.62, 5.4, { size: 14 });
  table(s, [
    [cellH("Havuz", { size: 12 }), cellH("kare", { size: 12, align: "right" }), cellH("instance", { size: 12, align: "right" }), cellH("teacher", { size: 12 })],
    [cell("dronevehicle_thermal", { size: 13 }), cell("12 995", { size: 13, align: "right" }), cell("155 180", { size: 13, align: "right", bold: true }), cell("SAM 2.1-L", { size: 13 })],
    [cell("vtuav_thermal", { size: 13 }), cell("38 202", { size: 13, align: "right" }), cell("38 202", { size: 13, align: "right" }), cell("SAM 2.1-L", { size: 13 })],
    [cell("kaggle_uav_thermal", { size: 13 }), cell("13 210", { size: 13, align: "right" }), cell("34 009", { size: 13, align: "right" }), cell("SAM 3", { size: 13 })],
    [cell("hituav_thermal", { size: 13 }), cell("2 865", { size: 13, align: "right" }), cell("23 440", { size: 13, align: "right" }), cell("SAM 2.1-L", { size: 13 })],
    [cell("vtuav_lt_thermal", { size: 13 }), cell("13 265", { size: 13, align: "right" }), cell("13 265", { size: 13, align: "right" }), cell("SAM 3", { size: 13 })],
    [cell("kust4k_thermal", { size: 13 }), cell("2 190", { size: 13, align: "right" }), cell("9 653", { size: 13, align: "right" }), cell("SAM 3", { size: 13 })],
  ], { x: 7.3, y: 2.02, w: 5.43, colW: [2.23, 0.95, 1.15, 1.1], rowH: 0.38 });
  bigNumber(s, "270 K+", "instance, thermal, box'tan üretilmiş", L, 5.35, 6.0, { size: 44 });
  caption(s, "İki teacher aynı sette: havuz başına fark tek nedene bağlanamaz; kayıtta hangi teacher olduğu yazılı.", 7.3, 5.05, 5.4);
}

// ---------------------------------------------------------------- 16. Semantic -> instance
{
  const s = content("Semantic Map → Instance: Kust4K ve SegFly", "SegFly örneği: thermal frame · yayınlanan semantic harita · temiz sette kalan hedefler");
  s.addImage({ path: IMG("uclu_row1.jpg"), x: 0.9, y: 1.55, w: 11.5, h: 3.31 });
  bullets(s, [
    "Semantic map \"bütün arabalar araba\" der; tracker \"şu araba\"yı ister",
    "Things sınıfları → sınıf başına connected component → box prompt + mask",
    "Bitişik araçlar tek bileşen: watershed ile bölme, fill / alan gate",
    "Palet tuzağı: Kust4K id 6 = tree, SegFly id'ler boşluklu → indirilen veride doğrulandı",
  ], { x: L, y: 5.0, w: 12.1, h: 1.9, size: 15, gap: 5 });
}

// ---------------------------------------------------------------- 17. SegFly temizliği
{
  const s = content("SegFly Temizliği: Hayalet ve Kaynaşma", "15 007 thermal kare elden geçti; iki hata sınıfı ölçülüp elendi");
  s.addImage({ path: IMG("hayalet_row1.jpg"), x: L, y: 1.6, w: 8.3, h: 1.365 });
  caption(s, "Hayalet: mask altında araç yok (sol thermal, sağ RGB)", L, 2.97, 8.3);
  s.addImage({ path: IMG("kaynasma_row1.jpg"), x: L, y: 3.4, w: 8.3, h: 1.715 });
  caption(s, "Kaynaşma: iki araç tek hedef, tek box", L, 5.15, 8.3);
  tintBox(s, 9.2, 1.6, 3.53, 5.0, TINT);
  bullets(s, [
    "Hayalet: kenar yoğunluğu ≈ 0 → 1 162 silindi",
    "Kaynaşma: 1 914 bölündü / silindi",
    "truck sınıfı kamyon değil (246 px) → düşürüldü",
    "Kalan: 3 439 kare, 10 751 instance",
    "Bedel: hayalet gate'i ~%12 gerçek aracı da götürdü",
  ], { x: 9.45, y: 1.8, w: 3.1, h: 4.6, size: 14, gap: 9 });
}

// ---------------------------------------------------------------- 18. SegFly kayma
{
  const s = content("SegFly: Mask Kayması Çözülemedi", "Otomatik düzeltme denendi, gözle bakınca geri alındı (üst: önce, alt: sonra)");
  s.addImage({ path: IMG("kayma_once_sonra.jpg"), x: L, y: 1.55, w: 12.13, h: 3.58 });
  bullets(s, [
    "706 karede (%20,5) mask araçtan 25–50 px kaymış; yön rastgele → ortalamada görünmez",
    "Sıcak gövdeye hizalama testi geçti, ama doğru mask'i parlak zemine (beton, çatı) çekti",
    "Şüpheli kareler işaretlendi; gerçek çözüm teacher ile yeniden maskeleme (point prompt)",
  ], { x: L, y: 5.3, w: 12.1, h: 1.6, size: 15, gap: 6 });
}

agenda(4);

// ---------------------------------------------------------------- 20. Eğitim düzeni
{
  const s = content("Eğitim Düzeni: Stage B ve Stage C", "Memory yolu her iki aşamada donuk; encoder ve decoder uyarlanıyor");
  const c = (t, o = {}) => cell(t, { size: 13, ...o });
  table(s, [
    [cellH(""), cellH("Stage B — tek kare"), cellH("Stage C — video")],
    [c("Eğitilen", { bold: true }), c("image encoder + mask decoder"), c("+ object score head; encoder düşük LR")],
    [c("Donuk", { bold: true }), c("memory attention / encoder / Perceiver"), c("aynı: memory koordinatları bozulmasın")],
    [c("Veri", { bold: true }), c("pool + Kust4K + SegFly temiz: 10 982 kare, 40 K instance"), c("VTUAV / VTUAV-VIS 8 frame clip, low-contrast 2× örnekleme")],
    [c("Prompt", { bold: true }), c("mix: box / jitter / point"), c("frame 0 box; sonrası kendi memory'si (teacher forcing yok)")],
    [c("Ek", { bold: true }), c("anchor 0,5, eval() BN donuk, EMA, LR pilotu"), c("temporal low-contrast augment, SAMURAI A/B")],
  ], { x: L, y: 1.65, w: 12.13, colW: [1.6, 5.0, 5.53], rowH: 0.5 });
  tintBox(s, L, 4.9, 12.13, 1.6, TINT2);
  label(s, "Ölçüm disiplini", 0.9, 5.0, 4, { color: ACCENT });
  para(s, "Prompt olarak tam box çoğu mask'i zaten söyler; encoder farkı point prompt'ta görünür. Held-out kareler drawn set'ten, aynı kare iki ayrı kaynaktan eğitime sızmıyor.", 0.9, 5.4, 11.5, 1.0, { size: 15 });
}

// ---------------------------------------------------------------- 21. Stage B sonuçları
{
  const s = content("Stage B: Stock vs Fine-Tune (Thermal)", "25 747 held-out instance, 9 kaynak, test split; tek-kare ölçüm");
  tintBox(s, L, 1.6, 12.13, 1.95);
  bigNumber(s, "+0,14", "point prompt IoU: 0,57 → 0,71", 0.95, 1.7, 3.8);
  bigNumber(s, "+0,07", "box prompt IoU: 0,77 → 0,84", 4.95, 1.7, 3.6);
  bigNumber(s, "%83", "point prompt'ta IoU ≥ 0,5 (stock %63)", 8.75, 1.7, 3.8);
  table(s, [
    [cellH("Prompt"), cellH("Metrik"), cellH("Stock", { align: "right" }), cellH("Fine-tune", { align: "right" }), cellH("Δ", { align: "right" })],
    [cell("box"), cell("mean IoU"), cell("0,771", { align: "right" }), cell("0,838", { align: "right", bold: true }), cell("+0,067", { align: "right", color: ACCENT, bold: true })],
    [cell("box"), cell("IoU ≥ 0,5 oranı"), cell("0,960", { align: "right" }), cell("0,992", { align: "right", bold: true }), cell("+0,032", { align: "right", color: ACCENT, bold: true })],
    [cell("point"), cell("mean IoU"), cell("0,568", { align: "right" }), cell("0,705", { align: "right", bold: true }), cell("+0,137", { align: "right", color: ACCENT, bold: true })],
    [cell("point"), cell("IoU ≥ 0,5 oranı"), cell("0,629", { align: "right" }), cell("0,828", { align: "right", bold: true }), cell("+0,199", { align: "right", color: ACCENT, bold: true })],
    [cell("point"), cell("küçük hedef (< 32 px) IoU"), cell("0,586", { align: "right" }), cell("0,622", { align: "right", bold: true }), cell("+0,036", { align: "right", color: ACCENT, bold: true })],
  ], { x: L, y: 3.85, w: 8.2, colW: [1.2, 3.0, 1.3, 1.4, 1.3], rowH: 0.4 });
  tintBox(s, 9.1, 3.85, 3.63, 2.65, TINT2);
  label(s, "Nasıl okunmalı", 9.35, 3.95, 3.2, { color: ACCENT, size: 15 });
  para(s, "Tam box mask'in çoğunu zaten söyler. Point satırı encoder'ın gerçek kazancı. Hiçbiri tracking ölçümü değil; memory yolu kapalı.", 9.35, 4.35, 3.2, 2.1, { size: 14 });
}

// ---------------------------------------------------------------- 22. Fine-tune kazandırdıkları
{
  const s = content("Fine-Tune'un Kazandırdığı Örnekler", "Point prompt; yeşil = yalnız fine-tune buldu, kırmızı = yalnız stock, mavi = ikisi de");
  s.addImage({ path: IMG("panel_gained.jpg"), x: L, y: 1.55, w: 12.13, h: 2.116 });
  caption(s, "Üstte IoU: stock → fine-tune (fark). Kaynaklar: VTUAV, DroneVehicle, HIT-UAV.", L, 3.72, 12.1);
  bullets(s, [
    "Stock point prompt'ta 0,01–0,13 IoU: hedefi neredeyse hiç bulamıyor",
    "Fine-tune aynı karelerde 0,92–0,96: araç, yaya, sıcak gövde",
    "Held-out 1 327 instance'ın 1 006'sı iyileşti, 239'u geriledi",
  ], { x: L, y: 4.35, w: 12.1, h: 2.2, size: 16, gap: 8 });
}

// ---------------------------------------------------------------- 23. Çözülemeyen case
{
  const s = content("Çözülemeyen Case: Düşük Contrast ve Clutter", "Hedef arka planla aynı tona gelince mask komşuya kayıyor; şekil değil sinyal sorunu");
  s.addChart(pres.ChartType.bar, [
    { name: "Stock", labels: ["Clutter içinde (< 1)", "Benzer ton (1–3)", "Ayrışan (> 3)"], values: [0.462, 0.568, 0.717] },
    { name: "Fine-tune", labels: ["Clutter içinde (< 1)", "Benzer ton (1–3)", "Ayrışan (> 3)"], values: [0.617, 0.710, 0.821] },
  ], {
    x: L, y: 1.5, w: 6.3, h: 3.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["BFBFBF", NAVY], showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valGridLine: { color: "E6E6E6", size: 0.5 },
    catAxisLabelFontSize: 11, catAxisLabelColor: INK, catGridLine: { style: "none" }, catAxisLabelFontFace: F_BODY,
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: INK, showTitle: false,
  });
  caption(s, "Point prompt mean IoU, hedef–halka kontrastı (σ birimi) ile gruplanmış", L, 4.7, 6.3);
  bullets(s, [
    "Küçük hedef + clutter: 0,36 → 0,38, kazanç yok",
    "Fine-tune her grupta artırıyor ama sıra değişmiyor",
    "Video'da aynı kare memory'yi zehirliyor",
    "Yön: contrast-aware augment, distractor-aware memory, daha çok düşük-contrast video",
  ], { x: 7.2, y: 1.6, w: 5.5, h: 3.3, size: 15, gap: 8 });
  s.addImage({ path: IMG("panel_lost.jpg"), x: L, y: 5.05, w: 10.9, h: 1.926 });
}

// ---------------------------------------------------------------- 24. Tracking / SAMURAI
{
  const s = content("Tracking: Memory Zehirlenmesi ve SAMURAI", "VTUAV thermal, yaya: mask → yalnız box → kayıp (YOK); tek zor kare sonrası model toparlamıyor");
  s.addImage({ path: IMG("sheet_ped223.jpg"), x: L, y: 1.5, w: 12.13, h: 2.278 });
  bullets(s, [
    "object_score < 0 → memory'e no_obj_ptr yazılır; sonraki 7 frame onu okur",
    "SAMURAI: Kalman filter ile aday mask re-score (0,15·kf + 0,85·mask) + memory gate",
    "Maliyet ~0: KF numpy'da; TensorRT'de yalnız +0,2 ms",
    "Denendi: stage B / temporal / temporal+SAMURAI A/B → ölçülebilir kazanç alınamadı",
    "Şüphe: gate warm-up'ta kf_score = 0 ile kareleri reddediyor; implementasyon doğrulanmalı",
  ], { x: L, y: 3.95, w: 12.1, h: 2.9, size: 15, gap: 6 });
}

agenda(5);

// ---------------------------------------------------------------- 26. RGB bonus
{
  const s = content("Bonus: RGB'de Orijinal Ağırlıklarla Tracking", "VTUAV-VIS, 768 native crop; bisiklet (194 frame) ve gemi (548 frame), her karede mask");
  s.addImage({ path: IMG("sheet_bike009.jpg"), x: L, y: 1.5, w: 11.7, h: 2.197 });
  s.addImage({ path: IMG("sheet_ship001.jpg"), x: L, y: 3.75, w: 11.7, h: 2.197 });
  bullets(s, [
    "EdgeTAM RGB video (SA-V) ile eğitildi → RGB alan içi, thermal'deki gap yok",
    "1080p'den 768 native crop, resample yok; aynı engine'ler 768'de çalışır",
    "RGB pool ile fine-tune sırada; stock zaten sağlam bir taban",
  ], { x: L, y: 6.02, w: 12.1, h: 0.95, size: 14, gap: 3 });
}

// ---------------------------------------------------------------- 27. Anti-UAV410 bonus
{
  const s = content("Bonus: Anti-UAV410 ile Scope Dışı Deneme", "Yerden göğe thermal; küçük dataset ile Stage C, hedef birkaç piksel");
  tintBox(s, L, 1.6, 6.9, 4.95, "E8EAEF");
  s.addText("Video Alanı", { x: L, y: 3.3, w: 6.9, h: 0.6, fontFace: F_SUB, fontSize: 24, color: GREY, align: "center", margin: 0, isTextBox: true });
  s.addText("Anti-UAV410 takip demosu (fine-tune checkpoint)", { x: L, y: 3.9, w: 6.9, h: 0.5, fontFace: F_BODY, fontSize: 15, color: MUTED, align: "center", margin: 0, isTextBox: true });
  bullets(s, [
    "Normal scope havadan yere; burada kamera yerde, hedef gökyüzünde",
    "410 thermal video, box + exist etiketi; 80 dizi ile eğitim",
    "SAM 2.1-L teacher, 4× zoom-crop ile pseudo-mask (4 gate)",
    "Sonuç: birkaç piksellik hedefte takip başarılı",
    "Mesaj: class-agnostic + sağlam temel → küçük dataset ile yeni scope",
  ], { x: 7.8, y: 1.7, w: 4.95, h: 4.8, size: 16, gap: 10 });
}

// ---------------------------------------------------------------- 28. Sonuç
{
  const s = content("Sonuç ve Sonraki Adımlar");
  tintBox(s, L, 1.4, 5.9, 5.2);
  label(s, "Bulgular", 0.9, 1.55, 5, { size: 18 });
  bullets(s, [
    "Darboğaz memory attention (%62 FLOP); 4 engine + CUDA Graph: 3,34×, p99 < 40 ms",
    "Preprocessing 4,7× ucuzladı; 768 detay / hız dengesi",
    "Box → mask: teacher pool ile 270 K+ thermal instance",
    "Fine-tune: point IoU 0,57 → 0,71; clutter hâlâ zor",
  ], { x: 0.9, y: 2.1, w: 5.35, h: 4.3, size: 15, gap: 10 });
  tintBox(s, 6.83, 1.4, 5.9, 5.2, TINT2);
  label(s, "Sırada", 7.13, 1.55, 5, { size: 18, color: ACCENT });
  bullets(s, [
    "768 engine + fine-tune checkpoint export, Orin ölçümü",
    "INT8: TensorRT Model Optimizer (calibrated PTQ)",
    "Stage C A/B temiz tekrar; SAMURAI gate düzeltmesi, DAM4SAM",
    "SegFly kayma: teacher ile yeniden mask",
    "Daha çok düşük-contrast thermal video",
  ], { x: 7.13, y: 2.1, w: 5.35, h: 4.3, size: 15, gap: 10 });
}

// ---------------------------------------------------------------- 29. Teşekkürler
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText("TEŞEKKÜRLER", { x: L, y: 2.6, w: 12.1, h: 1.2, fontFace: F_TITLE, fontSize: 44, color: NAVY, margin: 0, isTextBox: true, valign: "middle" });
  s.addText("www.aselsan.com", { x: L, y: 3.9, w: 6, h: 0.5, fontFace: F_BODY, fontSize: 18, color: GREY, margin: 0, isTextBox: true });
  footer(s);
}

if (page !== TOTAL) throw new Error(`slide count ${page} != TOTAL ${TOTAL}`);

pres.writeFile({ fileName: path.join(__dirname, "EdgeTAM_Thermal_Tracking_Sunum.pptx") }).then((f) => console.log("wrote", f));
