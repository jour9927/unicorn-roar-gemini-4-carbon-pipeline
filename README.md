# 獨角獸那麼叫 Gemini 4 Carbon 無人值守雙軌產線 V3 (`@unicorn-roar`)

完整以 **每秒 1 幀 (`1fps`) 逐幀對比 + 聲學波形級對標** 逆向復刻 YouTube 頻道 **獨角獸那麼叫 (`https://www.youtube.com/@unicorn-roar`)** 之無人值守 AI 影音生成產線（V3 終極版），同時支援 **16:9 六幕式深度解析長片 (`1920x1080@30fps`，12～15 分鐘)** 與 **9:16 滿版舞台導流 Short (`1080x1920@30fps`，1.5～2.5 分鐘)** 一鍵全自動產出並推送至 Telegram。

---

## V3 六大產線站點架構（1fps 逐秒與聲學波形完全對標原片）

```
[1. 無人值守議題雷達 Topic Radar]
   │  自動巡檢台灣食安標示、消費定價、薪資統計、產業政策爭議，比對頻道既有 198 部正片與 106 部 Shorts 去重計分
   ▼
[2. 多源公開資訊全文爬蟲 Web Crawler + Source Citation]
   │  串接台灣新聞全文檢索（Bing News TW RSS 解密真實網址 + HTML 多段正文擷取）、DuckDuckGo .gov.tw 法規公報、
   │  Google News 台灣與維基百科繁中全文，強制主題核心詞過濾並綁定左上角 16px 精緻 accessed:YYYYMMDD 與說明欄來源
   ▼
[3. 11萬字劇本蒸餾庫 + Gemini 4 Hydrogen 零水詞八道關卡產稿站 Scriptwriter] (agy gemini-4-hydrogen-high 主力)
   │  • 內建 25 部真人正片 + 7 部 Shorts（共 110,956 中文字）蒸餾劇本庫 (`assets/script_distillation/`)
   │  • 封殺 AI 過渡水詞，強制開場實拍 + 反應梗片鉤子、純算式黑板與全鏈路 OpenCC s2twp 台灣繁中鎖定
   ▼
[4. 分句級 PyTorch 本人聲音克隆 + 純淨廣播母帶站 Voice & Audio] (Demucs 純淨母帶 + PyTorch F5-TTS)
   │  • 將每幕拆為 18～34 字短句分段推論 F5-TTS，完美保留本人聲線抑揚頓挫與問句後 190ms 自然呼吸氣口
   │  • 字幕時間軸 0.000s 視覺無縫對齊（零閃爍、零詞中斷行），-13.0 LUFS 響度母帶 + 原版 Huh?! 與轉場音效（零廉價合成器底噪）
   ▼
[5. 商業實拍 HD 動態影片 + 反應梗片 + Chrome 滿版實景擷取站 Visuals]
   │  每 2.2～3.4 秒（每分鐘 ~22 刀）高速切換原片真跡視覺：
   │   • 實拍動態 B-Roll   : Mixkit 商業無版權 30fps HD .mp4 實拍影片庫（倒油、賣場、產線、料理、數鈔）
   │   • 招牌反應梗片包    : 開場與吐槽重擊自動切入頻道招牌白貓 Huh? / 打架貓 / 貨架探頭貓等無字幕殘留梗片
   │   • 滿版真實新聞擷取  : Chrome 158 優先直連真實新聞 URL 擷取實景 DOM（或帶實拍配圖之 100% 滿版白底新聞頁）+ 1.50x 紅線推進
   │   • 全國法規資料庫    : 100% 滿版無黑邊之綠標「第 X 條」法規條文頁面 + 紅線推進
   │   • 亮白實體對照板    : 正面宣稱 vs 背面成分真相滿版亮白雙卡 + 鮮紅斜章（徹底廢除深藍底 PPT 框）
   │   • 30fps 滾動留言牆  : 滿版白底 YouTube 台灣鄉民留言區連續平滑向上滾動 (78 px/s)
   │   • 左對齊純算式黑板  : 純 #000000 黑底左對齊白／金／綠／紅遞進算式板（移除底部藍色假尺規）
   ▼
[6. 雙軌廣播級合成與 Telegram 直推站 Composer & Delivery]
   ├── output/long_episode_1080p.mp4        (1920x1080 長片 + 44px 緊湊字幕框 + 完整來源說明欄)
   └── output/short_promo_1080x1920.mp4     (1080x1920 導流 Short：滿寬雙行大字 + 1080x1420 超高滿版舞台)
```

完整逐站原片實測數據與差距診斷請見：[`assets/forensic_gap_audit.md`](assets/forensic_gap_audit.md) 與 [`assets/script_distillation/distilled_playbook.md`](assets/script_distillation/distilled_playbook.md)。

---

## 快速上手（本機優先，斷線零遺失）

### 1. 無人值守全自動產出（自動選題 → 爬蟲附來源 → 32B 產稿 → 本人克隆配音剪氣口 → Chrome/SDXL 產圖 → 長短雙成片）
```bash
./.venv/bin/unicorn-roar run
```

### 2. 指定不同類型主題或自訂題目（重複使用產線生成各類影片）
```bash
# 指定四大核心領域之一（food_labeling | consumer_pricing | wage_labor_stats | industry_policy）
./.venv/bin/unicorn-roar run --pillar food_labeling

# 直接指定任意自訂題目（自動針對該題目爬蟲找公開證據並產出長片 + 導流 Short）
./.venv/bin/unicorn-roar run --topic "掃QR Code點餐還要收10%服務費？台灣餐廳服務費到底服務了誰？"
```

### 3. 執行自動化測試套件（驗證爬蟲浮水印、Chrome DOM 渲染、蒸餾劇本庫與自動剪氣口）
```bash
./.venv/bin/python -c "import tempfile, pathlib; from tests.test_pipeline_components import *; test_source_citation_watermark_format(); test_v2_chrome_renderer_and_kinetic_beats(pathlib.Path(tempfile.mkdtemp())); test_distilled_playbook_and_breath_cutter()"
```

### 4. 下載／同步頻道完整歷史影片庫（含正片與 Shorts）
```bash
./.venv/bin/python scripts/sync_full_channel.py --dest reference_channel/downloads
```

### 5. Tailscale SSH 連線至 `b850` 節點與跨機調度
```bash
# 透過專案內建 Tailscale Mesh 設定直連 b850 (100.100.230.119 / RTX 5070 Ti)
./scripts/b850_ssh.sh "nvidia-smi && ls -la ~/unicorn-roar-gemini4carbon"
```

### 6. 臨時高速節點 Colab RTX PRO 6000 啟動方式
於 Colab 終端機執行單行指令即可啟動隔離式重活 Worker（自動動態偵測可用 VRAM 保護並行任務，與其他 Session 完全隔離）：
```bash
curl -fsSL https://raw.githubusercontent.com/jour9927/unicorn-roar-gemini-4-carbon-pipeline/main/scripts/colab_rtx6000_worker.py | python3 -
```
將輸出的 JSON 存入本機 `config/runtime_endpoints.json`，主產線即自動將 `Qwen2.5-32B` 產稿、PyTorch `F5-TTS` 聲音克隆與 24 步 `SDXL Base 1.0` 產圖優先派發至 RTX PRO 6000，並即時將所有產物拉回本機保存；若臨時節點離線，系統自動無縫降級至本機／Tailscale 節點執行。
