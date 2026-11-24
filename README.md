# 獨角獸那麼叫 Gemini 4 Carbon 無人值守雙軌產線 V2 (`@unicorn-roar`)

完整逐站逆向復刻 YouTube 頻道 **獨角獸那麼叫 (`https://www.youtube.com/@unicorn-roar`)** 之無人值守 AI 影音生成產線（V2 全站點對標重構版），同時支援 **16:9 六幕式深度解析長片 (`1920x1080@30fps`，12～15 分鐘)** 與 **9:16 對應導流正片 Short (`1080x1920@30fps`，1～3 分鐘)** 一鍵全自動產出。

---

## V2 六大產線站點架構（逐幀逐軌對標原片規格）

```
[1. 無人值守議題雷達 Topic Radar]
   │  自動巡檢台灣食安標示、消費定價、薪資統計、產業政策爭議，比對頻道既有 198 部正片與 106 部 Shorts 去重計分
   ▼
[2. 多源公開資訊全文爬蟲 Web Crawler + Source Citation]
   │  串接台灣新聞全文檢索（Bing News TW RSS 解密真實網址 + HTML 多段正文擷取）、DuckDuckGo .gov.tw 法規公報、
   │  Google News 台灣與維基百科繁中全文，強制主題相關性過濾並自動綁定畫面左上角 accessed:YYYYMMDD 與說明欄來源
   ▼
[3. 11萬字劇本蒸餾庫 + 32B 八道關卡產稿站 Scriptwriter] (RTX PRO 6000 Qwen2.5-32B-Instruct 4-bit)
   │  • 內建 25 部真人正片 + 7 部 Shorts（共 110,956 中文字）蒸餾劇本庫 (`assets/script_distillation/`)
   │  • 6 幕式長篇 RAG 產稿 + 導流 Short 產稿 + 第 8 道「總編去 AI 套話與雙問號標題審查站」+ 5-gram 去重防呆
   ▼
[4. PyTorch 本人聲音克隆 + 自動剪氣口混音站 Voice & Audio] (Demucs 純淨母帶 + PyTorch F5-TTS)
   │  • 依段落情緒自動切換頻道主 4 組高信噪比聲線錨點（開場反問 / 理性拆解 / 吐槽金句 / Short 快節奏）
   │  • 內建「自動剪氣口引擎 (`tight_cut_breaths`)」切除 >95ms 停頓至 35ms，段間 0.04s 緊湊銜接，實測語速 ~316 字/分
   │  • 對齊原片 -13.0 LUFS 廣播響度、立體聲輕快撥弦節奏底樂 (-34.5 dB RMS) 與紅線推進／彈窗／對撞實體音效
   ▼
[5. Chrome 真實網頁／公文 3-Beat 推進 + 24步 SDXL 視覺站 Visuals]
   │  每 2.6～3.5 秒高速切換七大頻道招牌視覺節拍：
   │   • news_portal        : Chrome 158 渲染真實台灣新聞入口網站（全景 1.0x → 1.36x → 1.44x 推進 + 動態紅線塗抹）
   │   • official_gov_doc   : 中華民國正式公文 A4 函釋排版（受文者／發文字號／主旨／說明 + 動態紅線塗抹）
   │   • gov_web_portal     : .gov.tw 政府法規與統計資料庫深藍入口頁（推進特寫 + 動態紅線塗抹）
   │   • product_comparison : 左側包裝話術 vs 右側法規換算真相雙欄對撞卡 + 鮮紅落差徽章
   │   • netizen_comments   : 台灣社群網友留言牆逐則彈出（1 則 → 2 則 → 4 則 + 彈窗音效）
   │   • kinetic_black      : 純黑底黃紅高對比算術公式逐行重擊 + 比例尺揭曉
   │   • ai_scene           : Wikimedia 紀實攝影圖庫 + RTX PRO 6000 24步標準 SDXL Base 1.0 三鏡位快切
   ▼
[6. 雙軌廣播級合成站 Composer]
   ├── output/long_episode_1080p.mp4        (1920x1080 長片：44px 緊湊黑底白字無孤字字幕框 + 完整來源說明欄)
   └── output/short_promo_1080x1920.mp4     (1080x1920 導流 Short：居中超粗 84px 黃紅雙行標題 + 原生乾淨版面)
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
