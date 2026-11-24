# 獨角獸那麼叫 Gemini 4 Carbon 無人值守雙軌產線 (`@unicorn-roar`)

完整逆向復刻 YouTube 頻道 **獨角獸那麼叫 (`https://www.youtube.com/@unicorn-roar`)** 之無人值守 AI 影音生成產線，同時支援 **16:9 深度解析長片** 與 **9:16 對應導流 Short** 一鍵全自動產出。

## 核心架構亮點

1. **全自動無人值守主題探勘 + 多源公開資料爬蟲（附可驗證來源引用）**
   - 自動巡檢台灣消費爭議、食安法規、標示規範、薪資物價統計、政府公報與公開新聞數據。
   - 每個論點與數據皆綁定原始來源標題、機構、日期與 URL，自動渲染於長片畫面右上/左下資訊來源卡與影片說明欄。
2. **頻道靈魂級 AI 產稿引擎 (`@unicorn-roar` 語氣與敘事結構復刻)**
   - 復刻頻道標誌性的「犀利反問開場 Hook → 法規/商業話術拆解 → 跨國/數據對比 → 業界默契揭露 → 結尾反思」節奏，並同步產出強懸念「回答我 / 導流正片」Short 腳本。
3. **PyTorch 本人聲音克隆配音引擎 (Owner Voice Cloning TTS)**
   - 從頻道正片母帶自動分離人聲、VAD 切分、ASR 校對與信噪比篩選，透過 PyTorch Zero-Shot / Reference Pack 聲音克隆直接合成頻道主本人聲線與語氣停頓，並以 Forced Alignment 精準對齊逐句字幕時間軸。
4. **AI 產圖與動態視覺排版合成引擎 (Long 1920x1080 + Short 1080x1920)**
   - 自動生成符合頻道視覺調性的諷刺插畫／情境圖卡／數據對比圖表，結合動態縮放 (Ken Burns)、醒目描邊繁中字幕、來源浮水印卡片、章節進度與 BGM 自動閃避混音 (Sidechain Ducking)。
5. **本機主控 + 跨節點異質算力調度 (Local-First Master + Remote GPU Offload)**
   - **所有主腳本、設定檔與產物優先保存於本機**，斷線零資料遺失風險。
   - 支援將配音克隆、Whisper 對齊與 AI 產圖重活優先派發至 **Colab RTX PRO 6000 (96GB)** 高速節點，並支援透過 Tailscale SSH 自動切換至 `b850` (RTX 5070 Ti)、`alan-fw` (AI Max 395 128GB) 或本機 Apple Silicon MPS。
