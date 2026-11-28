# 獨角獸那麼叫 (@unicorn-roar) 無人值守雙軌產線 — 交接文件 (HANDOFF)

## 1. 專案目標與目前狀態（V4 完修穩定版）
本專案為 YouTube 頻道 **「獨角獸那麼叫」(`https://www.youtube.com/@unicorn-roar`)** 1:1 復刻重建之無人值守雙軌影音產線，一鍵自動完成：
1. **自動選題雷達**（`food_labeling` / `consumer_pricing` / `wage_labor_stats` / `industry_policy` 四大支柱）
2. **多源公開資訊與法規爬蟲**（自動附上左上角 `16px` `accessed:YYYYMMDD` 浮水印與 YouTube 說明欄完整來源網址）
3. **六幕式深度文案產稿（主引擎：`Gemini 4 Hydrogen` / `gemini-4-hydrogen-high`）**
   - 依據 25 支參考長片＋7 支參考 Short（共 110,956 字）蒸餾之 `assets/script_distillation/distilled_playbook.json` 撰寫
   - 正片規格：約 `4,000–4,400` 中文字／55–65 幕／`12–14` 分鐘（`1920x1080@30fps`）
   - 對應導流 Short 規格：約 `600–850` 中文字／8–10 幕／`1.8–2.3` 分鐘（`1080x1920@30fps`）
4. **PyTorch 頻道主本人聲音克隆配音（`F5-TTS` V4 台灣語境耳聽精修版）**
   - 台灣語境正音層：`src/unicorn_roar/voice/tw_pronunciation.py`（數字／西元年／持續年數／百分比／破音字「急著／調合油／過份／樂色／鋪光」／外文縮寫全數轉為地道台灣國語口語，畫面字幕則保留清晰繁中與阿拉伯數字）
   - `htdemucs` 母帶分離四組純人聲聲紋錨點：`assets/voice_clone/reference_pack/`（含 `300ms` 句尾靜音隔離帶，零句首滲音；完整句級連續生成，零句內硬切傷子音）
   - 本機 M3 Pro (36GB) Metal `mps` 支援雙進程平行加速合成，並具備遠端 GPU 節點優先派送與斷線無縫自動容錯
5. **中文詞界保護＋口語音節加權字幕斷句器**
   - `src/unicorn_roar/voice/subtitle_segmenter.py`（`rjieba` 分詞＋數字／日期／英文原子鎖定＋口語音節數時間軸對齊，零破詞、零破數字、零單邊殘留引號）
6. **滿版白底證據網頁＋實拍 B-Roll 不放回發牌盒＋配樂音效合成**
   - `src/unicorn_roar/visuals/chrome_cards.py`：Chrome 158 滿版 `#FFFFFF` 台灣新聞／`全國法規資料庫` 紅底線放大特寫（嚴格排除 `<footer>` 頁尾誤截，保留最多 2 次核心證據前後回扣）、首幀完整雙欄對比卡、10 則全不重複台灣網友滾動留言牆
   - `src/unicorn_roar/visuals/video_broll.py` + `src/unicorn_roar/composer/video_composer.py`：75+ 支商業 HD 實拍 B-Roll 不重複發牌盒＋18 張 24-step SDXL 專屬場景圖 30fps Ken Burns 緩推運鏡（保留開場與片尾首尾母題呼應，全片重複重現次數 `13 次` 對齊參考片 `9–20 次` 區間）
   - 混音：`-13.0 LUFS` 人聲＋`-28.6 dB` 室內樂輕撥弦襯底（`investigative_pizzicato_bgm.mp3`）＋原片萃取 `Huh?!` 貓咪迷因與轉場音效

---

## 2. 儲存庫、設備節點與嚴格隔離規範（必讀！）
- **本機主控倉庫（永遠優先寫入本機）**：`/Users/alan_dingchaoliao/Downloads/獨角獸那麼叫`（`alan-hub`，Apple M3 Pro 36GB，macOS）
- **GitHub 遠端倉庫**：`https://github.com/jour9927/unicorn-roar-gemini-4-carbon-pipeline`（分支：`main`）
- **Tailscale SSH 節點 `b850`**：`100.100.230.119`（帳號 `alan-eagle`，路徑 `/home/alan-eagle/unicorn-roar-gemini4carbon`，統一透過 `./scripts/b850_ssh.sh "<cmd>"` 執行）
- **選配遠端高速節點（Colab RTX PRO 6000 96GB）**：設定檔位於 `config/runtime_endpoints.json`（伺服器腳本 `scripts/colab_rtx6000_worker.py`；若通道在線自動優先派送重運算，斷線時自動 fallback 回本機 M3 Pro Metal `mps` 執行）
- **Telegram 成片投遞**：`scripts/send_to_telegram.py`（讀取環境變數 `FIVERR_TG_BOT_TOKEN`，預設傳送至頻道主 `chat_id = 542571922` / `@Oscarlam2016`；**切勿對該 Bot 呼叫 `getUpdates` 以免干擾其他正在運行的 Agent**）
- **嚴格隔離鐵律**：
  1. **嚴禁讀取或參考本機任何其他專案資料夾／其他 Repo**。
  2. 本機、`b850`、`alan-fw` 及瀏覽器可能有其他 Session 正在執行工作，**嚴禁殺死、修改或干擾非本專案之背景程序與 GPU 任務**。
  3. 所有主腳本一律先保存在本機 `/Users/alan_dingchaoliao/Downloads/獨角獸那麼叫`，再推送 GitHub 與同步 `b850`。

---

## 3. 常用操作指令速查

```bash
cd /Users/alan_dingchaoliao/Downloads/獨角獸那麼叫

# 1. 無人值守一鍵產出新集數（16:9 長片 + 9:16 導流 Short + SRT 字幕 + 說明欄來源）
# 可選 --pillar: food_labeling | consumer_pricing | wage_labor_stats | industry_policy
# 可選 --topic "自訂題目"（留空即全自動爬蟲選題）
.venv/bin/unicorn-roar run --pillar food_labeling --scale 0.95 --run-id ep07_auto_run

# 2. 產出完成後一鍵打包並傳送至頻道主 Telegram (chat_id=542571922)
source ~/.zshrc 2>/dev/null || true
source ~/.zshenv 2>/dev/null || true
.venv/bin/python scripts/send_to_telegram.py --run-dir runs/ep07_auto_run

# 3. 執行自動化單元與整合測試（共 5 項核心驗證）
.venv/bin/python -c '
import tempfile
from pathlib import Path
from tests.test_pipeline_components import (
    test_source_citation_watermark_format,
    test_v2_chrome_renderer_and_kinetic_beats,
    test_distilled_playbook_and_breath_cutter,
    test_taiwan_tts_pronunciation_normalizer,
    test_linguistic_subtitle_segmenter_no_broken_words,
)
test_source_citation_watermark_format()
with tempfile.TemporaryDirectory() as td:
    test_v2_chrome_renderer_and_kinetic_beats(Path(td))
test_distilled_playbook_and_breath_cutter()
test_taiwan_tts_pronunciation_normalizer()
test_linguistic_subtitle_segmenter_no_broken_words()
print("ALL_5_UNIT_TESTS_PASSED")
'

# 4. 同步 GitHub 與 Tailscale b850 節點
git push origin main && ./scripts/b850_ssh.sh "cd ~/unicorn-roar-gemini4carbon && git pull --ff-only"
```

---

## 4. 最新基準驗證集與後續可選擴充方向
- **最新完修基準集**：`runs/v3_hydrogen_ep06_food_labeling/`
  - 正片：《瓶身寫義大利橄欖油，翻過來竟是調合油？特級初榨跟精煉果渣油價差數倍，這樣也算合規？食藥署品名標示到底在護誰？》（`750.2s` / 12.50 分鐘 / 57 幕 / 4,302 字）
  - Short：《瓶身寫義大利橄欖油翻過來竟是調合油？特級初榨跟果渣油差在哪？ #橄欖油 #食藥署 #獨角獸那麼叫》（`117.08s` / 8 幕 / 606 字）
  - 已驗證送達 Telegram（訊息 `#1477`、`#1478`、`#1479`）
- **頻道主現階段決策**：保留目前 V4 視覺版型與配音設定（已單獨完成字幕斷句修復與無效撞片去重）；未來若頻道主主動提出需要進階調整，可參考：
  - 自動產出其他支柱（`consumer_pricing`、`wage_labor_stats`、`industry_policy`）全新無人值守集數
  - 串接排程定時任務（`scripts/run_unattended_daemon.sh`）或 YouTube Data API 自動上傳草稿＋自動封面縮圖（Thumbnail）生成
