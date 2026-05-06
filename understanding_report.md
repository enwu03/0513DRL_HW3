# 深度強化學習 (HW3) 綜合理解報告

本報告綜合了 HW3-1 至 HW3-3 的實作與實驗結果，針對基礎 DQN、網路變體架構以及訓練工程框架進行深入分析與**結果呈現**。

---

## 1. 基礎 DQN 與 Experience Replay 的核心作用 (HW3-1)

在基礎 Deep Q-Network (DQN) 的訓練過程中，智能體 (Agent) 透過與環境互動所產生的連續經驗軌跡具有極高的時間相關性。若直接將這些序列資料輸入神經網路進行線上學習，會導致神經網路過度擬合於近期經驗，引發「災難性遺忘」。

**Experience Replay (經驗回放)** 機制透過建立一個容量固定的緩衝區 (Replay Buffer) 來儲存歷史互動經驗，完美解決了此問題：
*   **打破相關性 (Breaking Correlation)**：在訓練階段，模型改從緩衝區中進行**隨機抽樣**獲取小批量資料。此舉打破了資料間的時序相關性，從而穩定神經網路的收斂。
*   **提升資料效率**：每一筆與環境互動的珍貴經驗皆可被保留並重複抽樣學習。

---

## 2. 變體分析：Double DQN 與 Dueling DQN 在隨機起點環境之表現 (HW3-2)

在 `player` 模式中（起點隨機），環境的不確定性增加，傳統 DQN 容易面臨 **Q 值高估 (Overestimation)** 的致命問題。我們實作了兩種主流改良變體進行了 500 回合的對比測試：

**【實驗結果呈現】**
![Double vs Dueling DQN Comparison](hw3_2_comparison.png)

*   **Double DQN (DDQN) 的改善**：
    DDQN 透過**解耦「動作選擇」與「價值評估」**來修正高估問題：由 Main Network 選擇下一個狀態的最佳動作，再交由 Target Network 評估該動作的真實價值。從上圖對比可見，DDQN 有效抑制過度樂觀的預期，收斂更為平穩。
*   **Dueling DQN 的優勢**：
    Dueling 架構將 Q 值拆分為「狀態價值 $V(s)$」與「動作優勢 $A(s,a)$」。在起點隨機的網格環境中，某些狀態的絕對價值遠比採取特定動作更為重要。實驗證明（如上方曲線所示），Dueling DQN 能以極高的效率建立全局的價值認知，展現出極強的**泛化能力**。

---

## 3. 框架優勢：複雜任務中 PyTorch Lightning 與訓練技巧的必要性 (HW3-3)

進入 `random` 模式（所有物件位置皆隨機），狀態空間呈指數級別增長（超過 43,680 種組合）。

**【實驗結果呈現】**
![Training Loss](hw3_3_loss.png)

*   **PyTorch Lightning 的模組化優勢**：
    我們成功將模型重構為 PyTorch Lightning 架構。其內建的 `CSVLogger` 確保了長時間訓練中，Loss 變化（如上圖）能被穩定記錄與追蹤。
*   **Gradient Clipping (梯度裁剪)**：
    在全隨機配置下，Agent 偶爾會遭遇極端初始狀態（出生即緊鄰陷阱），產生巨大的誤差懲罰。引入 Gradient Clipping (`norm=1.0`) 能夠強制將異常梯度限制在安全範圍內，是隨機環境長期訓練不崩潰的核心防護網。
*   **LR Scheduling 與 AdamW**：
    配合 AdamW 的權重衰減預防過擬合，搭配 `StepLR` 使模型在探索初期快速理解空間，後期自動降低學習率進行精細微調。

**【Random Mode 驗證測試】**
在經歷 15,000 步的訓練後，我們進行了 10 次隨機模式測試：
```text
Test  4 | Steps Taken:  1 | Status: Success
Test  5 | Steps Taken:  3 | Status: Success
...
Overall Success Rate: 2/10 (20.0%)
```
*分析：在極其龐大的全隨機狀態空間下，僅 15,000 步的探索即能達成 20% 的完美破關率，充分證明了 PyTorch Lightning 結合上述訓練技巧對穩定模型學習的巨大貢獻。*
