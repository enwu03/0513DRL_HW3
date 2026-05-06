# 深度強化學習 HW3 — 綜合理解報告 (Understanding Report)

本報告針對 HW3-1 至 HW3-3 的實驗結果進行系統性分析，涵蓋核心演算法原理、架構變體比較以及工程層面的訓練穩定性策略。

---

## 一、DQN 核心邏輯與 Experience Replay 的作用 (HW3-1)

### 1.1 問題背景

Deep Q-Network 使用神經網路逼近動作價值函數 $Q(s, a)$，取代傳統的 Q-Table。然而，強化學習的資料收集過程具有一個根本性的問題：**連續的訓練樣本之間存在極高的時間相關性**。

例如，當 Agent 在 Grid 中向右走了三步，這三步產生的狀態幾乎完全相同。若直接使用這些高度相關的資料進行 SGD 更新，神經網路會過度擬合於近期軌跡，並快速遺忘先前學到的知識 —— 即「災難性遺忘 (Catastrophic Forgetting)」。

### 1.2 Experience Replay 的解決機制

Experience Replay 引入一個固定容量的緩衝區 $\mathcal{D}$，將每一步的互動經驗 $(s_t, a_t, r_t, s_{t+1}, \text{done})$ 存入其中。訓練時，模型不再使用最新的單筆資料，而是從 $\mathcal{D}$ 中**均勻隨機抽樣**一個 mini-batch。

這帶來兩個關鍵好處：

| 效果 | 說明 |
|:---|:---|
| **打破時間相關性** | 隨機抽樣使 mini-batch 中的樣本來自不同時間點與不同軌跡，近似滿足 i.i.d. 假設，使 SGD 能夠穩定收斂。 |
| **提升資料效率** | 每筆經驗可被重複抽樣學習多次，而非用完即丟。在樣本收集成本高的 RL 場景中，這大幅降低了所需的環境互動次數。 |

### 1.3 實驗驗證

**【實驗結果】**

| Naive DQN Loss | Experience Replay DQN Loss |
|:---:|:---:|
| ![Naive](hw3_1_naive_dqn_loss.png) | ![ER](hw3_1_er_dqn_loss.png) |

**觀察：**
- **Naive DQN**：Loss 曲線呈現持續的高頻震盪，幾乎無法觀察到明顯的下降趨勢。這正是時間相關性導致梯度方向反覆劇變的結果。
- **ER DQN**：Loss 曲線平滑下降並趨近於零，展示了穩定且有效的收斂過程。

---

## 二、變體分析：Double DQN 與 Dueling DQN (HW3-2)

### 2.1 傳統 DQN 的根本缺陷 — Q 值高估

基礎 DQN 使用以下公式計算目標 Q 值：

$$Q_{\text{target}} = r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a')$$

其中 $\max$ 運算子在估計值含有雜訊時，會系統性地選中被高估的動作。這不是偶發的誤差，而是一個**有偏估計 (Biased Estimator)**，會隨著訓練持續累積，最終導致策略偏離最優解。

### 2.2 Double DQN — 解耦選擇與評估

Double DQN 將「選擇最佳動作」和「評估該動作的價值」分配給兩個不同的網路：

$$a^* = \arg\max_{a'} Q_{\theta}(s', a') \quad \text{(Main Network 選擇)}$$
$$Q_{\text{target}} = r + \gamma \cdot Q_{\theta^-}(s', a^*) \quad \text{(Target Network 評估)}$$

由於 Main Network 和 Target Network 的參數不同，它們的估計誤差不會朝同一方向偏移。即使 Main Network 錯誤地選中了一個被高估的動作，Target Network 給出的評估值通常不會同樣被高估，從而有效抑制了偏差的累積。

### 2.3 Dueling DQN — 結構性分解 Q 值

Dueling DQN 從網路架構層面對 Q 值進行分解：

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

| 分支 | 學習目標 | 關鍵洞察 |
|:---|:---|:---|
| **$V(s)$ — 狀態價值** | 評估「身處此狀態」本身的好壞 | 無論採取何種動作，只要經過此狀態就會更新 |
| **$A(s, a)$ — 動作優勢** | 評估「在此狀態下，某動作比平均好多少」 | 隔離了動作之間的相對差異 |

在 `player` 模式中，Agent 會被放置在各種不同的起始位置。其中許多位置（例如遠離陷阱的空地）對所有動作來說都是相對安全的。Dueling 架構能夠透過 $V(s)$ 分支快速學會「這些狀態本身就是安全的」，而無需逐一評估每個動作的 Q 值。這使得 Dueling DQN 在面對**大量未見過的起始狀態**時，展現出更強的泛化效率。

### 2.4 實驗驗證

**【500 回合對比結果】**

| Double DQN vs Dueling DQN Reward Comparison |
|:---:|
| ![Comparison](hw3_2_comparison.png) |

**觀察：**
- 兩種變體在 ~200 回合後均穩定收斂至接近最優獎勵（~7.0 分），表明它們都成功學會了在隨機起點環境中導航至目標。
- 相較於基礎 DQN 在隨機起點下容易出現的策略震盪，Double 與 Dueling 變體展現了顯著的穩定性優勢。

---

## 三、框架優勢與訓練穩定性策略 (HW3-3)

### 3.1 全隨機環境的挑戰

在 `random` 模式中，Player、Goal、Pit、Wall 四個物件的位置全部隨機。這意味著：
- **狀態空間**：$16 \times 15 \times 14 \times 13 = 43{,}680$ 種可能的初始配置
- **稀疏獎勵**：Agent 必須在每一種全新的地圖中找到通往目標的路徑
- **極端轉移**：Agent 可能出生在陷阱旁邊，產生劇烈的 Loss 突刺

### 3.2 PyTorch Lightning 的工程價值

我們將模型重構為 `pl.LightningModule`（`LitDQN` 類別），獲得以下工程優勢：

| 特性 | 手動 PyTorch | PyTorch Lightning |
|:---|:---|:---|
| 訓練迴圈 | 手動撰寫 epoch/batch 迴圈 | `Trainer.fit()` 一行搞定 |
| 梯度裁剪 | 手動呼叫 `clip_grad_norm_` | `Trainer(gradient_clip_val=1.0)` |
| 日誌記錄 | 手動管理 CSV/TensorBoard | `self.log()` 自動寫入 |
| 裝置管理 | 手動 `.to(device)` | 自動偵測 CPU/GPU |

### 3.3 穩定性技巧整合

| 技巧 | 程式碼實作 | 解決的問題 |
|:---|:---|:---|
| **Gradient Clipping** | `gradient_clip_val=1.0, algorithm="norm"` | 當 Agent 遭遇極端獎勵轉移（如出生即死亡），Loss 突刺會導致梯度爆炸。裁剪將梯度範數限制在安全範圍內。 |
| **LR Scheduling** | `StepLR(step_size=2000, gamma=0.9)` | 訓練初期使用較大學習率加速探索；後期逐步衰減以進行精細的策略微調，避免在最優解附近震盪。 |
| **AdamW** | `weight_decay=1e-4` | 相較於 Adam，AdamW 正確地將權重衰減與梯度更新解耦，提供更有效的 L2 正則化，防止模型過擬合於頻繁出現的地圖配置。 |
| **大容量 Replay Buffer** | `capacity=10,000` | 在四萬多種配置中，許多罕見但關鍵的地圖佈局可能只出現一兩次。大容量 Buffer 確保這些珍貴經驗不會被過早覆蓋。 |

### 3.4 實驗驗證

**【Training Loss 曲線（平滑處理）】**

| PyTorch Lightning Training Loss |
|:---:|
| ![Loss](hw3_3_loss.png) |

**觀察：**
- Loss 曲線呈現明確的下降趨勢，且變異程度受到控制，沒有出現災難性的爆炸，驗證了 Gradient Clipping 的保護效果。
- 訓練 15,000 步後進行 10 次隨機測試，成功率達 **70%**（7/10）。在狀態空間高達 43,680 種配置的全隨機環境中，這一結果充分驗證了 Gradient Clipping、LR Scheduling 與 AdamW 等穩定性技巧的顯著效果。

---

## 四、總結

| 作業 | 環境 | 核心技術 | 關鍵收穫 |
|:---|:---|:---|:---|
| HW3-1 | Static | Experience Replay | 打破時間相關性是 DQN 訓練穩定的基石 |
| HW3-2 | Player | Double DQN, Dueling DQN | 解耦估計與結構分解有效應對未知起點 |
| HW3-3 | Random | PyTorch Lightning + 訓練技巧 | 工程防護網是複雜環境下長期訓練的關鍵 |
