# 深度強化學習 (HW3) 綜合理解報告

本專案包含了 Deep Q-Network (DQN) 及其變體的實作，應用於 Grid World 環境。

## 1. 基礎 DQN 與 Experience Replay 的核心作用
在基礎 DQN 的訓練中，我們使用神經網路來取代傳統的 Q-Table。然而，如果 Agent 直接使用連續收集到的經驗進行訓練，會因為資料的高度時間相關性 (Temporal Correlation) 導致網路陷入過擬合與災難性遺忘。

**Experience Replay** 透過建立一個固定容量的 Buffer 儲存歷史經驗，並在訓練時隨機抽樣 (Random Sampling) 一批資料來更新網路。這打破了資料的時序相關性，穩定了神經網路的收斂，並大幅提升了資料的使用效率。

---

## 2. 變體分析：Double DQN 與 Dueling DQN

在 `player` 模式中（起點隨機），傳統 DQN 容易面臨 Q 值高估 (Overestimation) 的問題。

![Double vs Dueling DQN Comparison](hw3_2_comparison.png)

*   **Double DQN (DDQN)**：透過解耦「動作選擇」與「價值評估」的網路，使用 Main Network 選擇最佳動作，再由 Target Network 計算 Q 值。這有效抑制了過度樂觀的預期，使收斂曲線更為平穩。
*   **Dueling DQN**：將網路層級的 Q 值拆分為「狀態價值 $V(s)$」與「動作優勢 $A(s,a)$」。在起點隨機的網格中，這種架構能極快地建立對全圖的價值認知，展現出極強的泛化能力。從上圖可見，兩者皆能順利收斂至理論高分。

---

## 3. 框架優勢：PyTorch Lightning 與訓練技巧

在 `random` 模式（所有物件位置皆隨機）下，狀態空間極其龐大（超過四萬種配置）。

![Training Loss](hw3_3_loss.png)

*   **PyTorch Lightning**：模組化的框架使我們能輕鬆管理複雜的訓練迴圈與記錄，內建的 `CSVLogger` 幫助我們輕鬆畫出平滑的訓練 Loss 曲線（如上圖）。
*   **Gradient Clipping (梯度裁剪)**：在隨機配置下，Agent 偶爾會遭遇極端狀態（如出生即陷阱），這會產生巨大的 Loss。引入梯度裁剪 (`norm=1.0`) 強制將異常梯度限制在安全範圍內，是確保長期訓練不崩潰的防護網。
*   **AdamW 與 Learning Rate Scheduler**：配合權重衰減預防過擬合，並使用 StepLR 在後期降低學習率進行精細微調，成功讓模型在複雜隨機環境中找到最佳策略。
