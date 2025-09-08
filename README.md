# 🧠 Chain-of-Thought Cheating Monitor

This repo explores how language models solve reasoning problems and whether they **"cheat"** by directly copying numbers from the input instead of reasoning step-by-step.

We fine-tune `distilgpt2` on a **synthetic Chain-of-Thought (CoT) dataset** and monitor **attention weights** to detect suspicious number-copying behavior.

---

## 📌 Features

- ✅ Synthetic **Chain-of-Thought dataset** (math word problems)
- ✅ **Fine-tuning loop** for `distilgpt2` with PyTorch
- ✅ Custom **Dataset class** for tokenization & batching
- ✅ **Cheating Monitor**: tracks attention on number tokens during generation
- ✅ (Optional) Visualization: plot attention heatmaps / scores per step

---

## 🚀 Quickstart

### 1. Clone Repo

```bash
 git clone https://github.com/yourusername/cot-cheating-monitor.git
 cd cot-cheating-monitor
```

### 2. Install Requirements

```bash
 pip install torch transformers matplotlib numpy
```

### 3. Train Model

```python
from your_module import create_cot_dataset, run_finetuning

train_data = create_cot_dataset()
model = run_finetuning(model, tokenizer, train_data, epochs=3, batch_size=2, lr=5e-5)
```

### 4. Generate with Monitoring

```python
from your_module import generate_and_monitor

prompt = "Question: John has 12 oranges. He eats 4. Then he buys 7 more. How many oranges does he have?\nThought:"

generated_text, attention_data = generate_and_monitor(model, tokenizer, prompt)
print(generated_text)
```

### 5. Visualize Attention (Optional)

```python
import matplotlib.pyplot as plt
import numpy as np

scores = [np.sum(d["weights"]) for d in attention_data]
plt.plot(scores)
plt.title("Attention on Number Tokens per Generation Step")
plt.xlabel("Step")
plt.ylabel("Attention Score")
plt.show()
```

## 📊 Example Output

- **Generated CoT reasoning**: `He starts with 12. After eating 4, he has 8. Then he buys 7. So he has 15.`
- **Cheating Monitor log**:
  ```
  [Step 1] Token: "12" | Attention to numbers: 0.81
  [Step 2] Token: "-"  | Attention to numbers: 0.02
  [Step 3] Token: "4"  | Attention to numbers: 0.73
  ```

---

##
