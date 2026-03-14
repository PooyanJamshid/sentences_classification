# Sentence Classification with Transformers

This project implements and compares various Transformer-based models for sentence classification. By leveraging pre-trained architectures, we apply **transfer learning** to adapt models to our specific dataset efficiently.

## 🏗 Methodology
We evaluated several state-of-the-art models:
- **BERT (Base)**
- **ALBERT (Base)**
- **RoBERTa (Base)**
- **ELECTRA**

### Transfer Learning Strategy
To balance computational efficiency and performance, we employed **partial fine-tuning**:
- **Frozen Layers:** Base layers remain fixed to preserve learned linguistic features.
- **Trainable Layers:** Only the task-specific classification head and select upper layers are updated to adapt to our dataset.

## 📊 Evaluation & Results
The final model achieved an **overall accuracy of 92%**.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.90 | 0.96 | 0.93 | 55 |
| 1 | 0.95 | 0.87 | 0.91 | 45 |
| **Average** | **0.92** | **0.92** | **0.92** | **100** |


## 🛠 Tech Stack
- **Languages/Frameworks:** Python, PyTorch, Transformers (Hugging Face)
- **Data Analysis:** Pandas, Scikit-learn
- **Visualization:** Matplotlib

## 🚀 How to Run
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/PooyanJamshid/sentences_classification.git](https://github.com/PooyanJamshid/sentences_classification.git)
   cd your-repo
