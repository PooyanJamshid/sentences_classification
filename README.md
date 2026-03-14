Sentence Classification with Transformers
This project explores and implements efficient transfer learning for sentence classification. By utilizing pre-trained Transformer architectures (BERT, ALBERT, RoBERTa, and ELECTRA), we achieve high-performance results while working within practical computational constraints.

🏗 Methodology
We evaluated several state-of-the-art models to determine the most effective approach for our specific dataset.

Efficient Fine-Tuning Strategy
Rather than training models from scratch, which is computationally expensive, we employed a partial fine-tuning strategy:

Transfer Learning: We leverage models pre-trained on massive text corpora to capture nuanced linguistic representations.

Partial Freezing: Base layers were frozen to preserve pre-learned features, while task-specific classification heads were made learnable. This approach optimizes for hardware efficiency while maintaining high accuracy.

Diagnostic Analysis: We monitored the loss function throughout training to prevent overfitting and utilized confusion matrices to identify model biases.

📊 Evaluation & Results
The final model achieved an overall accuracy of 92%. Below is the detailed classification report:

Class	Precision	Recall	F1-Score	Support
0	0.90	0.96	0.93	55
1	0.95	0.87	0.91	45
Average	0.92	0.92	0.92	100
Analytical Insight: The model demonstrates high precision (0.95) for Class 1, ensuring reliable positive predictions. The high recall for Class 0 (0.96) indicates a robust ability to identify samples within that category, minimizing false negatives.

🛠 Tech Stack
Core: Python, PyTorch, Transformers (Hugging Face)

Analysis: Scikit-learn (Metrics), Matplotlib (Visualization)

Optimization: Partial Layer Freezing

🚀 How to Use
Clone the repository:

Bash
git clone https://github.com/PooyanJamshid/sentences_classification
Install dependencies:

Bash
pip install pandas torch transformers scikit-learn matplotlib tqdm
