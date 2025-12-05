SemEval-2026 Task 13 — Detecting Machine-Generated Code
System Description by Marwa Alnajjar & Shamma Alnuaimi
This repository contains our full system for SemEval-2026 Task 13, which focuses on detecting and attributing machine-generated code across multiple programming languages, generators, and authorship scenarios.
We implemented scalable and lightweight pipelines based on TF-IDF character n-grams combined with SGD linear classifiers, achieving competitive performance while maintaining computational efficiency.
Task Overview
The SemEval-2026 Task 13 consists of three subtasks:
Subtask A — Binary Classification
Classifies code as:
0: Human-written
1: Machine-generated
Focuses heavily on out-of-distribution generalization across:
Unseen programming languages
Unseen code domains
Mismatched generator distributions
Subtask B — Multi-Class Authorship Attribution
Classifies code into 11 classes (1 human + 10 LLM families), including:
DeepSeek-AI, Qwen, O1-AI, Mistral, Meta-LLaMA, OpenAI, Granite, Phi, Gemma, BigCode.
Challenges:
Large dataset
Overlapping coding styles
Unseen model variants during testing
Subtask C — Hybrid & Adversarial Code Detection
Four-way classification:
0: Human-written
1: Machine-generated
2: Hybrid (human + AI)
3: Adversarial AI code
This subtask is the most difficult due to mixed stylistic signals.
Dataset & Preprocessing
All datasets provided in .parquet format
Preprocessing kept minimal to preserve stylistic cues
Character-level TF–IDF used for all subtasks
Combined training + validation for final training runs
Model Architecture
We use a lightweight and scalable pipeline:
TF–IDF character-level vectorizer
n-grams: (3,4) or (3,5)
max_features: up to 200k
SGDClassifier (linear model)
hinge loss
L2 regularization
Tuned learning rate α
Advantages:
Fast to train on CPU
Handles millions of sparse features
Strong performance for authorship detection
Experiments & Results
Subtask A (Binary)
Best configuration:
n-gram = (3,4)
max_features = 150,000
α = 5 × 10⁻⁵
Model showed high recall for machine-generated code but struggled with human-written code → slight bias.
Subtask B (Multi-Class)
Best configuration:
n-gram = (3,4)
max_features = 150,000
α = 5 × 10⁻⁵
Achieved:
Macro F1 ≈ 0.2746 on validation
Handles major LLM families well, struggles on minority classes
Subtask C (Hybrid & Adversarial)
Best configuration found:
n-gram = (3,5)
max_features = 150,000
α = 5 × 10⁻⁵
Kernel stalled due to:
Very large dataset
High-dimensional feature space
Heavy RAM usage
Future improvements:
Reduce feature size
Reduce dataset portion
Use GPU / larger RAM environment
