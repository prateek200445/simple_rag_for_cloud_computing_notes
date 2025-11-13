import os
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from openai import OpenAI

# 🔹 Explicitly set your OpenAI key
os.environ["OPENAI_API_KEY"] = "sk-proj-your_key_here"

# 🔹 Initialize OpenAI client
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset
dataset = load_dataset("json", data_files="ragas_eval_data.json")["train"]

# Choose metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Evaluate using explicit LLM client
results = evaluate(dataset=dataset, metrics=metrics, llm=llm)

print("\n📊 Final RAGAS Evaluation Results:\n")
for metric in results:
    print(f"{metric.name}: {results[metric]:.3f}")
