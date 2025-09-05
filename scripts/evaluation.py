"""
Script for evaluating a fine-tuned chatbot model with Perplexity, BLEU, ROUGE, and BERTScore
"""
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
import evaluate

# Download NLTK data if not already installed
nltk.download("punkt", quiet=True)


class ModelEvaluator:
    def __init__(self, model_path, base_model_name):
        self.model_path = model_path
        self.base_model_name = base_model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load fine-tuned model (LoRA adapter merged into base)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.merge_and_unload()

    def calculate_perplexity(self, texts):
        """Compute perplexity on given texts"""
        total_loss = 0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating perplexity"):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)

                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()

                total_loss += loss * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity

    def generate_responses(self, test_inputs, max_length=256):
        """Generate model responses for user inputs"""
        responses = []

        for user_input in tqdm(test_inputs, desc="Generating responses"):
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            prompt_length = inputs.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=prompt_length + max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
            response = response.replace("<|im_end|>", "").strip()
            responses.append(response)

        return responses

    def compute_text_metrics(self, generated, ground_truths):
        """Compute BLEU, ROUGE, BERTScore"""
        # BLEU
        bleu_scores = []
        for hyp, ref in zip(generated, ground_truths):
            hyp_tokens = nltk.word_tokenize(hyp.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            bleu_scores.append(nltk.translate.bleu_score.sentence_bleu(ref_tokens, hyp_tokens))
        avg_bleu = np.mean(bleu_scores)

        # ROUGE
        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1, rouge2, rougel = [], [], []
        for hyp, ref in zip(generated, ground_truths):
            scores = rouge.score(ref, hyp)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougel.append(scores["rougeL"].fmeasure)
        avg_rouge1 = np.mean(rouge1)
        avg_rouge2 = np.mean(rouge2)
        avg_rougel = np.mean(rougel)

        # BERTScore (Hugging Face evaluate)
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(predictions=generated, references=ground_truths, lang="en")
        avg_bertscore_f1 = np.mean(bert_results["f1"])

        return {
            "bleu": avg_bleu,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougel,
            "bertscore_f1": avg_bertscore_f1,
        }

    def evaluate_on_dataset(self, val_file):
        """Evaluate model on a validation dataset"""
        with open(val_file, "r", encoding="utf-8") as f:
            val_data = [json.loads(line) for line in f]

        texts = []
        test_inputs = []
        ground_truths = []

        for item in val_data:
            messages = item["messages"]
            if len(messages) >= 2:
                user_msg = messages[0]["content"]
                assistant_msg = messages[1]["content"]

                full_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
                texts.append(full_text)

                test_inputs.append(user_msg)
                ground_truths.append(assistant_msg)

        print("Computing perplexity...")
        perplexity = self.calculate_perplexity(texts)

        print("Generating responses...")
        generated_responses = self.generate_responses(test_inputs[:50])
        ground_truths_subset = ground_truths[:50]

        print("Computing BLEU, ROUGE, BERTScore...")
        text_metrics = self.compute_text_metrics(generated_responses, ground_truths_subset)

        metrics = {
            "perplexity": perplexity,
            "num_samples": len(texts),
            "avg_input_length": np.mean([len(inp.split()) for inp in test_inputs]),
            "avg_response_length": np.mean([len(resp.split()) for resp in generated_responses]),
            "avg_ground_truth_length": np.mean([len(gt.split()) for gt in ground_truths_subset]),
            **text_metrics,
        }

        return metrics, list(zip(test_inputs[:50], generated_responses, ground_truths_subset))

    def create_evaluation_report(self, metrics, samples, output_file):
        """Save evaluation results to a JSON file"""
        report = {
            "metrics": metrics,
            "samples": [
                {"input": inp, "generated": gen, "ground_truth": gt}
                for inp, gen, gt in samples
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print(f"\nReport saved to: {output_file}")

        print("\nSAMPLE DEMONSTRATIONS:")
        for i, (inp, gen, gt) in enumerate(samples[:3], 1):
            print(f"\n--- Sample {i} ---")
            print(f"User Input: {inp}")
            print(f"Generated Response: {gen}")
            print(f"Ground Truth: {gt}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned chatbot model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model name")
    parser.add_argument("--val_file", type=str, default="./data/processed/val.jsonl", help="Validation file path")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report file path")

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path, args.base_model)
    metrics, samples = evaluator.evaluate_on_dataset(args.val_file)
    evaluator.create_evaluation_report(metrics, samples, args.output)


if __name__ == "__main__":
    main()
