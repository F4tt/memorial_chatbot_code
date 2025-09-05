"""
Script đánh giá chất lượng model
"""
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model_path, config_path, base_model_name):
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        # Load model và tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.merge_and_unload()
    
    def calculate_perplexity(self, texts):
        """Tính perplexity trên tập texts"""
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
                )
                
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
                
                total_loss += loss * inputs.input_ids.shape[1]
                total_tokens += inputs.input_ids.shape[1]
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def generate_responses(self, test_inputs, max_length=256):
        """Generate responses cho test inputs"""
        responses = []
        
        for user_input in tqdm(test_inputs, desc="Generating responses"):
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
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
    
    def evaluate_on_dataset(self, val_file):
        """Đánh giá trên validation dataset"""
        # Load validation data
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = [json.loads(line) for line in f]
        
        # Prepare texts for perplexity
        texts = []
        test_inputs = []
        ground_truths = []
        
        for item in val_data:
            messages = item['messages']
            if len(messages) >= 2:
                user_msg = messages[0]['content']
                assistant_msg = messages[1]['content']
                
                # Full text cho perplexity
                full_text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
                texts.append(full_text)
                
                # Test inputs
                test_inputs.append(user_msg)
                ground_truths.append(assistant_msg)
        
        # Calculate perplexity
        print("📊 Đang tính perplexity...")
        perplexity = self.calculate_perplexity(texts)
        
        # Generate responses
        print("🤖 Đang generate responses...")
        generated_responses = self.generate_responses(test_inputs[:50])  # Test 50 samples đầu
        
        # Calculate metrics
        metrics = {
            'perplexity': perplexity,
            'num_samples': len(texts),
            'avg_input_length': np.mean([len(inp.split()) for inp in test_inputs]),
            'avg_response_length': np.mean([len(resp.split()) for resp in generated_responses]),
            'avg_ground_truth_length': np.mean([len(gt.split()) for gt in ground_truths[:50]])
        }
        
        return metrics, list(zip(test_inputs[:50], generated_responses, ground_truths[:50]))
    
    def create_evaluation_report(self, metrics, samples, output_file):
        """Tạo báo cáo đánh giá"""
        report = {
            'metrics': metrics,
            'samples': [
                {
                    'input': inp,
                    'generated': gen,
                    'ground_truth': gt
                }
                for inp, gen, gt in samples
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("📈 KẾT QUẢ ĐÁNH GIÁ")
        print("="*60)
        print(f"🔢 Perplexity: {metrics['perplexity']:.2f}")
        print(f"📊 Số samples: {metrics['num_samples']}")
        print(f"📝 Độ dài input trung bình: {metrics['avg_input_length']:.1f} từ")
        print(f"🤖 Độ dài response trung bình: {metrics['avg_response_length']:.1f} từ")
        print(f"📋 Độ dài ground truth trung bình: {metrics['avg_ground_truth_length']:.1f} từ")
        print(f"💾 Chi tiết đã lưu vào: {output_file}")
        
        # Show some samples
        print("\n📝 MỘT SỐ MẪU DEMO:")
        for i, (inp, gen, gt) in enumerate(samples[:3], 1):
            print(f"\n--- Sample {i} ---")
            print(f"👤 Input: {inp}")
            print(f"🤖 Generated: {gen}")
            print(f"📋 Ground Truth: {gt}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path đến model đã finetune")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Base model name")
    parser.add_argument("--val_file", type=str, default="./data/processed/val.jsonl", help="Validation file")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator(args.model_path, None, args.base_model)
    
    # Đánh giá
    metrics, samples = evaluator.evaluate_on_dataset(args.val_file)
    
    # Tạo báo cáo
    evaluator.create_evaluation_report(metrics, samples, args.output)

if __name__ == "__main__":
    main()