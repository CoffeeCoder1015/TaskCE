from datasets import load_dataset
# -- Original datasets--
# medical_reasoning = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[:10000]")
# chat_to_sql = load_dataset("philschmid/gretel-synthetic-text-to-sql",split="train")

# Datasets (for testing)
medical_reasoning = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[10000:]")
chat_to_sql = load_dataset("philschmid/gretel-synthetic-text-to-sql",split="test")

model = ""
task = ""
check_point = ""
LoraCheckpointPath = f"models/{model}/{task}/{check_point}"