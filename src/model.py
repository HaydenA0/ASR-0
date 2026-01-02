import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from globals import PROJECT_SOURCE


base_model = "openai/whisper-large-v3-turbo"
lora_model = "anaszil/whisper-large-v3-turbo-darija"

d_type = torch.float16 if torch.cuda.is_available() else torch.float32
device = 0 if torch.cuda.is_available() else "cpu"
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)


base = WhisperForConditionalGeneration.from_pretrained(base_model, dtype=d_type)
model = PeftModel.from_pretrained(base, lora_model)
processor = WhisperProcessor.from_pretrained(base_model, language="Arabic", task="transcribe")

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    device=device,
    generate_kwargs={
        "condition_on_prev_tokens": False,  
        "compression_ratio_threshold": 1.35,
        "temperature": 0.0, 
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6
    }
)

output = asr(PROJECT_SOURCE + "/audio/buffer.wav")
output_text = output["text"]
with open(PROJECT_SOURCE + "/transcription/buffer.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

    


