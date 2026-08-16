from .config import Settings
from .dialogue import DialogueDecision
from .nlp import ExtractedFeatures


class ConversationModelUnavailable(RuntimeError):
    pass


class ConversationGenerator:
    """Uses either Groq Llama 3.3 or the trained local LoRA dialogue model."""

    def __init__(self, settings: Settings):
        self.client = None
        self.local_model = None
        self.local_tokenizer = None
        self.torch = None
        self.model = settings.groq_model
        if settings.enable_local_dialogue_model and settings.local_dialogue_model_dir.exists():
            self._load_local(settings)
        if settings.groq_api_key:
            try:
                from groq import Groq

                self.client = Groq(api_key=settings.groq_api_key)
            except Exception as exc:
                print(f"[WARNING] Groq client unavailable: {exc}")

    def _load_local(self, settings: Settings) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.local_tokenizer = AutoTokenizer.from_pretrained(settings.local_dialogue_model_dir)
            base_model = AutoModelForCausalLM.from_pretrained(settings.local_dialogue_base_model)
            self.local_model = PeftModel.from_pretrained(base_model, settings.local_dialogue_model_dir)
            self.local_model.eval()
        except Exception as exc:
            print(f"[WARNING] Local dialogue model could not be loaded: {exc}")
            self.local_model = None
            self.local_tokenizer = None

    @property
    def source(self) -> str:
        if self.local_model is not None:
            return "local-lora-dialogue"
        return self.model if self.client else "unavailable"

    @property
    def ready(self) -> bool:
        return self.local_model is not None or self.client is not None

    def generate(
        self,
        user_text: str,
        features: ExtractedFeatures,
        decision: DialogueDecision,
        history: list[dict],
    ) -> str:
        if self.local_model is not None:
            return self._generate_local(user_text, history)
        if not self.client:
            raise ConversationModelUnavailable(
                "No dialogue model is available. Add GROQ_API_KEY or train the local LoRA model."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MindTriage, a non-diagnostic listening and triage assistant. "
                    "Use Motivational Interviewing OARS. Write one brief reflection and at most "
                    "one question. Never diagnose, prescribe treatment, promise confidentiality, "
                    "or invent crisis resources. Do not mention internal risk scores. "
                    f"Current emotions: {', '.join(features.emotions)}. "
                    f"Problem category: {features.category}. "
                    f"Conversation objective: {decision.target_slot}. Policy: {decision.policy}."
                ),
            }
        ]
        for turn in history[-4:]:
            messages.extend(
                [
                    {"role": "user", "content": turn["user"]},
                    {"role": "assistant", "content": turn["bot"]},
                ]
            )
        messages.append({"role": "user", "content": user_text})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=120,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            raise ConversationModelUnavailable(f"Groq request failed: {exc}") from exc
        if not reply:
            raise ConversationModelUnavailable("Groq returned an empty response")
        return reply

    def _generate_local(self, user_text: str, history: list[dict]) -> str:
        context = []
        for turn in history[-4:]:
            context.append(f"<|user|>\n{turn['user']}</s>\n<|assistant|>\n{turn['bot']}</s>")
        context.append(f"<|user|>\n{user_text}</s>\n<|assistant|>\n")
        inputs = self.local_tokenizer(
            "\n".join(context), return_tensors="pt", truncation=True, max_length=1024
        )
        device = next(self.local_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.local_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        reply = self.local_tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not reply:
            raise ConversationModelUnavailable("The local dialogue model returned an empty response")
        return reply
