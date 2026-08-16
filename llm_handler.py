"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Ashraf Mohammad

Reference local Hugging Face language-model wrapper.
Ashraf should review, modify, test, and commit his own version.

If a model cannot be loaded in free Colab, the RAG pipeline can continue
in retrieval-only mode instead of crashing.
"""

from typing import Optional
import torch

from config import (
    BACKUP_LLM_NAME,
    MAX_NEW_TOKENS,
    PRIMARY_LLM_NAME,
    TEMPERATURE,
)


SYSTEM_PROMPT = """You are a course-support assistant.
Answer ONLY from the supplied course-document context.
Do not invent requirements, grades, deadlines, or instructor policies.
If the context does not support an answer, say that you could not find
enough information in the uploaded course documents.
Keep the answer concise and useful."""


class LocalLLM:
    def __init__(
        self,
        primary_model: str = PRIMARY_LLM_NAME,
        backup_model: str = BACKUP_LLM_NAME,
    ):
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.model = None
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self.load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _try_load(self, model_name: str) -> bool:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.model_name = model_name
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def load(self) -> bool:
        if self.available:
            return True

        if self._try_load(self.primary_model):
            return True

        return self._try_load(self.backup_model)

    def generate(self, question: str, context: str) -> str:
        if not self.available and not self.load():
            raise RuntimeError(
                "No language model could be loaded. "
                f"Last error: {self.load_error}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"COURSE DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"COURSE DOCUMENT CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}\nANSWER:"
            )

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if TEMPERATURE and TEMPERATURE > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer
