"""
dataset.py — DiagnoSys ML Classification
DatasetBuilder: loads raw problems from PostgreSQL, preprocesses text,
and produces multi-hot encoded labels for 20 engineering domains.
"""

import re
import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Domain taxonomy ──────────────────────────────────────────────────────────
DOMAIN_LABELS = [
    "databases", "backend", "frontend", "ml", "devops",
    "security", "mobile", "systems", "networking", "web",
    "cloud", "testing", "performance", "architecture",
    "data_engineering", "nlp", "computer_vision", "rl",
    "robotics", "other",
]
NUM_LABELS = len(DOMAIN_LABELS)
LABEL2IDX = {label: idx for idx, label in enumerate(DOMAIN_LABELS)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}

MAX_TOKEN_LEN = 512


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_html(raw: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"http\S+", " ", text)          # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_labels(label_list: list[str]) -> np.ndarray:
    """Convert a list of domain strings to a multi-hot numpy vector."""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for lbl in label_list:
        if lbl in LABEL2IDX:
            vec[LABEL2IDX[lbl]] = 1.0
        else:
            vec[LABEL2IDX["other"]] = 1.0
    return vec


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class ProblemDataset(Dataset):
    """
    Wraps a list of (text, multi-hot label) pairs for DataLoader consumption.
    Tokenises on-the-fly so the tokeniser can be swapped without re-building.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[np.ndarray],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_TOKEN_LEN,
    ):
        assert len(texts) == len(labels), "texts and labels must be same length"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── DatasetBuilder ────────────────────────────────────────────────────────────
class DatasetBuilder:
    """
    Pulls raw engineering problems from PostgreSQL, cleans them, and
    returns train / validation / test splits as ProblemDataset objects.

    Expected table schema
    ─────────────────────
    problems (
        id          SERIAL PRIMARY KEY,
        title       TEXT NOT NULL,
        body        TEXT NOT NULL,
        source      VARCHAR(32),          -- 'stackoverflow' | 'github' | 'reddit'
        domains     TEXT[],               -- PostgreSQL array of domain strings
        created_at  TIMESTAMPTZ
    )
    """

    FETCH_SQL = text(
        """
        SELECT title, body, domains
        FROM   problems
        WHERE  domains IS NOT NULL
          AND  array_length(domains, 1) > 0
        ORDER  BY id
        """
    )

    def __init__(
        self,
        db_url: str,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        val_size: float = 0.10,
        test_size: float = 0.10,
        max_length: int = MAX_TOKEN_LEN,
        seed: int = 42,
    ):
        self.db_url = db_url
        self.model_name = model_name
        self.val_size = val_size
        self.test_size = test_size
        self.max_length = max_length
        self.seed = seed

        logger.info("Loading tokeniser: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        # Llama 3.1 has no pad token by default — use EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fetch_raw(self) -> list[dict]:
        engine = create_engine(self.db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(self.FETCH_SQL).fetchall()
        logger.info("Fetched %d labelled problems from DB", len(rows))
        return rows

    def _preprocess(self, rows) -> tuple[list[str], list[np.ndarray]]:
        texts, labels = [], []
        for row in rows:
            title = clean_html(row.title or "")
            body  = clean_html(row.body  or "")
            # Combine title (repeated for emphasis) + body
            combined = f"{title} {title} {body}".strip()
            if not combined:
                continue
            texts.append(combined)
            labels.append(encode_labels(row.domains or []))
        return texts, labels

    # ── public API ────────────────────────────────────────────────────────────

    def build(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> dict[str, DataLoader]:
        """
        Returns a dict with keys 'train', 'val', 'test' — each a DataLoader.
        """
        rows = self._fetch_raw()
        texts, labels = self._preprocess(rows)

        # Stratified split is tricky for multi-label; use random split
        n = len(texts)
        indices = list(range(n))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_size + self.test_size,
            random_state=self.seed,
        )
        relative_test = self.test_size / (self.val_size + self.test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=self.seed,
        )

        def _make_loader(idxs: list[int], shuffle: bool) -> DataLoader:
            ds = ProblemDataset(
                texts=[texts[i] for i in idxs],
                labels=[labels[i] for i in idxs],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

        loaders = {
            "train": _make_loader(train_idx, shuffle=True),
            "val":   _make_loader(val_idx,   shuffle=False),
            "test":  _make_loader(test_idx,  shuffle=False),
        }
        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train_idx), len(val_idx), len(test_idx),
        )
        return loaders

    def build_from_jsonl(
        self,
        path: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> dict[str, DataLoader]:
        """
        Offline / unit-test alternative: load from a JSONL file where each
        line is {"title": "...", "body": "...", "domains": ["backend", ...]}.
        """
        import json

        rows = []
        with open(path) as fh:
            for line in fh:
                obj = json.loads(line)
                # Mimic SQLAlchemy Row with a simple namespace
                from types import SimpleNamespace
                rows.append(SimpleNamespace(**obj))

        texts, labels = self._preprocess(rows)
        n = len(texts)
        indices = list(range(n))
        train_idx, temp_idx = train_test_split(
            indices, test_size=self.val_size + self.test_size, random_state=self.seed
        )
        relative_test = self.test_size / (self.val_size + self.test_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=relative_test, random_state=self.seed
        )

        def _make_loader(idxs, shuffle):
            ds = ProblemDataset(
                texts=[texts[i] for i in idxs],
                labels=[labels[i] for i in idxs],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)

        return {
            "train": _make_loader(train_idx, True),
            "val":   _make_loader(val_idx,   False),
            "test":  _make_loader(test_idx,  False),
        }


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    db_url = os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost/diagnosys")
    builder = DatasetBuilder(db_url=db_url)
    loaders = builder.build(batch_size=4, num_workers=0)
    batch = next(iter(loaders["train"]))
    print("input_ids shape :", batch["input_ids"].shape)
    print("labels shape    :", batch["labels"].shape)
    print("Sample labels   :", batch["labels"][0])
