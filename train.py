import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm
from gpt import GPT  # 앞서 정의한 GPT 모델 클래스

# ------------------------------
# Hyperparameters and Config
# ------------------------------
VOCAB_SIZE = None  # 나중에 토크나이저 로딩 후 설정
D_MODEL = 768
N_LAYERS = 12
HEADS = 12
D_FF = 3072
MAX_LEN = 512
DROPOUT = 0.1
BATCH_SIZE = 8
BLOCK_SIZE = 128
LR = 5e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 데이터셋 준비 (Wikitext-2)
# ------------------------------
# 3.1. HuggingFace Dataset 로드
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
# split: "train", "validation", "test"

# 3.2. GPT-2 토크나이저 불러오기
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
VOCAB_SIZE = len(tokenizer)


# 3.3. 텍스트를 토큰화하여 input_ids 컬럼 생성
def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# 3.4. 토큰 시퀀스를 BLOCK_SIZE 길이로 묶기
def group_texts(examples):
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)
    total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    concatenated = concatenated[:total_length]

    result = [
        concatenated[i: i + BLOCK_SIZE]
        for i in range(0, total_length, BLOCK_SIZE)
    ]
    return {"input_ids": result}


# ← 여기서 "input_ids"와 "attention_mask"를 함께 제거하도록 수정
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    remove_columns=["input_ids", "attention_mask"]
)


# 3.5. PyTorch DataLoader 준비
def collate_fn(batch):
    input_ids = torch.stack(
        [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    )
    return input_ids


train_dataset = lm_datasets["train"]
val_dataset = lm_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# ------------------------------
# 모델, 손실함수, 옵티마이저
# ------------------------------
model = GPT(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    heads=HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# ------------------------------
# 학습 함수 정의
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)  # (batch_size, BLOCK_SIZE)

        logits = model(batch)  # (batch, BLOCK_SIZE, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, VOCAB_SIZE),
            shift_labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch = batch.to(device)
            logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1)
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ------------------------------
# 전체 학습 루프
# ------------------------------
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, DEVICE)
    val_loss = evaluate(model, val_dataloader, criterion, DEVICE)
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    torch.save(model, "model.pth")
