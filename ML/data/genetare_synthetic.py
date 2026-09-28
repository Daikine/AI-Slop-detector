"""Expand seed CSV into a bigger synthetic dataset (template augmentation).

Usage:
    python ML/data/generate_synthetic.py
Reads:  ML/data/processed/synthetic_seed.csv  (text,label,source,ai_assisted)
Writes: ML/data/processed/synthetic_full.csv
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "processed"
SEED_CSV = DATA_DIR / "synthetic_seed.csv"
OUT_CSV = DATA_DIR / "synthetic_full.csv"
AUG_PER_ROW = 30          # augmentations per seed row
TEMPLATE_ROWS = 400       # fully synthetic template rows
SEED = 42

random.seed(SEED)

# --- AI-style building blocks -------------------------------------------------
AI_GREET = [
    "Уважаемый клиент,", "Уважаемый пользователь,", "Здравствуйте!",
    "Дорогой друг,", "Уважаемый абонент,", "Здравствуйте, уважаемый клиент!",
]
AI_SIGNOFF = [
    "С уважением, служба поддержки.",
    "С уважением, команда {company}.",
    "Благодарим за понимание.",
    "Будем рады ответить на ваши вопросы.",
]
AI_CONNECT = [
    "Настоящим сообщаем, что ",
    "Обращаем ваше внимание, что ",
    "Информируем вас о том, что ",
    "Доводим до вашего сведения, что ",
]
AI_HAM_BODY = [
    "ваш запрос зарегистрирован под номером {n}.",
    "ваша подписка истекает через {d} дней. Пожалуйста, продлите её для сохранения доступа.",
    "мы обновили политику конфиденциальности. Ознакомьтесь с условиями по ссылке.",
    "ваш заказ успешно оформлен и передан в доставку. Ожидайте уведомление.",
    "запись на приём подтверждена. Дата визита: {d} числа текущего месяца.",
    "ваш профиль выбран для участия в программе лояльности с бонусами до 50%.",
    "доступ к личному кабинету восстановлен. Рекомендуем сменить пароль.",
    "направляем вам отчёт за прошедший период во вложении к данному письму.",
]
AI_SPAM_BODY = [
    "ваш аккаунт будет заблокирован в течение {d} часов. Подтвердите данные по ссылке ниже.",
    "мы заметили подозрительную активность. Авторизуйтесь для подтверждения личности.",
    "ваш платёж был отклонён. Обновите платёжные данные, чтобы продолжить использование.",
    "ваш билет подтверждён, однако требуется регистрация в течение {d} часов.",
    "для получения выигрыша необходимо подтвердить личность, перейдя по указанной ссылке.",
]

# --- Human-style building blocks ---------------------------------------------
HUMAN_GREET = ["Привет!", "Добрый день!", "Здарова,", "Эй,", "Приветик,"]
HUMAN_SPAM_BODY = [
    "ВЫ ВЫИГРАЛИ В ЛОТЕРЕЮ! Отправьте нам ваши банковские данные для получения приза.",
    "Купите Виагру со скидкой 90% без рецепта. Доставка за 24 часа.",
    "Хочешь заработать 5000$ в день работая из дома? Пиши в личку!",
    "Срочно!!! Ваша посылка задержана на таможне. Оплати сбор по ссылке!!!",
    "Ты не поверишь, я нашёл способ заработать миллион за неделю!",
    "Кредит без справок и поручителей за 5 минут, заходит на сайт",
]
HUMAN_HAM_BODY = [
    "Давай встретимся завтра в 15:00 в кафе на Ленина.",
    "Не забудь взять ноутбук с презентацией на созвон в понедельник.",
    "Отправляю акт по договору №{n}, проверь и подпиши пожалуйста.",
    "Скинь отчёт за прошлый месяц, не могу найти его в почте.",
    "Мама, позвони как сможешь.",
    "Заказ №{n} отправлен, ждите доставку 2-3 дня.",
]

COMPANIES = ["Банк", "Магазин", "Сервис", "Онлайн", "Маркет"]


def _fill(t: str) -> str:
    return (t.replace("{n}", str(random.randint(100, 99999)))
             .replace("{d}", str(random.choice([12, 24, 48, 72, 3, 7, 14])))
             .replace("{company}", random.choice(COMPANIES)))


def _swap_typo(text: str) -> str:
    """Swap two adjacent chars inside a random long word (human noise)."""
    words = text.split()
    idxs = [i for i, w in enumerate(words) if len(w) > 4]
    if not idxs:
        return text
    i = random.choice(idxs)
    w = words[i]
    j = random.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def augment_human(text: str) -> str:
    """Mutations that preserve 'human-written' style."""
    t = text
    op = random.random()
    if op < 0.15:
        words = t.split()
        longs = [i for i, w in enumerate(words) if len(w) > 3]
        if longs:
            i = random.choice(longs)
            words[i] = words[i].upper()
            t = " ".join(words)
    elif op < 0.30:
        t = t.rstrip(".!") + random.choice(["!!!", "!!", "!!!!!"])
    elif op < 0.42:
        t = _swap_typo(t)
    elif op < 0.52:
        t = t + random.choice([" :)", " ;)", " )))"])
    elif op < 0.62:
        t = t[0].lower() + t[1:] if t and t[0].isupper() else t
    elif op < 0.72:
        t = random.choice(HUMAN_GREET) + " " + t
    return _fill(t)


def augment_ai(text: str) -> str:
    """Mutations that preserve 'AI-generated' style."""
    t = text
    op = random.random()
    if op < 0.25:
        t = random.choice(AI_CONNECT) + t[0].lower() + t[1:]
    elif op < 0.45:
        t = t.rstrip(".") + ". " + random.choice(AI_SIGNOFF)
    elif op < 0.60:
        t = t.replace("!", ".")            # AI avoids exclamation marks
    elif op < 0.72:
        t = random.choice(AI_GREET) + " " + t[0].lower() + t[1:]
    elif op < 0.82:
        t = t + " Пожалуйста, не отвечайте на данное письмо."
    return _fill(t)


def make_templates() -> list[dict]:
    rows = []
    for _ in range(TEMPLATE_ROWS // 4):
        ai = random.random() < 0.5
        spam = random.random() < 0.5
        greet = random.choice(AI_GREET if ai else HUMAN_GREET)
        body = random.choice(AI_SPAM_BODY if spam else AI_HAM_BODY) if ai \
            else random.choice(HUMAN_SPAM_BODY if spam else HUMAN_HAM_BODY)
        text = _fill(f"{greet} {body}")
        rows.append({
            "text": text,
            "label": 1 if spam else 0,
            "source": "ai" if ai else "human",
            "ai_assisted": 1 if ai else 0,
        })
    return rows


def main() -> None:
    if not SEED_CSV.exists():
        raise SystemExit(
            f"[!] Seed file not found: {SEED_CSV}\n"
            "    Save your 50-row dataset there first (text,label,source,ai_assisted)."
        )
    with open(SEED_CSV, encoding="utf-8") as f:
        seeds = list(csv.DictReader(f))
    print(f"[i] loaded {len(seeds)} seed rows")

    rows: list[dict] = []
    seen: set[str] = set()

    def push(r: dict) -> None:
        t = r["text"].strip()
        if t and t not in seen and len(t) > 15:
            seen.add(t)
            rows.append(r)

    for r in seeds:
        push(dict(r))
        is_ai = r["ai_assisted"].strip() == "1"
        for _ in range(AUG_PER_ROW):
            t = augment_ai(r["text"]) if is_ai else augment_human(r["text"])
            push({
                "text": t,
                "label": r["label"].strip(),
                "source": r["source"].strip(),
                "ai_assisted": r["ai_assisted"].strip(),
            })

    for r in make_templates():
        push(r)

    random.shuffle(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source", "ai_assisted"],
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()