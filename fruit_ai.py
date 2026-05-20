"""
PersistentFruitAI v11 — Enhanced, Error‑free, GitHub‑Actions‑Ready
================================================================
- Fixed smoothing formula (proper Laplace smoothing).
- Full persistence: tape, stats, and contexts are saved/loaded.
- Clean imports, robust JSON handling.
- Runs automatically on GitHub Actions (detected via env var).
"""

from __future__ import annotations

import json
import os
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────
# إعداد الفواكه
# ─────────────────────────────
FRUITS: Tuple[str, ...] = (
    "watermelon", "tomato", "kiwi", "orange",
    "pomegranate", "dates", "lemon", "avocado",
)

N: int = len(FRUITS)
IDX: Dict[str, int] = {f: i for i, f in enumerate(FRUITS)}

# إعدادات النموذج
WINDOW: int = 40
MAX_ORDER: int = 6
SMOOTH: float = 0.5          # تجانس لابلاس
TOP_K: int = 6

# ملف حفظ الحالة
FILE: str = "fruit_brain_v11.json"


# ─────────────────────────────
# الإحصائيات
# ─────────────────────────────
@dataclass
class Stats:
    total_predictions: int = 0
    correct_predictions: int = 0
    total_updates: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


# ─────────────────────────────
# النموذج الأساسي
# ─────────────────────────────
class PersistentFruitAI:
    def __init__(self) -> None:
        self.ctx: Dict[Tuple[int, ...], array] = {}
        self.tot: Dict[Tuple[int, ...], int] = {}
        self.tape: deque[int] = deque(maxlen=WINDOW)
        self.stats: Stats = Stats()

    # ───────────── تعلُّم ─────────────
    def learn(self, idx: int) -> None:
        """تسجيل انتقال tape الحالي → idx."""
        t = tuple(self.tape)
        max_len = min(len(t), MAX_ORDER)
        for L in range(1, max_len + 1):
            key = t[-L:]
            if key not in self.ctx:
                self.ctx[key] = array("f", [0.0] * N)
                self.tot[key] = 0
            self.ctx[key][idx] += 1
            self.tot[key] += 1

    # ───────────── حساب الدرجات ─────────────
    def scores(self) -> Optional[List[float]]:
        """
        تحسب درجات غير مُطبَّعة لكل فاكهة بدمج السياقات المُختلفة.
        تُستخدَم تجانس لابلاس الحقيقي لكل سياق.
        """
        if not self.tape:
            return None

        t = tuple(self.tape)
        scores = [0.0] * N

        for L in range(1, min(len(t), MAX_ORDER) + 1):
            key = t[-L:]
            if key not in self.ctx:
                continue
            total = self.tot[key]
            if total == 0:
                continue

            weight = L * L
            # تجانس لابلاس: (count + SMOOTH) / (total + N * SMOOTH)
            denom = total + N * SMOOTH
            for i in range(N):
                scores[i] += (self.ctx[key][i] + SMOOTH) * weight / denom

        return scores

    # ───────────── أفضل التوقعات ─────────────
    def predict(self) -> List[Tuple[str, float]]:
        """أعلى TOP_K فواكه مع احتمالاتها المُطبَّعة."""
        s = self.scores()
        if s is None:
            uniform = 1.0 / N
            return [(f, uniform) for f in FRUITS[:TOP_K]]

        total = sum(s)
        if total == 0:
            uniform = 1.0 / N
            return [(f, uniform) for f in FRUITS[:TOP_K]]

        probs = [(FRUITS[i], s[i] / total) for i in range(N)]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:TOP_K]

    def top(self) -> int:
        """مُعرّف الفاكهة صاحبة أعلى درجة."""
        s = self.scores()
        if s is None:
            return 0
        return max(range(N), key=lambda i: s[i])  # type: ignore[arg-type]

    # ───────────── خطوة تحديث ─────────────
    def update(self, fruit: str) -> bool:
        """
        إدخال فاكهة جديدة.
        تُحدِّث الإحصائيات، تتعلَّم، وتُضيف للشريط.
        تُرجِع True لو كانت التوقُّع السابق صحيحًا.
        """
        if fruit not in IDX:
            return False

        idx = IDX[fruit]
        correct = False

        if self.tape:
            pred = self.top()
            correct = (pred == idx)
            self.stats.total_predictions += 1
            if correct:
                self.stats.correct_predictions += 1

        self.learn(idx)
        self.tape.append(idx)
        self.stats.total_updates += 1
        return correct

    # ───────────── حفظ / تحميل ─────────────
    def save(self) -> None:
        """حفظ الحالة كاملة (سياقات، شريط، إحصائيات) إلى JSON."""
        payload = {
            "ctx": [
                {"k": list(k), "v": list(v)}
                for k, v in self.ctx.items()
            ],
            "tape": list(self.tape),
            "stats": {
                "total_predictions": self.stats.total_predictions,
                "correct_predictions": self.stats.correct_predictions,
                "total_updates": self.stats.total_updates,
            },
        }
        tmp_path = Path(FILE).with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(FILE)          # كتابة ذرّية
        except OSError as e:
            print(f"[تحذير] فشل الحفظ: {e}")

    def load(self) -> bool:
        """
        تحميل الحالة من الملف.
        تُرجِع True إذا وُجِد ملف صالح.
        """
        path = Path(FILE)
        if not path.is_file():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[تحذير] ملف التكوين تالف، سيتم البدء من الصفر: {e}")
            return False

        # استعادة السياقات
        for item in data.get("ctx", []):
            key = tuple(item["k"])
            values = array("f", item["v"])
            self.ctx[key] = values
            self.tot[key] = int(sum(values))

        # استعادة الشريط
        tape_list = data.get("tape", [])
        self.tape = deque(tape_list, maxlen=WINDOW)

        # استعادة الإحصائيات
        s = data.get("stats", {})
        self.stats = Stats(
            total_predictions=s.get("total_predictions", 0),
            correct_predictions=s.get("correct_predictions", 0),
            total_updates=s.get("total_updates", 0),
        )

        return True


# ─────────────────────────────
# تشغيل GitHub Actions
# ─────────────────────────────
def github_run() -> None:
    """دالة تُستخدَم عند التشغيل داخل GitHub Actions."""
    print("🚀 Running FruitAI on GitHub Actions")

    ai = PersistentFruitAI()
    loaded = ai.load()
    print("📂 Loaded previous state:", loaded)

    # بيانات اختبار للعرض (فواكه صحيحة فقط)
    test_data = [
        "kiwi", "tomato", "lemon", "orange", "avocado", "watermelon",
        "pomegranate", "dates", "kiwi", "orange", "lemon", "tomato"
    ]

    for fruit in test_data:
        if fruit in IDX:
            correct = ai.update(fruit)
            preds = ai.predict()
            print(f"\n🍈 Input: {fruit}")
            print("   Correctly predicted:", correct)
            print("   Top predictions:", preds)
        else:
            print(f"   ⚠️  '{fruit}' skipped (unknown)")

    print("\n📊 FINAL STATS")
    print("   Accuracy :", f"{ai.stats.accuracy:.2%}")
    print("   Updates  :", ai.stats.total_updates)
    print("   Predictions:", ai.stats.total_predictions)

    ai.save()
    print("💾 Model saved.")


# ─────────────────────────────
# نقطة البداية
# ─────────────────────────────
if __name__ == "__main__":
    if os.getenv("GITHUB_ACTIONS") == "true":
        github_run()
    else:
        print("تشغيل محلي فقط (لا توجد واجهة تفاعلية في نسخة CI).")
        print("لاستخدام الوضع التفاعلي، شغّل الكود الأصلي.")