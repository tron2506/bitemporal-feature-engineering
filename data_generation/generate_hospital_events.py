#!/usr/bin/env python3
"""
Generates a synthetic hospital_events.jsonl file (~100,000 events)
for an ICU admission prediction feature engineering task.

Key design choices:
- Each patient is admitted, then receives HEART_RATE and LAB_WBC events.
- Tpredict = admission_time + 2 hours for each patient.
- Edge cases are deliberately injected:
    1. Events where valid_time < Tpredict but transaction_time > Tpredict
       (late-arriving data — must be excluded to prevent leakage).
    2. Corrections (supersedes_event_id) that arrive after Tpredict
       for labs taken before Tpredict (must be excluded).
    3. Corrections that arrive *before* Tpredict (should be used).
    4. Events where both valid_time and transaction_time are after Tpredict
       (future data — must be excluded).
    5. Events where both are before Tpredict (normal, should be included).
"""

import json
import uuid
import random
from datetime import datetime, timedelta

random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_PATIENTS = 2000
# We target ~100k events.  Per patient we'll generate:
#   1 ADMISSION + ~25 HEART_RATE + ~20 LAB_WBC + corrections ≈ 50 events
# 2000 * 50 = 100,000
BASE_DATE = datetime(2024, 1, 1, 0, 0, 0)
OUTPUT_FILE = "hospital_events.jsonl"

HEART_RATE_MEAN = 80.0
HEART_RATE_STD = 15.0
WBC_MEAN = 9.0  # thousands per µL
WBC_STD = 4.0


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def make_event(
    patient_id: str,
    event_type: str,
    valid_time: datetime,
    transaction_time: datetime,
    value: float | None,
    supersedes_event_id: str | None = None,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "event_type": event_type,
        "valid_time": iso(valid_time),
        "transaction_time": iso(transaction_time),
        "value": value,
        "supersedes_event_id": supersedes_event_id,
    }


def generate_patient_events(patient_id: str, admission_time: datetime) -> list[dict]:
    """Generate all events for a single patient."""
    events = []
    t_predict = admission_time + timedelta(hours=2)

    # --- 1. ADMISSION event (always recorded promptly) ---
    tx_delay = timedelta(seconds=random.randint(0, 120))
    admission_evt = make_event(
        patient_id, "ADMISSION", admission_time, admission_time + tx_delay, None
    )
    events.append(admission_evt)

    # We'll track LAB_WBC event_ids so corrections can reference them
    lab_events_before_predict = []  # (event_dict, valid_time, transaction_time)

    # --- 2. HEART_RATE events ---
    # Generate over a window of ~6 hours from admission (spans before & after Tpredict)
    num_hr = random.randint(22, 32)
    for _ in range(num_hr):
        offset_minutes = random.uniform(-10, 360)  # slight pre-admission readings possible
        vt = admission_time + timedelta(minutes=offset_minutes)
        # Normal case: transaction_time is shortly after valid_time
        tx_delay_sec = random.uniform(0, 300)  # 0-5 min recording delay

        # 8% chance: late-arriving heart rate (valid before Tpredict, tx after)
        if random.random() < 0.08 and vt < t_predict:
            tx_time = t_predict + timedelta(minutes=random.uniform(5, 180))
        else:
            tx_time = vt + timedelta(seconds=tx_delay_sec)

        value = round(max(30, random.gauss(HEART_RATE_MEAN, HEART_RATE_STD)), 1)
        evt = make_event(patient_id, "HEART_RATE", vt, tx_time, value)
        events.append(evt)

    # --- 3. LAB_WBC events ---
    num_labs = random.randint(18, 27)
    for _ in range(num_labs):
        offset_minutes = random.uniform(0, 360)
        vt = admission_time + timedelta(minutes=offset_minutes)

        # Normal recording delay for labs is longer (processing time)
        tx_delay_min = random.uniform(5, 90)

        # 10% chance: very late transaction (lab arrives hours after drawn)
        if random.random() < 0.10:
            tx_delay_min = random.uniform(120, 360)

        tx_time = vt + timedelta(minutes=tx_delay_min)
        value = round(max(0.5, random.gauss(WBC_MEAN, WBC_STD)), 2)

        evt = make_event(patient_id, "LAB_WBC", vt, tx_time, value)
        events.append(evt)

        # Track labs that occurred before Tpredict for possible corrections
        if vt < t_predict:
            lab_events_before_predict.append((evt, vt, tx_time))

    # --- 4. LAB_WBC CORRECTIONS ---
    # Some fraction of pre-Tpredict labs get corrected
    if lab_events_before_predict:
        num_corrections = random.randint(
            1, max(1, len(lab_events_before_predict) // 4)
        )
        corrected_labs = random.sample(
            lab_events_before_predict,
            min(num_corrections, len(lab_events_before_predict)),
        )

        for original_evt, orig_vt, orig_tx in corrected_labs:
            original_value = original_evt["value"]
            # Correction adjusts the value somewhat
            corrected_value = round(
                original_value + random.uniform(-2.0, 2.0), 2
            )
            corrected_value = max(0.5, corrected_value)

            # The correction's valid_time is the SAME as original (same blood draw)
            correction_vt = orig_vt

            # --- Edge case split ---
            if random.random() < 0.5:
                # Case A: Correction arrives BEFORE Tpredict (should be visible)
                correction_tx = orig_tx + timedelta(
                    minutes=random.uniform(5, 60)
                )
                # Clamp to before Tpredict
                if correction_tx >= t_predict:
                    correction_tx = t_predict - timedelta(
                        seconds=random.randint(1, 600)
                    )
            else:
                # Case B: Correction arrives AFTER Tpredict (must be excluded)
                correction_tx = t_predict + timedelta(
                    minutes=random.uniform(5, 300)
                )

            correction_evt = make_event(
                patient_id,
                "LAB_WBC",
                correction_vt,
                correction_tx,
                corrected_value,
                supersedes_event_id=original_evt["event_id"],
            )
            events.append(correction_evt)

    return events


def main():
    all_events = []

    for i in range(NUM_PATIENTS):
        patient_id = f"P{i:05d}"
        # Spread admissions over ~30 days
        admission_offset = timedelta(
            hours=random.uniform(0, 30 * 24)
        )
        admission_time = BASE_DATE + admission_offset
        patient_events = generate_patient_events(patient_id, admission_time)
        all_events.extend(patient_events)

    # Shuffle to simulate real append-only log (events interleaved)
    random.shuffle(all_events)

    # Write to JSONL
    with open(OUTPUT_FILE, "w") as f:
        for evt in all_events:
            f.write(json.dumps(evt) + "\n")

    print(f"Generated {len(all_events):,} events for {NUM_PATIENTS} patients")
    print(f"Output: {OUTPUT_FILE}")

    # Print some stats
    types = {}
    corrections = 0
    for e in all_events:
        types[e["event_type"]] = types.get(e["event_type"], 0) + 1
        if e["supersedes_event_id"] is not None:
            corrections += 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c:,}")
    print(f"  Corrections (supersedes != null): {corrections:,}")


if __name__ == "__main__":
    main()
