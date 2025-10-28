from pathlib import Path
from collections import defaultdict, deque
import re
import os

TRAIN_A = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/trainA")
TRAIN_B = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/trainB")
TEST_A = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/testA")
TEST_B = Path("/Users/enke/Documents/CycleGAN/datasets/t1t2/testB")

# Pattern to extract NIfTI base name (everything before _slice<number>.png)
SLICE_RE = re.compile(r"^(?P<base>.+)_slice\d+\.png$")


def group_pngs_by_case(folder: Path):
    groups = defaultdict(list)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.lower().endswith(".png"):
            continue
        m = SLICE_RE.match(name)
        if not m:
            # Non-matching PNGs are ignored
            continue
        base = m.group("base")
        groups[base].append(p)
    # Sort each group's slices for determinism
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x.name)
    return groups


def compute_deletions_for_balance(folder_a: Path, folder_b: Path):
    groups_a = group_pngs_by_case(folder_a)
    count_a = sum(len(v) for v in groups_a.values())

    # B is the reference count
    count_b = sum(1 for p in folder_b.iterdir() if p.is_file() and p.name.lower().endswith('.png'))

    if count_a <= count_b:
        return {
            "needed": 0,
            "count_a": count_a,
            "count_b": count_b,
            "to_delete": []
        }

    target = count_b
    need_delete = count_a - target

    # Ensure at least one slice kept per group if possible
    groups = [(k, v) for k, v in groups_a.items()]
    # Prepare deletion queues per group (all but the first slice kept initially)
    per_group_delete_candidates = []
    for _, slices in groups:
        if len(slices) <= 1:
            per_group_delete_candidates.append([])
        else:
            per_group_delete_candidates.append(slices[1:])

    # First, reserve one slice per group (keep first). Now delete round-robin from remaining until we hit need_delete
    # If target < number of groups, we cannot keep at least one per group. Handle separately.
    num_groups = len(groups)
    min_keep_needed = min(num_groups, target)
    if target < num_groups:
        # We can only keep target groups; choose the first target groups to keep one slice; delete all from the rest
        to_delete = []
        # Groups beyond target: delete all slices
        for idx in range(target, num_groups):
            to_delete.extend(groups[idx][1]) if False else None
            to_delete.extend(groups[idx][1]) if False else None
        # Above is placeholder; correct logic below
        to_delete = []
        for idx, (_, slices) in enumerate(groups):
            if idx < target:
                # keep first slice, delete the rest in these groups when needed later via round-robin
                to_delete.extend(slices[1:])
            else:
                # delete all slices of groups beyond target
                to_delete.extend(slices)
        # Trim to exactly need_delete
        to_delete = to_delete[:need_delete]
        return {
            "needed": need_delete,
            "count_a": count_a,
            "count_b": count_b,
            "to_delete": to_delete
        }

    # target >= num_groups: we can keep one per group, delete the rest evenly
    # Build a round-robin queue of candidates
    queue = deque()
    for cand_list in per_group_delete_candidates:
        if cand_list:
            queue.append(list(cand_list))
    to_delete = []
    while need_delete > 0 and queue:
        cand_list = queue.popleft()
        if not cand_list:
            continue
        f = cand_list.pop(0)
        to_delete.append(f)
        need_delete -= 1
        if cand_list:
            queue.append(cand_list)

    # If still need_delete > 0 (e.g., many groups with 1 slice), delete additional first slices from groups in round-robin
    if need_delete > 0:
        first_slices = [slices[0] for _, slices in groups if slices]
        i = 0
        while need_delete > 0 and i < len(first_slices):
            to_delete.append(first_slices[i])
            need_delete -= 1
            i += 1

    return {
        "needed": count_a - target if count_a > count_b else 0,
        "count_a": count_a,
        "count_b": count_b,
        "to_delete": to_delete
    }


def summarize(folder: Path):
    pngs = [p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith('.png')]
    return len(pngs)


def balance_pair(folder_a: Path, folder_b: Path, dry_run: bool = True):
    plan = compute_deletions_for_balance(folder_a, folder_b)
    print(f"Ordner A: {folder_a}")
    print(f"Ordner B: {folder_b}")
    print(f"PNG A: {plan['count_a']}, PNG B (Referenz): {plan['count_b']}")
    print(f"Zu löschen in A: {plan['needed']}")

    if plan["needed"] <= 0:
        print("Keine Aktion nötig (A <= B).")
        return

    preview = plan["to_delete"][:20]
    if preview:
        print("Vorschau (erste 20 geplante Löschungen):")
        for p in preview:
            print(f" - {p}")

    if not dry_run:
        for p in plan["to_delete"]:
            try:
                os.remove(p)
            except Exception as e:
                print(f"Fehler beim Löschen {p}: {e}")
        print("Löschung abgeschlossen.")
        print(f"Neue Anzahl A: {summarize(folder_a)}")


def main(dry_run: bool = True):
    print("Balance trainA vs trainB:")
    balance_pair(TRAIN_A, TRAIN_B, dry_run=dry_run)
    print("\nBalance testA vs testB:")
    balance_pair(TEST_A, TEST_B, dry_run=dry_run)


if __name__ == "__main__":
    # Execute deletions (confirmed by user)
    main(dry_run=False)
