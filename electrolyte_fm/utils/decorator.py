import os


def leader_only(f):
    def wrapper():
        rank = os.environ.get("RANK")
        if rank is None or int(rank) != 0:
            print(f"Not rank 0 -> skipping {f}")
            return None
        return f()

    return wrapper
