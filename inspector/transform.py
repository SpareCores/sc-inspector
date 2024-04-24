import os


def raw(meta, task, task_dir, stdout, stderr) -> list[str]:
    outputs: list[str] = []
    for name in ("stdout", "stderr"):
        if len(locals()[name]):
            with open(os.path.join(task_dir, name), "wb") as f:
                f.write(locals()[name])
            outputs.append(name)

    return outputs
