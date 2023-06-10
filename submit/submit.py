import jinja2
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int)
    args = parser.parse_args()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent),
        trim_blocks=True,
        autoescape=False,
        lstrip_blocks=True,
    )
    template = env.get_template("polaris.j2")
    script = template.render(
        jobname="polaris",
        queue="debug-scaling",
        nodes=args.nodes,
        config={
            "trainer.max_epochs": 5,
            "trainer.limit_val_batches": 200,
            "trainer.limit_train_batches": 200,
            "data.batch_size": 128,
        },
    )
    print(script)
