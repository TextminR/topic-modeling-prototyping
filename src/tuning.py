import subprocess
import os


# Environment dictionary with new variables
env = os.environ.copy()

for topics in [15, 50, 100, 250, 500]:
    env["NUM_TOPICS"] = f"{topics}"
    env["LDA_FILENAME"] = f"data/{topics}topics.lda.model"
    env["LSI_FILENAME"] = f"data/{topics}topics.lsi.model"

    # Execute the other program
    subprocess.run(["python", "src/modeling.py"], env=env)
    subprocess.run(["python", "src/coherence.py"], env=env)
