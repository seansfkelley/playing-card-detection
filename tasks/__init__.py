from invoke import task
from glob import glob
import matplotlib.image as mpimage
import pickle

DATA_DIR = "data"
BACKGROUNDS_FILE = f"{DATA_DIR}/backgrounds.pickle"

@task
def fetch_backgrounds(c):
  c.run("rm dtd-r1.0.1.tar.gz")
  c.run("wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", pty=True)
  c.run("tar xf dtd-r1.0.1.tar.gz")

  images = []
  for f in glob("dtd/images/*/*.jpg"):
    images.append(mpimage.imread(f))

  print(f"loaded {len(images)} images")

  with open(BACKGROUNDS_FILE, "wb") as f:
    pickle.dump(images, f)

  print(f"saved to {BACKGROUNDS_FILE}")
