# something something cards something something

## install

```sh
poetry install
```

## demo

(recommend `alias i='poetry run invoke'`)

Demonstrate extracting and normalizing the image of a card.

```sh
poetry run invoke demo-extract-image -w 66 -h 120 example/16.png
```

Demonstrate extracting and normalizing images of a card from a video.

```sh
poetry run invoke demo-extract-video -w 66 -h 120 example/16.mov
```

# original README

# playing-card-detection

Generating a dataset of playing cards to train a neural net.

The notebook **creating_playing_cards_dataset.ipynb** is a guide through the creation of a dataset of playing cards. The cards are labeled with their name (ex: "2s" for "2 of spades", "Kh" for King for hearts) and with the bounding boxes delimiting their printed corners.

This dataset can be used for the training of a neural net intended to detect/localize playing cards. It was used on the project **[Playing card detection with YOLO v3](https://youtu.be/pnntrewH0xg)**

<img src="img/ex_generated_image.png" alt="Example of generated image "  title="Example of generated image " />

## install

```sh
poetry install
poetry run jupyter contrib nbextension install --user
```

## execute

```sh
poetry run jupyter notebook creating_playing_cards_dataset.ipynb
```
