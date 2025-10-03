# GPT -1

Trying to train a smaller version of GPT-1

## Setup

```sh
git clone https://github.com/Swastik2442/gpt-minus-1
cd gpt-minus-1
pip3 install -r requirements.txt
```

For Shakespeare Dataset,

```sh
# Prepare the Dataset
cd data/shakespeare
./download.sh
python3 prepare.py
cd ../..

# Train and Generate Samples
python3 train.py
python3 generate.py
```

For Hindi WikiPedia Dataset,

```sh
# Prepare the Dataset
cd data/hiwiki
./download.sh
python3 prepare.py
cd ../..

# Train and Generate Samples
python3 train.py --dataset hiwiki
python3 generate.py out/hiwiki/model_checkpoint_1000.pth
```

> Training can be done with different config parameters. Pass keyword arguments to `python3 train.py` for changing the parameters.
>
> Generation can be done with different checkpoint files. Pass the checkpoint file name to `python3 generate.py` to generate using a different checkpoint file.

### Acknowledgements

* [nanoGPT](https://github.com/karpathy/nanoGPT)
