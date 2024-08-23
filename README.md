# uphill
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/yinanxu0/uphill)

Easy to process and store data.

## installation
### install from pip
```
pip3 install uphill
```
### install 
```
git clone https://github.com/yinanxu0/uphill
cd uphill
pip3 install .
```

## Document
We afford python package and bin mode. For more details, please check `uphill -h`. 
```
usage: uphill [-h] [-v] {download,prepare,peek} ...

Uphill Line Interface

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show UpHill version

subcommands:
  use "uphill [sub-command] --help" to get detailed information about each sub-command

  {download,prepare,peek}
    download            ⏬ download a dataset automatically
    prepare             👋 prepare a dataset automatically
    peek                👀 peek a dataset quickly

uphill v0.1.3, a toolkit based on pytorch. Visit https://github.com/yinanxu0/uphill for tutorials and documents.
```
For convenience, you can use `uh` instead of `uphill`, like `uh -h`.

### Download dataset
For example, download Aishell dataset
```
uh download -d aishell -o ${download_dir}
```

More details of parameters in help mode.
```
uh download -h
```

### Prepare dataset
```
uh prepare -d aishell -c ${download_dir}/aishell -o ${data_dir} -j 8 -z
```

More details of parameters in help mode.
```
uh prepare -h
```

### Peek dataset
```
uh peek -i ${data_dir}/supervisions_train.jsonl -n 2
```
