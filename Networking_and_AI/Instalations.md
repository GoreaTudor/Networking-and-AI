## Install everything without issues

On the conda env, install those packages:

```
$ pip install mininet
$ pip install pandas
$ pip install scapy
$ pip install scikit-learn
$ pip install matplotlib
$ pip install tensorflow
```

Note that I didn't use `$ conda install ...` and used
`$ pip install ...`, because that's the only way to
make everything work, and also because some packages
just refuse to install correctly.

---

On the mininet environment, install:

```
$ sudo apt install hping3
$ sudo apt install nmap
```
