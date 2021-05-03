#!/bin/bash
sshpass -p ni-class9 scp samna_network.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
sshpass -p ni-class9 scp process_aedat.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
sshpass -p ni-class9 scp ctxctl_contrib/*.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code/ctxctl_contrib
sshpass -p ni-class9 scp -r ../data/*.aedat4 sistef@zemo.lan.ini.uzh.ch:/home/sistef/data

sshpass -p ni-class9 ssh sistef@zemo.lan.ini.uzh.ch "
    echo 
    cd code
    python3 samna_network.py
    echo
    "
