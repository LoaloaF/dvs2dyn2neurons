#!/bin/bash
local_code_dir=/home/loaloa/gdrive/career/nsc_master/NI/project/code
local_output_dir=/home/loaloa/gdrive/career/nsc_master/NI/project/data/vid2e
remote_code_dir=sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
remote_output_dir=sistef@zemo.lan.ini.uzh.ch:/home/sistef/data/vid2e

# sshpass -p  rsync -av --exclude */YOLO_SORT_example/* --info=progress2 "$local_code_dir"/* $remote_code_dir/
# sshpass -p  rsync -av --info=progress2 "$local_output_dir"/* $remote_output_dir
# sshpass -p  rsync -rav --exclude */models/*  --info=progress2 $remote_output_dir/* "$local_output_dir"

sshpass -p ni-class9 rsync -rav --info=progress2 $local_code_dir/* $remote_code_dir
sshpass -p ni-class9 rsync -rav --exclude */simulated_events.npy --exclude */*cached.npy --exclude */*.png --exclude */*.mp4 --info=progress2 $local_output_dir/* $remote_output_dir


# sshpass -p ni-class9 scp samna_network.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
# sshpass -p ni-class9 scp create_stimulus.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
# sshpass -p ni-class9 scp viz_dyn_output.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code
# sshpass -p ni-class9 scp ctxctl_contrib/*.py sistef@zemo.lan.ini.uzh.ch:/home/sistef/code/ctxctl_contrib
# sshpass -p ni-class9 scp -r ../data/*.npy sistef@zemo.lan.ini.uzh.ch:/home/sistef/data

sshpass -p ni-class9 ssh sistef@zemo.lan.ini.uzh.ch "
    cd code
    python3 samna_network.py
    exit
    "

# sshpass -p ni-class9 scp sistef@zemo.lan.ini.uzh.ch:/home/sistef/data/events_*.pkl ./../data/vid2e/

sshpass -p ni-class9 rsync -rav $remote_output_dir/* $local_output_dir
# sshpass -p  rsync -rav --exclude */models/*  --info=progress2 $remote_output_dir/* "$local_output_dir"