#!/bin/bash

inotifywait -q -m -e close_write -r * --exclude ".log|.png|.*.pyc|.coverage|.pytest_cache|.git|Session.vim"  | while read -r path event filename;
do
    clear
	echo -e "\n\n\033[1;7;37mRunning test because $path$filename was $event\033[0;1;0m"
	pipenv run unittests
done;
