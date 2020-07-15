set -e
ps -ef | grep python | grep mayy | awk '{print $2}'  | xargs kill -9