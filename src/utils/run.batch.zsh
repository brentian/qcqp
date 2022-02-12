data=$1
time_limit=$2
wdir=$(pwd)
echo $wdir
test_module=pyqptest.small

for f in $(ls ${data}/*.json); do
  echo "process: , $f"
  cd $wdir && python -u -m $test_module --r 7 --bg_pr admm --time_limit $time_limit --fpath $f
done
