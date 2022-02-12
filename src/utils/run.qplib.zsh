f=$1
wdir=$(pwd)
echo $wdir
qapraw=$HOME/workspace/benchmarkings-data/qplib/html/qplib/
qapjson=$wdir/examples/qplib_json
qapname=QPLIB_$f

python utils/parse_qplib_to_json.py --i $qapraw/$qapname.qplib --o $qapjson/$qapname.json
# cd $wdir/src && python -u -m pyqptest.small --r 1 --fpath $wdir/examples/qplib_json/$qapname.json --time_limit 2000 &>$f.grb.log &
cd $wdir/src && python -u -m pyqptest.small --r 10 --fpath $wdir/examples/qplib_json/$qapname.json --time_limit 2000 &>$f.adm.log &
