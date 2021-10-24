python -u -m pyqptest.random_small --n 50 --m 20 --time_limit 200 --r 1,10 &>1.log
python -u -m pyqptest.random_small --n 50 --m 50 --time_limit 200 --r 1,10 &>2.log
python -u -m pyqptest.random_small --n 100 --m 20 --time_limit 400 --r 1,10 &>3.log
python -u -m pyqptest.random_small --n 100 --m 50 --time_limit 400 --r 1,10 &>4.log
python -u -m pyqptest.random_small --n 200 --m 5 --time_limit 800 --r 1,10 &>5.log
python -u -m pyqptest.random_small --n 200 --m 20 --time_limit 800 --r 1,10 &>6.log
