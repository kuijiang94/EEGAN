import requests
import numpy as np
import os
import shutil
import re
import threading
import time
import sqlite3

def download(*h):
    def add_error(id_):
        sql = '''UPDATE urls SET error = 1 WHERE id = ?;'''
        update_status(sql, id_)

    def add_download(id_):
        sql = '''UPDATE urls SET download = 1 WHERE id = ?;'''
        update_status(sql, id_)

    def update_status(sql, id_):
        con = sqlite3.connect('data/imagenet.db')
        con.execute(sql, (str(id_),))
        con.commit()
        con.close()
        
    for h_ in h:
        id_, parent, object_, seq, url, _, _ = h_
        time.sleep(10)
        try:
            r = requests.get(url, stream=True, timeout=10)
        except:
            print('ERROR')
            add_error(id_)
            continue
        if r.status_code == 200:
            dir_ = os.path.join('data', 'raw', object_)
            if not os.path.exists(dir_):
                os.mkdir(dir_)
            path = os.path.join(dir_, '{}_{}.jpg'.format(parent, seq))
            with open(path, 'wb') as f:
                try:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                except:
                    print('ERROR')
                    add_error(id_)
                    continue
            add_download(id_)
            print(url)
        else:
            print('ERROR')
            add_error(id_)


def get_lists():
    con = sqlite3.connect('data/imagenet.db')
    cur = con.cursor()
    sql = '''SELECT * from urls WHERE download = 0 and error = 0;'''
    cur.execute(sql) 
    lists = cur.fetchall()
    cur.close()
    con.close()
    return lists
    

def missing_teddy():
    ''' "n04399382: teddy, teddy bear" cannot be downloaded. '''
    ''' There is no n04399382 image. '''
    if not os.path.exists('data/raw/n04399382'):
    	os.mkdir('data/raw/n04399382') 


def main()
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')
    
    n_threads = 3
    lists = get_lists()
    x = int(np.ceil(len(lists) / n_threads))
    for i in range(n_threads):
        h = lists[i*x:(i+1)*x]
        th = threading.Thread(target=download, args=h).start()
    missing_teddy()


if __name__ == '__main__':
    main()

