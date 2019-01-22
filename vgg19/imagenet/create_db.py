import os
import re
import sqlite3
from tqdm import tqdm
import sys
sys.path.append('./labels')
import objects

def create_db():
    lists = [
        ['urls/fall11.txt', 'fall11'],
        ['urls/winter11.txt', 'winter11'],
        ['urls/spring10.txt', 'spring10'],
        ['urls/fall09.txt', 'fall09'],
    ]
    target = list(objects.objects.keys())
        
    con = sqlite3.connect('data/imagenet.db')
    sql = '''CREATE TABLE urls (id integer primary key autoincrement,
        parent varchar(255), object varchar(255), seq varchar(255),
        url varchar(65535) unique, download boolean, error boolean);'''
    con.execute(sql)
    
    pattern = r"(.+)_(.+)\t(.+)\n"
    for l in lists:
        txt_path, parent = l
        print(parent)
        with open(txt_path, 'rb') as f:
            x = f.readlines()
        for i, x_ in tqdm(enumerate(x)):
            try: 
                x_ = str(x_, 'utf-8')
            except:
                continue
            matchOB = re.match(pattern, x_)
            object_ = matchOB.group(1)
            seq = matchOB.group(2)
            url = matchOB.group(3)
            if object_ not in target:
                continue
            sql = '''INSERT INTO urls (parent, object, seq, url, 
                     download, error) values (?, ?, ?, ?, ?, ?);'''
            user = (parent, object_, seq, url, 0, 0)
            try:
                con.execute(sql, user)
            except:
                continue
        con.commit()
    con.close()


if __name__ == '__main__':
    create_db()

