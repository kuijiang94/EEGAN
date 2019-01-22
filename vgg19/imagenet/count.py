import sqlite3

def count(db_path):
    con = sqlite3.connect('data/imagenet.db')
    cur = con.cursor()
    cur.execute(r'SELECT count(*) from urls;')
    all_ = cur.fetchall()[0][0]
    cur.execute(r'SELECT count(*) from urls WHERE download = 1;')
    download = cur.fetchall()[0][0]
    cur.execute(r'SELECT count(*) from urls WHERE error = 1 and download = 0;')
    error = cur.fetchall()[0][0]
    cur.close()
    con.close()
    print("download:", download)
    print("error:", error)
    print("sum:", download+error)
    print("all:", all_)

if __name__ == '__main__':
    count()

