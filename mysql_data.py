import time
import logging
import pymysql
from datetime import datetime, timedelta
from utils import str_to_datetime, get_need_dir

########## 用于与Mysql数据库进行交互 ##########
# 日志配置
today = datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(filename=get_need_dir(f'logs/{today}.log'), filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class MysqlData:

    def __init__(self):
        self.host = '127.0.0.1'
        self.user = 'root'
        self.password = 'xzj'
        self.db = 'ai_db'

    # 通用的数据库连接
    def connect(self):
        conn = None
        try:
            # 注意，Redis使用“localhost”会很慢，这里怕速度也慢，所以也用127.0.0.1 #量化专用库密码为：xzj@))%789fvgS
            conn = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.db)
        except Exception as e:
            logging.error(f'数据库连接出现错误：{e}')

        return conn

    def get_record_exist(self, name):
        """判断记录是否存在"""

        # 连接MySQL
        conn = self.connect()
        cursor = conn.cursor()

        # 操作MySQL
        sql = 'select id from vector_record where name = %s'
        cursor.execute(sql, (name,))
        results = cursor.fetchall()

        state = False
        if len(results) > 0:
            state = True

        # 关闭连接
        cursor.close()
        conn.close()

        return state

    def set_vector_record(self, name):
        """新增向量文件记录"""

        # 连接MySQL
        conn = self.connect()
        cursor = conn.cursor()

        # 操作MySQL
        sql = 'insert into vector_record(name, create_time)values(%s, now())'
        results = cursor.execute(sql, (name,))
        if results <= 0:
            print('新增向量文件记录失败！')

        # 因为python是自带事物的，不提交的话就不能添加
        conn.commit()

        # 关闭连接
        cursor.close()
        conn.close()

    def __del__(self):
        print(f"程序主动关闭！")


if __name__ == "__main__":
    mysql_data = MysqlData()
    print(f'状态为：{mysql_data.set_vector_record('食品安全标准管理办法.docx')}！')
