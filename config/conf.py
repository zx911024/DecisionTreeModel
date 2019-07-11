# -*- coding: utf-8 -*-

import os
import configparser as ConfigParser

cfg = ConfigParser.ConfigParser()
BASE_DIR = os.path.dirname(__file__)
cfg.read(os.path.join(BASE_DIR, 'config'))

# mysql wjy_rtc  1.19
db_name = cfg.get("mysql_test", "db_name")
db_user = cfg.get("mysql_test", "db_user")
db_pass = cfg.get("mysql_test", "db_pass")
db_ip = cfg.get("mysql_test", "db_ip")
# online  wjy　1.19
db_name_online = cfg.get("mysql_test1", "db_name")
db_user_online = cfg.get("mysql_test1", "db_user")
db_pass_online = cfg.get("mysql_test1", "db_pass")
db_ip_online = cfg.get("mysql_test1", "db_ip")
# sqlserver
db_name_sqlserver = cfg.get("sqlServer_line", "db_name")
db_user_sqlserver = cfg.get("sqlServer_line", "db_user")
db_pass_sqlserver = cfg.get("sqlServer_line", "db_pass")
db_ip_sqlserver= cfg.get("sqlServer_line", "db_ip")
